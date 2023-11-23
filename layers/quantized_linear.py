import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.pruning import random_prune_connectivity_inplace, l1_prune_connectivity
from layers.utils import no_op
from utils.rounding import StochasticRounding


def random_removal_strategy(module):
    C = torch.round(module.C_shadow.data)
    max_fan_in = module.rewiring_config['max_fan_in']
    fan_in = torch.sum(C, dim=1, keepdim=True)
    remove = torch.clip(fan_in - max_fan_in, 0, None)
    split = torch.where(C > 0, torch.rand_like(C), torch.zeros_like(C))
    split = split / (split.sum(dim=1, keepdim=True) + 1e-4)
    module.C_shadow.data -= split * remove


def random_encouragement_strategy(module):
    # find number of connections to add
    C = torch.round(module.C_shadow.data)
    min_nonzero_connections = module.rewiring_config['min_nonzero_connections']
    min_nonzero_connections = min(min_nonzero_connections, C.shape[1])
    nonzero_connections = torch.sum(C > 0, dim=1, keepdim=True)
    add = torch.clip(min_nonzero_connections - nonzero_connections, 0, None)

    # check if adding connections would violate fan_in constraint
    max_fan_in = module.rewiring_config['max_fan_in']
    fan_in = torch.sum(C, dim=1, keepdim=True)
    remove = torch.clip((fan_in + add) - max_fan_in, 0, None)
    split = torch.where(C > 0, torch.rand_like(C), torch.zeros_like(C))
    split = split / (split.sum(dim=1, keepdim=True) + 1e-4)
    module.C_shadow.data -= split * remove

    # add connections randomly in accordance with fan_in constraint
    split = torch.where(add > 0, torch.rand_like(C), torch.zeros_like(C))
    split = split / (split.sum(dim=1, keepdim=True) + 1e-4)
    module.C_shadow.data += split * add


def sparsity_encouragement_strategy(module):
    target_connectivity = module.rewiring_config['target_connectivity']


def l1_sparsity_removal_strategy(module):
    target_connectivity = module.rewiring_config['target_connectivity']
    module.C_shadow.data = l1_prune_connectivity(module.C_shadow.data, target_connectivity)


def basic_random_walk(module, grad):
    lr = module.rewiring_config['random_walk_config']['get_lr']()
    temperature = module.rewiring_config['random_walk_config']['temperature']
    noise = math.sqrt(2 * lr * temperature) * torch.randn_like(grad)
    zeros = torch.zeros_like(grad)
    noise = torch.where(grad != zeros, noise, zeros)
    return grad + noise


def forward_pre_hook(module, input):
    module.B = torch.where(module.C_shadow < 0, module.B * (-1), module.B)
    module.C_shadow.data = torch.clamp(module.C_shadow, 0)

    module.removal_strategy(module)
    module.encouragement_strategy(module)

    if module.training:
        module.C.data = StochasticRounding.apply(module.C_shadow.data)
    else:
        module.C.data = torch.round(module.C_shadow.data)

    module.weight = module.C * module.B


ENCOURAGEMENT_STRATEGIES = {
    'random_encouragement': random_encouragement_strategy,
    'none': no_op,
}

REMOVAL_STRATEGIES = {
    'random_removal': random_removal_strategy,
    'l1_sparsity_removal': l1_sparsity_removal_strategy,
    'none': no_op,
}

RANDOM_WALK_STRATEGIES = {
    'basic_random_walk': basic_random_walk,
    'none': None,
}


class QLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, rewiring_config: Dict, bias: bool = True,
                 device=None, dtype=None) -> None:
        super(QLinear, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.rewiring_config = rewiring_config
        self.removal_strategy = REMOVAL_STRATEGIES[rewiring_config['removal_strategy']]
        self.encouragement_strategy = ENCOURAGEMENT_STRATEGIES[rewiring_config['encouragement_strategy']]
        self.random_walk = RANDOM_WALK_STRATEGIES[rewiring_config['random_walk_strategy']]
        self.register_buffer('B', torch.empty((out_features, in_features), **factory_kwargs, requires_grad=False))
        self.C = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs, requires_grad=True))
        self.C_shadow = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs, requires_grad=False))
        self.S = torch.empty((out_features, in_features), **factory_kwargs, requires_grad=False)

        def update_grad(grad):
            if self.random_walk is not None:
                grad = self.random_walk(self, grad)
            grad = grad / torch.abs(self.B) ** 2
            self.C_shadow.grad = grad
            return grad

        self.C.register_hook(update_grad)

        self.weight_orig = torch.empty((out_features, in_features), **factory_kwargs)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.register_forward_pre_hook(forward_pre_hook)

    def reset_parameters(self, weight=None, bias=None) -> None:
        if weight is not None:
            self.weight_orig.data.copy_(weight)
        else:
            torch.nn.init.kaiming_uniform_(self.weight_orig, a=math.sqrt(5))
            if 'weight_gain' in self.rewiring_config:
                weight_gain = self.rewiring_config['weight_gain']
                self.weight_orig = self.weight_orig * weight_gain
            if 'target_connectivity' in self.rewiring_config:
                target_connectivity = self.rewiring_config['target_connectivity']
                if 'initial_pruning_method' in self.rewiring_config:
                    initial_pruning_method = self.rewiring_config['initial_pruning_method']
                    if initial_pruning_method == 'random':
                        random_prune_connectivity_inplace(self.weight_orig, target_connectivity)
                    else:
                        raise NotImplementedError('requested pruning method not implemented')
                else:
                    random_prune_connectivity_inplace(self.weight_orig, target_connectivity)
        if self.bias is not None:
            if bias is not None:
                self.bias.data.copy_(bias)
            else:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_orig)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.bias, -bound, bound)
        self.S = torch.sign(self.weight_orig)
        excitatory = self.rewiring_config['base_weights']['excitatory']
        inhibitory = self.rewiring_config['base_weights']['inhibitory']
        self.B = torch.where(self.weight_orig >= 0, excitatory, inhibitory)
        self.C_shadow.data.copy_(self.weight_orig / self.B)
        self.C.data = torch.round(self.C_shadow.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def compute_fan_in(self):
        return torch.sum(torch.round(self.C_shadow.data), dim=1)

    def compute_fan_in_stats(self):
        fan_in = self.compute_fan_in().detach().cpu().numpy()
        mean, std = np.mean(fan_in), np.std(fan_in)
        min, max = np.min(fan_in), np.max(fan_in)

        qs = [0.25, 0.5, 0.75]
        qlabel = ['Q1', 'Q2', 'Q3']
        quantiles = np.quantile(fan_in, qs)
        qdict = dict(zip(qlabel, quantiles))
        iqr = quantiles[2] - quantiles[0]
        connectivity = self.compute_connectivity_stats()

        return {'mean': mean, 'std': std, 'min': min, 'max': max, **qdict, 'IQR': iqr, 'connectivity': connectivity}

    def compute_connectivity_stats(self):
        num_nonzero = torch.count_nonzero(torch.round(self.C_shadow.data) > 0).detach().cpu().numpy()
        size = float(np.prod(list(self.C_shadow.data.size())))
        return num_nonzero / size

    def compute_fan_in_stats_list(self):
        stats = self.compute_fan_in_stats()
        return list(stats.items())

    def compute_nonzero_connections(self):
        return torch.sum(torch.round(self.C_shadow.data) > 0, dim=1)


class QLinearDelay(QLinear):
    def __init__(self, in_features: int, out_features: int, rewiring_config: Dict, bias: bool = True, device=None,
                 dtype=None) -> None:
        super().__init__(in_features, out_features, rewiring_config, bias, device, dtype)
        assert 'max_delay' in rewiring_config.keys()
        self.register_buffer('delay', torch.randint(1, rewiring_config['max_delay'], (in_features, out_features)))
        self.max_delay = rewiring_config['max_delay']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_delayed = torch.where(
            torch.nn.functional.one_hot(self.delay, self.max_delay) == 1.,
            self.weight[:, :, None],
            torch.zeros_like(self.weight[:, :, None])
        )
        out = torch.einsum('bij,ijk->bjk', x, weight_delayed)
        if self.bias is not None:
            out = out + self.bias
        return out
