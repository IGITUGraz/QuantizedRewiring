import math

import torch
from torch.nn.init import calculate_gain


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError('cannot compute fan in: too low dimensionality')
    fan_in = torch.mean(torch.sum(tensor, dim=list(range(1, tensor.dim()))).float())
    return fan_in


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    if mode != 'fan_in':
        raise ValueError("Mode {} not supported.")

    fan_in = _calculate_fan_in_and_fan_out(tensor)
    return fan_in


def kaiming_uniform_sparse_(tensor, sparsity, a=0, mode='fan_in', nonlinearity='relu'):
    if 0 in tensor.shape:
        return tensor

    mask = torch.zeros(tensor.shape, dtype=torch.bool)
    mask_flat = mask.flatten()
    K = int((sparsity) * mask_flat.shape[0])
    indices = torch.randperm(mask_flat.shape[0])[:K]
    mask_flat[indices] = True
    mask = torch.reshape(mask_flat, mask.shape)

    fan = _calculate_correct_fan(mask, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / torch.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        out = tensor.uniform_(-bound, bound)
        out[~mask] = 0.
        return out


def kaiming_uniform_other_(tensor: torch.Tensor, other, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = torch.nn.init._calculate_correct_fan(other, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
