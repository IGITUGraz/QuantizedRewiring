from typing import Tuple

import numpy as np
import torch


def unravel_index(
        indices: torch.LongTensor,
        shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices / dim

    return coord.flip(-1)


def l1_prune_connectivity_inplace(a: torch.Tensor, target_connectivity: float):
    size = np.prod(a.shape)
    current_connectivity = a.count_nonzero().item() / size
    if current_connectivity > target_connectivity:
        shape = a.shape
        a = torch.flatten(a)
        delta_connectivity = current_connectivity - target_connectivity
        delta_connections = int(np.ceil(delta_connectivity * size))
        base = torch.abs(a)
        base = torch.where(base == 0, torch.tensor(torch.inf, dtype=base.dtype, device=base.device), base)
        _, indices = torch.topk(base, delta_connections, largest=False)
        a[indices] = 0.
        return torch.reshape(a, shape)
    return a


def l1_prune_connectivity(a: torch.Tensor, target_connectivity: float):
    # out = torch.clone(a)
    return l1_prune_connectivity_inplace(a, target_connectivity)


def l1_prune_connectivity_multi(a: torch.Tensor, target_connectivity: float):
    size = np.prod(a.shape[:-1])
    asum = a.sum(-1)
    current_connectivity = asum.count_nonzero().item() / size
    if current_connectivity > target_connectivity:
        shape = a.shape
        asum = torch.flatten(asum)
        delta_connectivity = current_connectivity - target_connectivity
        delta_connections = int(np.ceil(delta_connectivity * size))
        base = torch.abs(asum)
        base = torch.where(base == 0, torch.tensor(torch.inf, dtype=base.dtype, device=base.device), base)
        _, indices = torch.topk(base, delta_connections, largest=False)
        u_indices = unravel_index(indices, a.shape[:-1])
        a[u_indices[:, 0], u_indices[:, 1], :] = 0.
        return torch.reshape(a, shape)
    return a


def explore_target_connectivity_multi(a: torch.Tensor, target_connectivity, generate):
    size = np.prod(a.shape)
    current_connectivity = a.count_nonzero().item() / size
    if current_connectivity < target_connectivity:
        shape = a.shape
        asum = torch.flatten(a)
        delta_connectivity = target_connectivity - current_connectivity
        delta_connections = int(np.round(delta_connectivity * size))
        base = torch.rand_like(asum)
        base[asum != 0] = 0
        _, indices = torch.topk(base, delta_connections)
        u_indices = unravel_index(indices, a.shape)
        temp = generate(a[u_indices[:, 0], u_indices[:, 1]])
        a[u_indices[:, 0], u_indices[:, 1]] = temp
        return torch.reshape(a, shape)
    return a


def l1_prune_connections(C: torch.Tensor, max_fan_in: int):
    values, indices = torch.sort(C, dim=1, descending=True)
    values_csum = torch.cumsum(values, dim=1)
    values_clamped = torch.clamp(values_csum - max_fan_in, 0, None)
    out = torch.clamp(values - values_clamped, 0, None)
    out = out.gather(1, indices.argsort(1))
    return out


def random_prune_connectivity_inplace(a: torch.Tensor, target_connectivity: float):
    size = int(np.prod(a.shape))
    current_connectivity = a.count_nonzero() / size
    if current_connectivity > target_connectivity:
        shape = a.shape
        a = torch.flatten(a)
        delta_connectivity = current_connectivity - target_connectivity
        delta_connections = int(np.round(delta_connectivity * size))
        _, indices = torch.topk(torch.rand_like(a), delta_connections)
        a[indices] = 0.
        a = torch.reshape(a, shape)
    return a


def random_prune_connectivity(a: torch.Tensor, target_connectivity: float):
    out = torch.clone(a)
    random_prune_connectivity_inplace(out, target_connectivity)
    return out
