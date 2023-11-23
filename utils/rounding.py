from typing import Any

import torch


class StochasticRounding(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        rand = torch.rand(*input.shape, device=input.device)
        prob = (input - torch.floor(input))
        return torch.where(prob > rand, torch.floor(input) + 1, torch.floor(input))

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> torch.Tensor:
        return grad_outputs

