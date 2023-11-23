import numpy as np
import torch
from scipy.optimize import minimize


def _build_Ck_shadow(weight_orig, bk):
    k = len(bk)
    Bk = torch.zeros((*weight_orig.shape, k), device=weight_orig.device)
    for i in range(k):
        Bk[:, :, i] = torch.where(weight_orig >= 0., bk[i], -bk[i])
    Ck_shadow = torch.zeros((*weight_orig.shape, k), device=weight_orig.device)

    # construct Ck_shadow
    R = torch.clone(weight_orig)
    for i in range(k - 1):
        Ck_shadow[:, :, i] = torch.floor(R / Bk[:, :, i])
        R = R % Bk[:, :, i]
    Ck_shadow[:, :, -1] = R / Bk[:, :, -1]
    return Ck_shadow, Bk


def _reconstruct_weights(Ck_shadow, Bk):
    Ck = torch.round(Ck_shadow)
    weight_k = (Ck * Bk).sum(dim=2)
    return weight_k, Ck


def _build_cost_function(weight_orig, small_weight):
    def cost_function(bk, alpha=0.5):
        bk = np.concatenate([bk, [small_weight]])
        Ck_shadow, Bk = _build_Ck_shadow(weight_orig, bk)
        weight_k, Ck = _reconstruct_weights(Ck_shadow, Bk)

        mae = (weight_k - weight_orig).abs().median()
        C_full_fan_in = torch.round(weight_orig / Bk[:, :, -1]).sum(dim=1)
        C_fan_in = Ck.sum(dim=(1, 2))
        C_ratio = (C_fan_in / C_full_fan_in).mean()
        loss = alpha * mae + (1 - alpha) * C_ratio
        return loss.item()

    return cost_function


def optimize_large_baseweight(weight, small_weight, upper_bound=1.0, alpha=0.5):
    cost_function = _build_cost_function(weight, small_weight)
    opt = minimize(
        fun=cost_function,
        x0=np.array([small_weight]),
        args=(alpha,),
        method='nelder-mead',
        bounds=[(small_weight, upper_bound)]
    )
    large_weight = opt.x[0]
    return large_weight
