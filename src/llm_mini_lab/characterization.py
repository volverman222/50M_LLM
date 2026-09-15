"""Passive feed-forward boundary observations; not an architecture variant."""

import torch


def tensor_record(value):
    x = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "all_finite": bool(torch.isfinite(x).all()),
        "mean": x.mean().item(),
        "std": x.std(unbiased=False).item(),
        "abs_max": x.abs().max().item(),
    }


@torch.no_grad()
def gate_record(value):
    x = value.detach().float()
    activation = torch.nn.functional.silu(x)
    sigmoid = torch.sigmoid(x)
    return {
        "ff_hidden_width": value.shape[-1],
        "gate_activation_rms": activation.square().mean().sqrt().item(),
        "gate_negative_fraction": (x < 0).float().mean().item(),
        "gate_sigmoid_extreme_fraction": ((sigmoid < 0.01) | (sigmoid > 0.99))
        .float()
        .mean()
        .item(),
    }


def rms(value):
    return value.detach().float().square().mean().sqrt().item()
