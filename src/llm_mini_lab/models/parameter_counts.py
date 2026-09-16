"""Helpers for reporting parameter counts in the Transformer models."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def _trainable_parameter_count(module: nn.Module) -> int:
    """Return the number of trainable scalar parameters in ``module``."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def count_transformer_parameters(model: nn.Module) -> dict[str, Any]:
    """Count attention and feed-forward parameters in every unique Transformer block.

    ``LoopedGPTModel`` reuses its blocks across loops.  This function iterates over
    ``model.trf_blocks`` only once, so shared weights are not counted repeatedly.
    """
    if not hasattr(model, "trf_blocks"):
        raise TypeError("El modelo debe definir el atributo 'trf_blocks'.")

    attention_per_block: list[int] = []
    feedforward_per_block: list[int] = []

    for index, block in enumerate(model.trf_blocks):
        if not hasattr(block, "att") or not hasattr(block, "ff"):
            raise TypeError(
                f"El bloque {index} debe definir los módulos 'att' y 'ff'."
            )
        attention_per_block.append(_trainable_parameter_count(block.att))
        feedforward_per_block.append(_trainable_parameter_count(block.ff))

    return {
        "attention_per_block": attention_per_block,
        "feedforward_per_block": feedforward_per_block,
        "attention_total": sum(attention_per_block),
        "feedforward_total": sum(feedforward_per_block),
    }
