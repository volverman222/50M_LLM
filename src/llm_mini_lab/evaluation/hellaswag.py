"""HellaSwag API built on the generic benchmark evaluator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from .suite import (
    adapt_hellaswag,
    evaluate_benchmark,
    render_multiple_choice,
)


def render_hellaswag_example(
    example: Mapping[str, Any],
    tokenizer: Any,
    *,
    context_length: int,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Turn one HellaSwag item into padded candidate rows and completion masks."""
    adapted = adapt_hellaswag(example)
    if len(adapted.choices) != 4:
        raise ValueError("A HellaSwag example must contain four non-empty endings")
    return render_multiple_choice(
        adapted, tokenizer, context_length=context_length, pad_token_id=pad_token_id
    )


@torch.inference_mode()
def evaluate_hellaswag(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device | str,
    *,
    examples: Iterable[Mapping[str, Any]] | None = None,
    split: str = "validation",
    max_examples: int | None = 1_000,
    context_length: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float | int]:
    """Evaluate HellaSwag and return accuracy/counter metrics.

    If ``examples`` is omitted, ``Rowan/hellaswag`` is loaded with the optional
    ``datasets`` dependency. Accuracy is based on the answer whose completion
    has the lowest mean autoregressive loss.
    """
    return evaluate_benchmark(
        "hellaswag", model, tokenizer, device, examples=examples, split=split,
        max_examples=max_examples, context_length=context_length,
        autocast_dtype=autocast_dtype,
    )
