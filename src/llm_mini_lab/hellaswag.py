"""HellaSwag evaluation utilities for decoder-only language models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn.functional as F


def _model_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Extract logits from this project's models and common HF-style outputs."""
    output = model(tokens)
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError("model(tokens) must return logits, an object with .logits, or a tuple")


def render_hellaswag_example(
    example: Mapping[str, Any],
    tokenizer: Any,
    *,
    context_length: int,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Turn one HellaSwag item into padded candidate rows and completion masks."""
    context_tokens = tokenizer.encode(example["ctx"])
    ending_tokens = [tokenizer.encode(" " + ending) for ending in example["endings"]]
    if not context_tokens:
        raise ValueError("HellaSwag context produced no tokens")
    if len(ending_tokens) != 4 or any(not ending for ending in ending_tokens):
        raise ValueError("A HellaSwag example must contain four non-empty endings")

    rows: list[list[int]] = []
    masks: list[list[int]] = []
    for ending in ending_tokens:
        # Keep the whole answer whenever possible and truncate old context first.
        ending = ending[-(context_length - 1):]
        kept_context = context_tokens[-(context_length - len(ending)):]
        row = kept_context + ending
        rows.append(row)
        masks.append([0] * len(kept_context) + [1] * len(ending))

    width = max(map(len, rows))
    tokens = torch.full((4, width), pad_token_id, dtype=torch.long)
    mask = torch.zeros((4, width), dtype=torch.float32)
    for index, (row, row_mask) in enumerate(zip(rows, masks)):
        tokens[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, : len(row_mask)] = torch.tensor(row_mask)

    return tokens, mask, int(example["label"])


def _normalized_completion_losses(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return mean next-token loss over each candidate completion."""
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_tokens = tokens[:, 1:].contiguous()
    shifted_mask = mask[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shifted_logits.flatten(0, 1), shifted_tokens.flatten(), reduction="none"
    ).view(tokens.size(0), -1)
    counts = shifted_mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Every candidate needs at least one scored completion token")
    return (token_losses * shifted_mask).sum(dim=1) / counts


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
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive or None")
    if examples is None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                'Install benchmark dependencies with: pip install -e ".[benchmarks]"'
            ) from exc
        examples = load_dataset("Rowan/hellaswag", split=split)

    if context_length is None:
        context_length = getattr(getattr(model, "pos_emb", None), "num_embeddings", None)
        if context_length is None:
            context_length = getattr(getattr(model, "config", None), "block_size", None)
    if context_length is None or context_length < 2:
        raise ValueError("Pass context_length (at least 2) for this model")

    target_device = torch.device(device)
    device_type = target_device.type
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    try:
        for example in examples:
            if max_examples is not None and total >= max_examples:
                break
            tokens, mask, label = render_hellaswag_example(
                example, tokenizer, context_length=context_length
            )
            tokens = tokens.to(target_device)
            mask = mask.to(target_device)
            autocast_enabled = autocast_dtype is not None and device_type in {"cuda", "cpu"}
            with torch.autocast(
                device_type=device_type,
                dtype=autocast_dtype or torch.float32,
                enabled=autocast_enabled,
            ):
                logits = _model_logits(model, tokens)
                losses = _normalized_completion_losses(logits, tokens, mask)
            correct += int(losses.argmin().item() == label)
            loss_sum += losses[label].float().item()
            total += 1
    finally:
        model.train(was_training)

    if total == 0:
        raise ValueError("HellaSwag evaluation received no examples")
    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "mean_correct_completion_loss": loss_sum / total,
    }

