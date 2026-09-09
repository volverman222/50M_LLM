"""Zero-shot multiple-choice benchmarks for decoder-only language models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MultipleChoiceExample:
    """A prompt, candidate continuations, and the correct candidate index."""

    context: str
    choices: Sequence[str]
    label: int


def _model_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    output = model(tokens)
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError("model(tokens) must return logits, an object with .logits, or a tuple")


def render_multiple_choice(
    example: MultipleChoiceExample,
    tokenizer: Any,
    *,
    context_length: int,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Tokenize and pad candidate rows, masking only scored continuations."""
    if context_length < 2:
        raise ValueError("context_length must be at least 2")
    if len(example.choices) < 2:
        raise ValueError("A multiple-choice example needs at least two choices")
    if not 0 <= example.label < len(example.choices):
        raise ValueError("The label is outside the candidate range")

    context_tokens = tokenizer.encode(example.context)
    choice_tokens = [tokenizer.encode(choice) for choice in example.choices]
    if not context_tokens:
        raise ValueError("The context produced no tokens")
    if any(not choice for choice in choice_tokens):
        raise ValueError("Every candidate must produce at least one token")

    rows: list[list[int]] = []
    masks: list[list[int]] = []
    for choice in choice_tokens:
        # Retain at least one context token so the first answer token is scored.
        choice = choice[-(context_length - 1) :]
        kept_context = context_tokens[-(context_length - len(choice)) :]
        row = kept_context + choice
        rows.append(row)
        masks.append([0] * len(kept_context) + [1] * len(choice))

    width = max(map(len, rows))
    tokens = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.float32)
    for index, (row, row_mask) in enumerate(zip(rows, masks)):
        tokens[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, : len(row_mask)] = torch.tensor(row_mask)
    return tokens, mask, example.label


def _normalized_completion_losses(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
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


def _context_length(model: torch.nn.Module, value: int | None) -> int:
    if value is None:
        value = getattr(getattr(model, "pos_emb", None), "num_embeddings", None)
    if value is None:
        value = getattr(getattr(model, "config", None), "block_size", None)
    if value is None or value < 2:
        raise ValueError("Pass context_length (at least 2) for this model")
    return value


@torch.inference_mode()
def evaluate_multiple_choice(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device | str,
    examples: Iterable[Mapping[str, Any]],
    adapter: Callable[[Mapping[str, Any]], MultipleChoiceExample],
    *,
    max_examples: int | None = 1_000,
    context_length: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float | int]:
    """Choose the candidate with the lowest mean completion loss."""
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive or None")
    context_length = _context_length(model, context_length)
    target_device = torch.device(device)
    was_training = model.training
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    try:
        for raw_example in examples:
            if max_examples is not None and total >= max_examples:
                break
            tokens, mask, label = render_multiple_choice(
                adapter(raw_example), tokenizer, context_length=context_length
            )
            tokens, mask = tokens.to(target_device), mask.to(target_device)
            enabled = autocast_dtype is not None and target_device.type in {"cuda", "cpu"}
            with torch.autocast(
                device_type=target_device.type,
                dtype=autocast_dtype or torch.float32,
                enabled=enabled,
            ):
                losses = _normalized_completion_losses(_model_logits(model, tokens), tokens, mask)
            correct += int(losses.argmin().item() == label)
            loss_sum += losses[label].float().item()
            total += 1
    finally:
        model.train(was_training)
    if total == 0:
        raise ValueError("Benchmark evaluation received no examples")
    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "mean_correct_completion_loss": loss_sum / total,
    }


def adapt_hellaswag(example: Mapping[str, Any]) -> MultipleChoiceExample:
    return MultipleChoiceExample(
        str(example["ctx"]), [" " + str(x) for x in example["endings"]], int(example["label"])
    )


def adapt_arc_easy(example: Mapping[str, Any]) -> MultipleChoiceExample:
    choices = example["choices"]
    labels = [str(label) for label in choices["label"]]
    answer = str(example["answerKey"])
    try:
        correct = labels.index(answer)
    except ValueError as exc:
        raise ValueError(f"ARC answerKey {answer!r} is absent from choice labels") from exc
    return MultipleChoiceExample(
        f"Question: {example['question']}\nAnswer:",
        [" " + str(text) for text in choices["text"]],
        correct,
    )


def adapt_piqa(example: Mapping[str, Any]) -> MultipleChoiceExample:
    return MultipleChoiceExample(
        f"Question: {example['goal']}\nAnswer:",
        [" " + str(example["sol1"]), " " + str(example["sol2"])],
        int(example["label"]),
    )


def adapt_winogrande(example: Mapping[str, Any]) -> MultipleChoiceExample:
    sentence = str(example["sentence"])
    if "_" not in sentence:
        raise ValueError("WinoGrande sentence must contain the '_' placeholder")
    prefix, suffix = sentence.split("_", 1)
    return MultipleChoiceExample(
        prefix,
        [str(example["option1"]) + suffix, str(example["option2"]) + suffix],
        int(example["answer"]) - 1,
    )


BENCHMARKS: dict[str, tuple[str, str | None, str, Callable[..., MultipleChoiceExample]]] = {
    "hellaswag": ("Rowan/hellaswag", None, "validation", adapt_hellaswag),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "validation", adapt_arc_easy),
    "piqa": ("ybisk/piqa", None, "validation", adapt_piqa),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation", adapt_winogrande),
}


@lru_cache(maxsize=None)
def _load_benchmark_dataset(name: str, split: str) -> Iterable[Mapping[str, Any]]:
    """Load once per process so periodic training evaluation reuses the dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            'Install benchmark dependencies with: pip install -e ".[benchmarks]"'
        ) from exc
    dataset_name, config, _, _ = BENCHMARKS[name]
    return load_dataset(dataset_name, config, split=split)


def evaluate_benchmark(
    name: str,
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device | str,
    *,
    examples: Iterable[Mapping[str, Any]] | None = None,
    split: str | None = None,
    max_examples: int | None = 1_000,
    context_length: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, float | int]:
    """Load and evaluate one supported benchmark, or use injected examples."""
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {sorted(BENCHMARKS)}")
    _, _, default_split, adapter = BENCHMARKS[name]
    if examples is None:
        examples = _load_benchmark_dataset(name, split or default_split)
    return evaluate_multiple_choice(
        model,
        tokenizer,
        device,
        examples,
        adapter,
        max_examples=max_examples,
        context_length=context_length,
        autocast_dtype=autocast_dtype,
    )


def evaluate_benchmark_suite(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device | str,
    *,
    names: Sequence[str] = tuple(BENCHMARKS),
    max_examples: int | None = 1_000,
    context_length: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, dict[str, float | int]]:
    """Evaluate all requested tasks; datasets are cached after their first download."""
    return {
        name: evaluate_benchmark(
            name,
            model,
            tokenizer,
            device,
            max_examples=max_examples,
            context_length=context_length,
            autocast_dtype=autocast_dtype,
        )
        for name in names
    }
