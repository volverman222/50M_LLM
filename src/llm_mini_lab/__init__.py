"""Educational utilities for building and evaluating small GPT models."""

from .model import GPTModel, TransformerBlock
from .hellaswag import evaluate_hellaswag, render_hellaswag_example
from .benchmarks import evaluate_benchmark, evaluate_benchmark_suite

__all__ = [
    "GPTModel",
    "TransformerBlock",
    "evaluate_hellaswag",
    "render_hellaswag_example",
    "evaluate_benchmark",
    "evaluate_benchmark_suite",
]
