"""Educational utilities for building and evaluating small GPT models."""

from .models import GPTModel, TransformerBlock
from .evaluation import (
    evaluate_benchmark,
    evaluate_benchmark_suite,
    evaluate_hellaswag,
    render_hellaswag_example,
)

__all__ = [
    "GPTModel",
    "TransformerBlock",
    "evaluate_hellaswag",
    "render_hellaswag_example",
    "evaluate_benchmark",
    "evaluate_benchmark_suite",
]
