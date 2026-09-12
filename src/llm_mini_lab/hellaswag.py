"""Compatibility imports for HellaSwag evaluation.

Use :mod:`llm_mini_lab.evaluation.hellaswag` in new code.
"""

from .evaluation.hellaswag import evaluate_hellaswag, render_hellaswag_example

__all__ = ["evaluate_hellaswag", "render_hellaswag_example"]
