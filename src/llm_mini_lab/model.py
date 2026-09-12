"""Compatibility imports for model architectures.

Use :mod:`llm_mini_lab.models` in new code.
"""

from .models import GPTModel, LoopedGPTModel, TransformerBlock

__all__ = ["GPTModel", "LoopedGPTModel", "TransformerBlock"]
