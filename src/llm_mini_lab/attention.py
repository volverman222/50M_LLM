"""Compatibility imports for Transformer layers.

Use :mod:`llm_mini_lab.models.layers` in new code.
"""

from .models.layers import FeedForward, GELU, LayerNorm, MultiHeadAttention, SwiGLU

__all__ = ["FeedForward", "GELU", "LayerNorm", "MultiHeadAttention", "SwiGLU"]
