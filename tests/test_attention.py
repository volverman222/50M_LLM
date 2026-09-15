import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from llm_mini_lab.models.layers import MultiHeadAttention


@pytest.mark.parametrize("num_kv_heads", [8, 2])
def test_sdpa_attention_shape_and_gradients(num_kv_heads):
    attention = MultiHeadAttention(
        d_in=64,
        d_out=64,
        context_length=16,
        dropout=0.0,
        num_heads=8,
        num_kv_heads=num_kv_heads,
    )
    inputs = torch.randn(2, 12, 64, requires_grad=True)

    output = attention(inputs)
    output.sum().backward()

    assert output.shape == inputs.shape
    assert inputs.grad is not None
    assert torch.isfinite(output).all()
    assert torch.isfinite(inputs.grad).all()


def test_gqa_uses_compact_key_and_value_projections():
    attention = MultiHeadAttention(
        d_in=64,
        d_out=64,
        context_length=16,
        dropout=0.0,
        num_heads=8,
        num_kv_heads=2,
    )

    assert attention.W_query.out_features == 64
    assert attention.W_key.out_features == 16
    assert attention.W_value.out_features == 16


def test_num_kv_heads_must_divide_query_heads():
    with pytest.raises(ValueError, match="divide num_heads exactly"):
        MultiHeadAttention(
            d_in=64,
            d_out=64,
            context_length=16,
            dropout=0.0,
            num_heads=8,
            num_kv_heads=3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gqa_runs_with_flash_attention_on_cuda():
    attention = MultiHeadAttention(
        d_in=64,
        d_out=64,
        context_length=128,
        dropout=0.0,
        num_heads=8,
        num_kv_heads=2,
    ).cuda().half()
    inputs = torch.randn(2, 128, 64, device="cuda", dtype=torch.float16)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = attention(inputs)

    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
