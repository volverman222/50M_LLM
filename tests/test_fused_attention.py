"""The fused causal attention (torch.nn.functional.scaled_dot_product_attention) must reproduce the previous
explicit implementation, which is kept as MultiHeadAttention.forward_reference.
Run: pytest tests/test_fused_attention.py -v   (CPU part is strict; the CUDA part runs only if a GPU is present)."""
import pytest
import torch

from llm_mini_lab.models.layers import MultiHeadAttention


def _pair(rope, d=512, heads=8, ctx=1024, seed=1):
    torch.manual_seed(seed)
    m = MultiHeadAttention(d, d, ctx, dropout=0.0, num_heads=heads, use_rope=rope)
    return m


@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("tokens", [37, 300, 1024])
def test_fused_matches_reference_cpu_fp32(rope, tokens):
    m = _pair(rope)
    x = torch.randn(2, tokens, 512)
    xa = x.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    ya = m.forward_reference(xa)
    yb = m(xb)
    assert torch.allclose(ya, yb, atol=1e-5, rtol=1e-5), (ya - yb).abs().max()
    ya.sum().backward()
    ga_x, ga_w = xa.grad.clone(), m.W_query.weight.grad.clone()
    m.zero_grad()
    yb.sum().backward()
    assert torch.allclose(ga_x, xb.grad, atol=1e-4, rtol=1e-4), (ga_x - xb.grad).abs().max()
    assert torch.allclose(ga_w, m.W_query.weight.grad, atol=1e-4, rtol=1e-4)


def test_causality():
    """A token's output must not depend on later tokens (the fused kernel is called with is_causal=True)."""
    m = _pair(True)
    x = torch.randn(1, 64, 512)
    y = m(x)
    x2 = x.clone(); x2[:, 40:] = torch.randn(1, 24, 512)   # perturb the future only
    y2 = m(x2)
    assert torch.allclose(y[:, :40], y2[:, :40], atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rope", [False, True])
def test_fused_matches_reference_cuda_mixed_precision(rope):
    """Under autocast the reference computes the scores in fp32 and the fused kernel in fp16/bf16 with fp32
    accumulation, so agreement is to mixed-precision tolerance, not bit-exact."""
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    m = _pair(rope).cuda()
    x = torch.randn(2, 1024, 512, device="cuda")
    with torch.autocast("cuda", dtype=dtype):
        ya = m.forward_reference(x).float()
        yb = m(x).float()
    rel = (ya - yb).norm() / ya.norm()
    assert rel < 2e-2, rel.item()
