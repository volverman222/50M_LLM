import torch
import torch.nn as nn


class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

""""
Deprecrated
class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
"""
class SwiGLU(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(emb_dim, hidden_dim)
        self.value_proj = nn.Linear(emb_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        value = self.value_proj(x)
        return self.out_proj(gate * value)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        activation = cfg.get("ff_activation", "gelu").lower()
        emb_dim = cfg["emb_dim"]

        if activation == "gelu":
            hidden_dim = cfg.get("ff_hidden_dim", 4 * emb_dim)

            self.layers = nn.Sequential(
                nn.Linear(emb_dim, hidden_dim),
                GELU(),
                nn.Linear(hidden_dim, emb_dim),
            )

        elif activation == "swiglu":
            # Aproximadamente los mismos parámetros que el FFN GELU 4x.
            hidden_dim = cfg.get(
                "ff_hidden_dim",
                int(8 * emb_dim / 3),
            )

            self.layers = SwiGLU(
                emb_dim=emb_dim,
                hidden_dim=hidden_dim,
            )

        else:
            raise ValueError(
                f"Activación FFN no soportada: {activation!r}. "
                "Utiliza 'gelu' o 'swiglu'."
            )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):

    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Keep the reduction in FP32 when the surrounding model uses AMP/FP16.
        # This prevents variance overflow in the custom LayerNorm implementation.
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        var = x_float.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x_float - mean) / torch.sqrt(var + self.eps)
        return (self.scale.float() * norm_x + self.shift.float()).to(dtype=x.dtype)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # QK^T and softmax are the numerically sensitive operations in FP16.
        # Compute them in FP32, then return to the AMP dtype for the output layer.
        with torch.autocast(device_type=x.device.type, enabled=False):
            attn_scores = queries.float() @ keys.float().transpose(2, 3)
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)
            context_vec = attn_weights @ values.float()

        context_vec = context_vec.to(dtype=x.dtype).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
