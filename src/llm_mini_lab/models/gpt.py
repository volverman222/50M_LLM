import torch
import torch.nn as nn

from .layers import FeedForward, LayerNorm, MultiHeadAttention


def _positional_encoding(cfg):
    """Return the configured positional encoding, preserving legacy defaults."""
    encoding = cfg.get("positional_encoding", "learned").lower()
    if encoding not in {"learned", "rope"}:
        raise ValueError(
            f"Codificación posicional no soportada: {encoding!r}. "
            "Utiliza 'learned' o 'rope'."
        )
    return encoding


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        positional_encoding = _positional_encoding(cfg)
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            use_rope=positional_encoding == "rope",
            rope_base=cfg.get("rope_base", 10_000))
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.context_length = cfg["context_length"]
        self.positional_encoding = _positional_encoding(cfg)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        if self.positional_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        if self.positional_encoding == "learned":
            positions = torch.arange(seq_len, device=in_idx.device)
            x = tok_embeds + self.pos_emb(positions)
        else:
            x = tok_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class LoopedGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.context_length = cfg["context_length"]
        self.positional_encoding = _positional_encoding(cfg)
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"],
            cfg["emb_dim"],
        )
        if self.positional_encoding == "learned":
            self.pos_emb = nn.Embedding(
                cfg["context_length"],
                cfg["emb_dim"],
            )
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.ModuleList([
            TransformerBlock(cfg)
            for _ in range(cfg["n_unique_layers"])
        ])

        self.num_loops = cfg["num_loops"]

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False,
        )

    def forward(self, in_idx, num_loops=None):
        _, seq_len = in_idx.shape

        x = self.tok_emb(in_idx)
        if self.positional_encoding == "learned":
            positions = torch.arange(seq_len, device=in_idx.device)
            x = x + self.pos_emb(positions)
        x = self.drop_emb(x)

        loops = self.num_loops if num_loops is None else num_loops

        if loops < 1:
            raise ValueError("num_loops debe ser al menos 1")

        for _ in range(loops):
            for block in self.trf_blocks:
                x = block(x)

        x = self.final_norm(x)
        return self.out_head(x)
