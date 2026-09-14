"""Before/after benchmark for the fused attention: tokens/s, ms/update and peak VRAM of a full training step
(forward + backward + AdamW) of LoopedGPTModel at a given context length, explicit-attention vs fused.

    python scripts/bench_attention.py --config student --impl naive,fused --micro-batch 8 --context 1024
    python scripts/bench_attention.py --config start12 --impl naive,fused --micro-batch 8 --context 1024 --json out.json

--config student = the submitted architecture (d1024, 3 unique layers x 2 loops, learned positions);
--config start12 = d512, 12 unique layers, RoPE. Both: sp16384 vocabulary, SwiGLU 1376, tied output head.
Mixed precision follows the trainer: bf16 on Ampere+ (RTX 30xx/40xx/50xx, A100, H100), fp16 + GradScaler otherwise (V100).
"""
from __future__ import annotations
import argparse, json, time
import torch, torch.nn.functional as F
from llm_mini_lab.models import LoopedGPTModel
from llm_mini_lab.models.layers import MultiHeadAttention
from llm_mini_lab.training.core import LOOPED_GPT_CONFIG, init_xavier

CONFIGS = {
    "student": dict(emb_dim=1024, n_unique_layers=3, num_loops=2, positional_encoding="learned"),
    "start12": dict(emb_dim=512, n_unique_layers=12, num_loops=1, positional_encoding="rope"),
    "default": {},
}


def build(config, context):
    cfg = {**LOOPED_GPT_CONFIG, **CONFIGS[config], "vocab_size": 16384, "context_length": context, "tokenizer_name": "sp16384"}
    torch.manual_seed(0)
    m = LoopedGPTModel(cfg); m.apply(init_xavier); m.out_head.weight = m.tok_emb.weight
    n = sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())
    return m, cfg, n


def bench(config, impl, micro_batch, context, steps, warm, device):
    fused_forward = MultiHeadAttention.forward
    MultiHeadAttention.forward = MultiHeadAttention.forward_reference if impl == "naive" else fused_forward
    try:
        m, cfg, n = build(config, context); m = m.to(device).train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
        cap = torch.cuda.get_device_capability()[0] if device == "cuda" else 0
        amp_dtype = torch.bfloat16 if cap >= 8 else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and amp_dtype == torch.float16))
        x = torch.randint(0, cfg["vocab_size"], (micro_batch, context + 1), device=device)
        if device == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                logits = m(x[:, :-1])
                loss = F.cross_entropy(logits.flatten(0, 1).float(), x[:, 1:].flatten())
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        for _ in range(warm): step()
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps): step()
        if device == "cuda": torch.cuda.synchronize()
        dt = time.time() - t0
        r = dict(config=config, impl=impl, params=n, micro_batch=micro_batch, context=context, steps=steps,
                 amp=str(amp_dtype).replace("torch.", ""), tok_s=micro_batch * context * steps / dt, ms_per_update=1e3 * dt / steps,
                 peak_alloc_gb=torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
                 peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9 if device == "cuda" else None,
                 device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu")
        del m, opt; 
        if device == "cuda": torch.cuda.empty_cache()
        return r
    finally:
        MultiHeadAttention.forward = fused_forward


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="student,start12"); ap.add_argument("--impl", default="naive,fused")
    ap.add_argument("--micro-batch", type=int, default=8); ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20); ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--json", default=None)
    a = ap.parse_args(); rows = []
    print(f"{'config':<8} {'impl':<6} {'params':>11} {'mb':>3} {'ctx':>5} {'amp':>8} {'tok/s':>9} {'ms/upd':>8} {'peak GB':>8}")
    for c in a.config.split(","):
        for impl in a.impl.split(","):
            try:
                r = bench(c, impl, a.micro_batch, a.context, a.steps, a.warm, a.device); rows.append(r)
                print(f"{c:<8} {impl:<6} {r['params']:>11,} {r['micro_batch']:>3} {r['context']:>5} {r['amp']:>8} {r['tok_s']:>9,.0f} {r['ms_per_update']:>8.0f} {r['peak_alloc_gb'] or 0:>8.2f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                rows.append(dict(config=c, impl=impl, micro_batch=a.micro_batch, context=a.context, error="OOM")); print(f"{c:<8} {impl:<6} OOM", flush=True); torch.cuda.empty_cache()
    if a.json: json.dump(rows, open(a.json, "w"), indent=1); print("→", a.json)
