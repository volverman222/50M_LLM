#!/usr/bin/env python3
"""Official evaluation of the Looped GPT checkpoint.

* HellaSwag, ARC-Easy, PIQA and WinoGrande: lm-evaluation-harness (pinned to
  0.4.9.1), zero-shot, full evaluation splits.
* Held-out perplexity: WikiText-103 test split (Salesforce/wikitext,
  wikitext-103-raw-v1, pinned revision), following
  configs/data_protocols/wikitext103_split_reference.json: SentencePiece 16K,
  one EOS after each text row, rows packed into non-overlapping 128-token
  windows, tail dropped. Token-level perplexity is reported together with the
  tokenizer-independent word perplexity and bits per byte.

The harness' default ``wikitext`` task (wikitext-2-raw-v1) is intentionally
not used, as required by configs/evaluation/official_eval_plan.json.

Usage::

    pip install -e ".[eval]"
    python scripts/eval_lm_harness.py --checkpoint path/to/checkpoint-5.0B.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import sentencepiece as spm
from lm_eval.evaluator import simple_evaluate
from lm_eval.api.model import LM
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

from llm_mini_lab.models import LoopedGPTModel

TASKS = ["hellaswag", "arc_easy", "piqa", "winogrande"]
WIKITEXT = ("Salesforce/wikitext", "wikitext-103-raw-v1", "b08601e04326c79dfdd32d625aee71d232d685c3")


class LoopedGPTLM(LM):
    """Minimal lm-eval wrapper: log-likelihood scoring with a 128-token window."""

    def __init__(self, model, tokenizer, device, context_length, batch_size=64):
        super().__init__()
        self.model, self.tok, self.device = model, tokenizer, device
        self.ctx = context_length
        self.batch_size = batch_size
        bos = tokenizer.bos_id()
        self.prefix_token = bos if bos >= 0 else tokenizer.eos_id()

    def encode(self, text):
        return self.tok.encode(text, out_type=int)

    def encode_pair(self, context, continuation):
        # Same convention as lm-eval's HFLM: move trailing spaces into the continuation.
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole = self.encode(context + continuation)
        ctx = self.encode(context) if context else [self.prefix_token]
        if context and whole[: len(ctx)] == ctx and len(whole) > len(ctx):
            return ctx, whole[len(ctx):]
        return ctx, self.encode(continuation)

    @torch.inference_mode()
    def score(self, pairs):
        """pairs: list of (context_ids, continuation_ids) -> list of (logprob, is_greedy)."""
        order = sorted(range(len(pairs)), key=lambda i: -len(pairs[i][0]) - len(pairs[i][1]))
        out = [None] * len(pairs)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            seqs, conts = [], []
            for i in idx:
                ctx, cont = pairs[i]
                cont = cont[-self.ctx:]
                full = (ctx + cont)[-(self.ctx + 1):]
                seqs.append(full)
                conts.append(len(cont))
            width = max(len(s) - 1 for s in seqs)
            inp = torch.zeros(len(seqs), width, dtype=torch.long)
            for r, s in enumerate(seqs):
                inp[r, : len(s) - 1] = torch.tensor(s[:-1])
            logp = F.log_softmax(self.model(inp.to(self.device)).float(), dim=-1).cpu()
            for r, (i, s, n) in enumerate(zip(idx, seqs, conts)):
                L = len(s) - 1
                target = torch.tensor(s[-n:])
                lp = logp[r, L - n:L]
                out[i] = (lp.gather(1, target[:, None]).sum().item(), bool((lp.argmax(-1) == target).all()))
        return out

    def loglikelihood(self, requests, disable_tqdm=False):
        return self.score([self.encode_pair(*req.args) for req in requests])

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        results = []
        for req in requests:
            windows = [make_disjoint_window(w) for w in get_rolling_token_windows(
                token_list=self.encode(req.args[0]), prefix_token=self.prefix_token,
                max_seq_len=self.ctx, context_len=1)]
            results.append(sum(lp for lp, _ in self.score(windows)))
        return results

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("Generation tasks are not part of this evaluation.")


@torch.inference_mode()
def wikitext103_perplexity(model, tokenizer, device, context_length, batch_size=64):
    from datasets import load_dataset
    name, config, revision = WIKITEXT
    rows = [r for r in load_dataset(name, config, split="test", revision=revision)["text"] if r.strip()]
    eos = tokenizer.eos_id()
    ids = []
    for row in rows:
        ids += tokenizer.encode(row, out_type=int) + [eos]
    n = len(ids) // (context_length + 1)
    x = torch.tensor(ids[: n * (context_length + 1)]).view(n, context_length + 1)
    total = 0.0
    for b in range(0, n, batch_size):
        xb = x[b:b + batch_size].to(device)
        logits = model(xb[:, :-1]).float()
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), xb[:, 1:].reshape(-1), reduction="sum").item()
    scored = n * context_length
    mean_nll = total / scored
    words = sum(len(r.split()) for r in rows)
    n_bytes = sum(len(r.encode("utf-8")) for r in rows)
    return {"dataset": name, "config": config, "revision": revision, "split": "test",
            "rows": len(rows), "tokens": len(ids), "windows": n, "scored_tokens": scored,
            "mean_token_nll": mean_nll, "token_perplexity": math.exp(mean_nll),
            "word_perplexity": math.exp(mean_nll * len(ids) / words),
            "bits_per_byte": mean_nll * len(ids) / n_bytes / math.log(2)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, default=PROJECT_ROOT / "tokenizers" / "fineweb_16384_bpe.model")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "eval_lm_harness_5B.json")
    parser.add_argument("--limit", type=int, default=None, help="Examples per task (for a quick smoke test).")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = LoopedGPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = spm.SentencePieceProcessor(model_file=str(args.tokenizer_model))
    print(f"Trainable parameters: {parameters:,} | device: {device}", flush=True)

    started = time.time()
    lm = LoopedGPTLM(model, tokenizer, device, config["context_length"], args.batch_size)
    results = simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, limit=args.limit, log_samples=False)
    report = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": sha256(args.checkpoint),
        "tokenizer_sha256": sha256(args.tokenizer_model),
        "trainable_parameters": parameters,
        "tokens_seen": checkpoint.get("tokens_seen"),
        "context_length": config["context_length"],
        "num_loops": config.get("num_loops"),
        "lm_eval_version": results.get("versions"),
        "limit": args.limit,
        "results": results["results"],
        "wikitext103_test": None if args.limit else wikitext103_perplexity(
            model, tokenizer, device, config["context_length"]),
        "seconds": round(time.time() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    for task, metrics in report["results"].items():
        shown = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
        print(f"{task:12s} {shown}")
    if report["wikitext103_test"]:
        print("wikitext-103 test:", {k: v for k, v in report["wikitext103_test"].items() if k != "revision"})
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
