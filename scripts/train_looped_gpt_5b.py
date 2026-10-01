#!/usr/bin/env python3
"""Two-phase pretraining of the submitted Looped GPT (42,070,080 parameters).

This script is a reconstruction of the run that produced the submitted 5B-token
checkpoint. The original script ran on a collaborator's machine and is not part
of this repository; every hyperparameter below was recovered from the
checkpoint (``checkpoint["config"]`` and ``promotion_metadata``) and from the
W&B runs ``qe9x3h4q`` (phase 1) and ``qutvxppl`` (phase 2) in project
``gpt2-50M``. It has not been re-executed end to end.

Phase 1 (``--phase base``): 0 -> 2B tokens, AdamW 3e-4, 5% warmup, cosine decay
to 3e-5. Original run: 11.96 h on one RTX 5060 Ti.

Phase 2 (``--phase extend``): continues the phase-1 checkpoint to a cumulative
5B tokens at a constant 3e-5. The original run restarted the data stream
instead of restoring the phase-1 data cursor. Original run: 17.9 h.

Usage::

    python scripts/train_looped_gpt_5b.py --phase base
    python scripts/train_looped_gpt_5b.py --phase extend \
        --init-from checkpoints/looped-gpt-2b.pt
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_mini_lab.evaluation import evaluate_benchmark_suite
from llm_mini_lab.models import LoopedGPTModel
from llm_mini_lab.training import (
    cosine_lr_multiplier,
    create_dataloader_smollm,
    generate_text_simple,
    init_xavier,
    load_tokenizer,
    make_fixed_eval_loaders,
    text_to_token_ids,
    token_ids_to_text,
)


# Exact model configuration stored in the submitted checkpoint.
MODEL_CONFIG = {
    "vocab_size": 16_384,
    "context_length": 128,
    "emb_dim": 1024,
    "n_heads": 8,
    "n_kv_heads": 8,
    "n_unique_layers": 3,
    "num_loops": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
    "positional_encoding": "rope",
    "ff_activation": "swiglu",
    "ff_hidden_dim": 1376,
    "tie_embeddings": True,
    "tokenizer_name": "sp16384",
}
EXPECTED_PARAMETERS = 42_070_080

PHASES = {
    "base": {"target_tokens": 2_000_000_000, "learning_rate": 3e-4,
             "schedule": "cosine", "run_name": "rsi-2b-cosmopedia-v2-wd005",
             "checkpoint": "looped-gpt-2b.pt"},
    "extend": {"target_tokens": 5_000_000_000, "learning_rate": 3e-5,
               "schedule": "constant", "run_name": "rsi-2b-to-5b-cosmopedia-v2-wd005",
               "checkpoint": "looped-gpt-5b.pt"},
}

MICRO_BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 4          # 16 x 128 x 4 = 8,192 tokens per update
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.05
MIN_LR_RATIO = 0.1
SEED = 123

SMOLLM_CONFIG = "cosmopedia-v2"
SHUFFLE_BUFFER = 10_000
TOKENIZER_MODEL = PROJECT_ROOT / "tokenizers" / "fineweb_16384_bpe.model"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
WANDB_PROJECT = "gpt2-50M"

EVAL_EVERY_UPDATES = 1_000         # validation loss every 8.192M tokens
EVAL_BATCHES = 20
BENCHMARK_EVERY_TOKENS = 100_000_000
BENCHMARK_NAMES = ("hellaswag", "arc_easy", "piqa", "winogrande")
BENCHMARK_MAX_EXAMPLES = 500
GENERATION_EVERY_TOKENS = 50_000_000
CHECKPOINT_EVERY_TOKENS = 500_000_000

# Prompts logged as ``generation/<id>`` in the original W&B runs.
GENERATION_PROMPTS = {
    "ai_continuation": "Artificial intelligence",
    "factual": "The capital of France is",
    "explanation": "Explain why the sky appears blue:",
    "dialogue": "User: Tell me one interesting fact about the Moon.\nAssistant:",
    "instruction": "User: What is 12 times 7?\nAssistant:",
    "ood": "The following sentence is deliberately unusual:",
    "reasoning": "If a box contains 3 red balls and 2 blue balls, then",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--init-from", type=Path,
                        help="Phase-1 checkpoint (required for --phase extend).")
    parser.add_argument("--data-seed", type=int, default=SEED,
                        help="Seed for the streaming shuffle buffer.")
    args = parser.parse_args()
    if args.phase == "extend" and args.init_from is None:
        parser.error("--phase extend requires --init-from")
    return args


def autocast(device, dtype):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def evaluate(model, loader, device, amp_dtype) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast(device, amp_dtype):
                logits = model(inputs)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            losses.append(loss.float().item())
    model.train()
    return sum(losses) / len(losses)


def log_generations(model, tokenizer, device, update, tokens_seen):
    model.eval()
    log = {"update": update}
    for prompt_id, prompt in GENERATION_PROMPTS.items():
        ids = text_to_token_ids(prompt, tokenizer).to(device)
        sample = generate_text_simple(model, ids, max_new_tokens=50,
                                      context_size=MODEL_CONFIG["context_length"])
        log[f"generation/{prompt_id}"] = token_ids_to_text(sample.cpu(), tokenizer)
    model.train()
    wandb.log(log)
    print(f"[{tokens_seen:,} tokens] {log['generation/ai_continuation']}")


def log_benchmarks(model, tokenizer, device, amp_dtype, update):
    results = evaluate_benchmark_suite(
        model, tokenizer, device,
        names=BENCHMARK_NAMES,
        max_examples=BENCHMARK_MAX_EXAMPLES,
        context_length=MODEL_CONFIG["context_length"],
        autocast_dtype=amp_dtype if device.type == "cuda" else None,
    )
    log = {"update": update}
    for name, metrics in results.items():
        log[f"benchmark/{name}_accuracy"] = metrics["accuracy"]
    if results:
        log["benchmark/mean_accuracy"] = (
            sum(m["accuracy"] for m in results.values()) / len(results))
    wandb.log(log)


def save_checkpoint(path, model, optimizer, scheduler, update, tokens_seen,
                    training_hours, phase):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": MODEL_CONFIG,
        "update": update,
        "tokens_seen": tokens_seen,
        "training_time_hours": training_hours,
        "phase": phase,
    }, path)


def main() -> None:
    args = parse_args()
    phase = PHASES[args.phase]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
    amp_dtype = (torch.bfloat16 if device.type == "cuda"
                 and torch.cuda.is_bf16_supported() else torch.float16)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)

    tokenizer = load_tokenizer(MODEL_CONFIG["tokenizer_name"], TOKENIZER_MODEL)
    model = LoopedGPTModel(MODEL_CONFIG)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {parameter_count:,}")
    assert parameter_count == EXPECTED_PARAMETERS, parameter_count
    assert parameter_count <= 50_000_000

    update = tokens_seen = 0
    previous_hours = 0.0
    parent = None
    if args.phase == "base":
        model.apply(init_xavier)
        model.out_head.weight = model.tok_emb.weight
    else:
        parent = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(parent["model_state_dict"])
        update = parent["update"]
        tokens_seen = parent["tokens_seen"]
        previous_hours = parent.get("training_time_hours", 0.0)
    # Move before building the optimizer so its restored state lands on the
    # same device as the parameters.
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=phase["learning_rate"],
                                  weight_decay=WEIGHT_DECAY)
    if parent is not None:
        optimizer.load_state_dict(parent["optimizer_state_dict"])
        # The parent's scheduler left initial_lr at 3e-4; pin phase 2 to 3e-5.
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] = phase["learning_rate"]

    tokens_per_update = MICRO_BATCH_SIZE * MODEL_CONFIG["context_length"] * GRADIENT_ACCUMULATION
    max_updates = math.ceil(phase["target_tokens"] / tokens_per_update)
    if phase["schedule"] == "cosine":
        warmup_steps = round(WARMUP_RATIO * max_updates)
        schedule = lambda step: cosine_lr_multiplier(
            step, warmup_steps, max_updates, MIN_LR_RATIO)
    else:
        schedule = lambda step: 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    train_loader, validation_loader = create_dataloader_smollm(
        batch_size=MICRO_BATCH_SIZE,
        max_length=MODEL_CONFIG["context_length"],
        seed=args.data_seed,
        shuffle_buffer=SHUFFLE_BUFFER,
        config_name=SMOLLM_CONFIG,
        encoder=tokenizer,
    )
    _, fixed_validation_loader = make_fixed_eval_loaders(
        train_loader, validation_loader, max_train_batches=1,
        max_val_batches=EVAL_BATCHES)

    wandb.init(project=WANDB_PROJECT, name=phase["run_name"], config={
        **MODEL_CONFIG, "parameter_count": parameter_count,
        "target_tokens": phase["target_tokens"],
        "learning_rate": phase["learning_rate"], "lr_schedule": phase["schedule"],
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
        "warmup_ratio": WARMUP_RATIO, "min_lr_ratio": MIN_LR_RATIO,
        "smollm_config": SMOLLM_CONFIG, "seed": SEED, "data_seed": args.data_seed,
    })
    wandb.define_metric("update")
    wandb.define_metric("*", step_metric="update")

    checkpoint_path = CHECKPOINT_DIR / phase["checkpoint"]
    next_benchmark = (tokens_seen // BENCHMARK_EVERY_TOKENS + 1) * BENCHMARK_EVERY_TOKENS
    next_generation = (tokens_seen // GENERATION_EVERY_TOKENS + 1) * GENERATION_EVERY_TOKENS
    next_checkpoint = (tokens_seen // CHECKPOINT_EVERY_TOKENS + 1) * CHECKPOINT_EVERY_TOKENS
    started = time.perf_counter()
    hours = lambda: previous_hours + (time.perf_counter() - started) / 3600

    print(f"Device: {device} | {tokens_seen:,} -> {phase['target_tokens']:,} tokens")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    while tokens_seen < phase["target_tokens"]:
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast(device, amp_dtype):
                logits = model(inputs)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at update {update + 1}")
            scaler.scale(loss / GRADIENT_ACCUMULATION).backward()
            tokens_seen += inputs.numel()
            accumulated += 1
            if accumulated < GRADIENT_ACCUMULATION:
                continue

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update += 1
            accumulated = 0

            wandb.log({
                "update": update,
                "tokens_seen": tokens_seen,
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": grad_norm.item(),
                "training/total_time_hours": hours(),
            })
            if update % EVAL_EVERY_UPDATES == 0:
                validation_loss = evaluate(model, fixed_validation_loader, device, amp_dtype)
                wandb.log({"update": update, "validation/loss": validation_loss})
                print(f"update {update:,} | {tokens_seen:,} tokens | "
                      f"train {loss.item():.4f} | val {validation_loss:.4f}")
            if tokens_seen >= next_generation:
                log_generations(model, tokenizer, device, update, tokens_seen)
                next_generation += GENERATION_EVERY_TOKENS
            if tokens_seen >= next_benchmark:
                log_benchmarks(model, tokenizer, device, amp_dtype, update)
                next_benchmark += BENCHMARK_EVERY_TOKENS
            if tokens_seen >= next_checkpoint:
                save_checkpoint(checkpoint_path, model, optimizer, scheduler,
                                update, tokens_seen, hours(), args.phase)
                next_checkpoint += CHECKPOINT_EVERY_TOKENS
            if tokens_seen >= phase["target_tokens"]:
                break

    validation_loss = evaluate(model, fixed_validation_loader, device, amp_dtype)
    wandb.log({"update": update, "validation/loss": validation_loss})
    save_checkpoint(checkpoint_path, model, optimizer, scheduler,
                    update, tokens_seen, hours(), args.phase)
    print(f"Done: {tokens_seen:,} tokens, val loss {validation_loss:.4f}, "
          f"{hours():.2f} h total -> {checkpoint_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
