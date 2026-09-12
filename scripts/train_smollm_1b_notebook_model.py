#!/usr/bin/env python3
"""Preentrena en SmolLM Corpus el Looped GPT del notebook v4.6.

La configuración es deliberadamente fija para que el comando de ejecución sea
simple: ``python scripts/train_smollm_1b_notebook_model.py``.
"""

from __future__ import annotations

import contextlib
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_mini_lab.models import LoopedGPTModel
from llm_mini_lab.training import (
    LOOPED_GPT_CONFIG,
    cosine_lr_multiplier,
    create_dataloader_smollm,
    generate_text_simple,
    init_xavier,
    load_tokenizer,
    make_fixed_eval_loaders,
    text_to_token_ids,
    token_ids_to_text,
    tokenizer_vocab_size,
)


# Modelo del notebook Test_pretrain-v4.6_improve_batch.ipynb.
CONTEXT_LENGTH = 128
EMB_DIM = 1024
TARGET_TOKENS = 1_000_000_000
MICRO_BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.05
MIN_LR_RATIO = 0.1
SEED = 123

# Datos, seguimiento y salida: valores internos, no opciones de línea de comandos.
SMOLLM_CONFIG = "cosmopedia-v2"
TOKENIZER_NAME = "sp16384"
TOKENIZER_MODEL = PROJECT_ROOT / "tokenizers" / "fineweb_16384_bpe.model"
EVAL_EVERY = 500
EVAL_BATCHES = 20
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "smollm-1b-notebook-model.pt"


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device, amp_enabled, amp_dtype) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch_index, (inputs, targets) in enumerate(loader):
            if batch_index == EVAL_BATCHES:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            amp = (torch.autocast("cuda", dtype=amp_dtype) if amp_enabled
                   else contextlib.nullcontext())
            with amp:
                logits = model(inputs)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            losses.append(loss.float().item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(model, optimizer, scheduler, config, update, tokens_seen):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "updates": update,
        "tokens_seen": tokens_seen,
    }, CHECKPOINT_PATH)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = choose_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = load_tokenizer(TOKENIZER_NAME, TOKENIZER_MODEL)
    config = {
        **LOOPED_GPT_CONFIG,
        "context_length": CONTEXT_LENGTH,
        "emb_dim": EMB_DIM,
        "vocab_size": tokenizer_vocab_size(tokenizer),
        "tokenizer_name": TOKENIZER_NAME,
    }
    model = LoopedGPTModel(config)

    model.apply(init_xavier)
    model.out_head.weight = model.tok_emb.weight
    model.to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parámetros entrenables: {parameter_count:,}")
    tokens_per_update = MICRO_BATCH_SIZE * CONTEXT_LENGTH * GRADIENT_ACCUMULATION
    max_updates = math.ceil(TARGET_TOKENS / tokens_per_update)
    warmup_steps = round(WARMUP_RATIO * max_updates)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_lr_multiplier(step, warmup_steps, max_updates,
                                          MIN_LR_RATIO),
    )
    amp_enabled = device.type == "cuda"
    amp_dtype = (torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    train_loader, validation_loader = create_dataloader_smollm(
        batch_size=MICRO_BATCH_SIZE,
        max_length=CONTEXT_LENGTH,
        seed=SEED,
        shuffle_buffer=10_000,
        config_name=SMOLLM_CONFIG,
        encoder=tokenizer,
    )
    _, fixed_validation_loader = make_fixed_eval_loaders(
        train_loader, validation_loader, max_train_batches=1,
        max_val_batches=EVAL_BATCHES,
    )

    print(f"Dispositivo: {device}")
    print(f"Tokenizador: {TOKENIZER_NAME} ({config['vocab_size']:,} tokens)")
    print(f"Modelo del notebook: {parameter_count:,} parámetros")
    print(f"Objetivo: {TARGET_TOKENS:,} tokens en {max_updates:,} updates")
    print(f"Batch efectivo: {MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION} secuencias")

    update = tokens_seen = accumulated = 0
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while update < max_updates:
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            amp = (torch.autocast("cuda", dtype=amp_dtype) if amp_enabled
                   else contextlib.nullcontext())
            with amp:
                logits = model(inputs)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                loss = loss / GRADIENT_ACCUMULATION
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Pérdida no finita en update {update + 1}")
            scaler.scale(loss).backward()
            tokens_seen += inputs.numel()
            accumulated += 1

            if accumulated < GRADIENT_ACCUMULATION and tokens_seen < TARGET_TOKENS:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update += 1
            accumulated = 0

            if update == 1 or update % 10 == 0:
                elapsed = time.perf_counter() - started
                print(f"{update:,}/{max_updates:,} | {tokens_seen:,} tokens | "
                      f"loss {(loss * GRADIENT_ACCUMULATION).item():.4f} | "
                      f"{tokens_seen / elapsed:,.0f} tok/s")
            if update % EVAL_EVERY == 0:
                print(f"Validación: {evaluate(model, fixed_validation_loader, device, amp_enabled, amp_dtype):.4f}")
            if update == max_updates or tokens_seen >= TARGET_TOKENS:
                break
        if update == max_updates or tokens_seen >= TARGET_TOKENS:
            break

    save_checkpoint(model, optimizer, scheduler, config, update, tokens_seen)
    prompt = text_to_token_ids("Artificial intelligence", tokenizer).to(device)
    sample = generate_text_simple(model, prompt, max_new_tokens=30,
                                  context_size=CONTEXT_LENGTH)
    print("Muestra:", token_ids_to_text(sample.cpu(), tokenizer))
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
