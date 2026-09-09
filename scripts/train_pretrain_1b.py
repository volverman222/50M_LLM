#!/usr/bin/env python3
"""Preentrena el Looped GPT de 50M sobre ~1B de tokens de FineWeb-Edu.

Ejemplo:
    python scripts/train_pretrain_1b.py --wandb --run-name looped-gpt-1b

El dataset se consume en streaming desde Hugging Face. No se descarga el corpus
completo ni se mantiene tokenizado en memoria.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from llm_mini_lab.benchmarks import evaluate_benchmark_suite
from llm_mini_lab.model import LoopedGPTModel
from llm_mini_lab.pretraining import (
    LOOPED_GPT_CONFIG,
    cosine_lr_multiplier,
    create_dataloader_fineweb,
    create_dataloader_smollm,
    generate_text_simple,
    init_xavier,
    make_fixed_eval_loaders,
    text_to_token_ids,
    token_ids_to_text,
)


def parse_args(dataset: str = "fineweb") -> argparse.Namespace:
    dataset_label = ("FineWeb-Edu" if dataset == "fineweb"
                     else "HuggingFaceTB/smollm-corpus")
    parser = argparse.ArgumentParser(
        description=f"Preentrenamiento streaming sobre 1B tokens de {dataset_label}."
    )
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--context-length", type=int, default=256)
    # Conservador para una GPU de 24 GB; el batch efectivo sigue siendo 64.
    parser.add_argument("--micro-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--val-mod", type=int, default=100)
    if dataset == "smollm":
        parser.add_argument(
            "--smollm-config", default="cosmopedia-v2",
            help="Subconjunto/configuración de HuggingFaceTB/smollm-corpus.",
        )
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--benchmark-every", type=int, default=0,
                        help="0 desactiva benchmarks durante el entrenamiento.")
    parser.add_argument("--benchmark-max-examples", type=int, default=100)
    default_checkpoint_dir = ("checkpoints/pretrain-1b" if dataset == "fineweb"
                              else "checkpoints/pretrain-smollm-1b")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path(default_checkpoint_dir))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="gpt2-50M")
    parser.add_argument("--run-name", default=f"looped-gpt-50m-{dataset}-1b")
    parser.add_argument(
        "--hourly-cost", type=float, default=None, metavar="USD",
        help="Coste del equipo en USD/hora; si se omite, se pregunta al iniciar.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                        default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Valida argumentos/modelo sin abrir el dataset.")
    args = parser.parse_args()

    positive = ("target_tokens", "context_length", "micro_batch_size",
                "gradient_accumulation", "eval_every", "eval_batches",
                "log_every", "save_every")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} debe ser mayor que 0")
    if not 0 < args.warmup_ratio < 1:
        parser.error("--warmup-ratio debe estar entre 0 y 1")
    if args.hourly_cost is not None and args.hourly_cost < 0:
        parser.error("--hourly-cost no puede ser negativo")
    return args


def create_stream_loaders(args: argparse.Namespace, dataset: str, seed: int):
    common = {
        "batch_size": args.micro_batch_size,
        "max_length": args.context_length,
        "val_mod": args.val_mod,
        "seed": seed,
        "num_workers": args.num_workers,
        "shuffle_buffer": args.shuffle_buffer,
    }
    if dataset == "fineweb":
        return create_dataloader_fineweb(**common)
    if dataset == "smollm":
        return create_dataloader_smollm(
            **common, config_name=args.smollm_config,
        )
    raise ValueError(f"Dataset no soportado: {dataset}")


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS no está disponible")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_settings(device: torch.device, disabled: bool) -> tuple[bool, torch.dtype]:
    enabled = device.type == "cuda" and not disabled
    dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    return enabled, dtype


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def get_hourly_cost(cli_value: float | None) -> float:
    if cli_value is not None:
        return cli_value
    while True:
        try:
            value = input("Coste del equipo en USD por hora: ").strip()
        except EOFError as exc:
            raise RuntimeError(
                "No hay entrada interactiva. Indica --hourly-cost USD."
            ) from exc
        try:
            hourly_cost = float(value.replace(",", "."))
        except ValueError:
            print("Introduce un número válido, por ejemplo: 0.75")
            continue
        if hourly_cost < 0:
            print("El coste no puede ser negativo.")
            continue
        return hourly_cost


def checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler.LRScheduler,
                       scaler: torch.amp.GradScaler, cfg: dict[str, Any],
                       update: int, tokens_seen: int, epoch: int,
                       wandb_run_id: str | None, elapsed_seconds: float,
                       estimated_cost_usd: float) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": cfg,
        "update": update,
        "tokens_seen": tokens_seen,
        "stream_epoch": epoch,
        "wandb_run_id": wandb_run_id,
        "training_time_seconds": elapsed_seconds,
        "training_time_hours": elapsed_seconds / 3600,
        "estimated_cost_usd": estimated_cost_usd,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def evaluate(model: torch.nn.Module, loader: Any, max_batches: int,
             device: torch.device, amp_enabled: bool,
             amp_dtype: torch.dtype) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            amp = (torch.autocast("cuda", dtype=amp_dtype) if amp_enabled
                   else contextlib.nullcontext())
            with amp:
                logits = model(inputs)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            losses.append(loss.float().item())
    model.train(was_training)
    if not losses:
        raise RuntimeError("El loader de validación no produjo batches")
    return sum(losses) / len(losses)


def main(dataset: str = "fineweb") -> None:
    args = parse_args(dataset)
    device = choose_device(args.device)
    seed_everything(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    tokens_per_update = (args.micro_batch_size * args.context_length
                         * args.gradient_accumulation)
    max_updates = math.ceil(args.target_tokens / tokens_per_update)
    warmup_steps = max(1, min(max_updates - 1,
                              round(max_updates * args.warmup_ratio)))
    if max_updates < 2:
        raise ValueError("La configuración debe producir al menos 2 updates")

    cfg = {**LOOPED_GPT_CONFIG, "context_length": args.context_length}
    model = LoopedGPTModel(cfg)
    model.apply(init_xavier)
    model.out_head.weight = model.tok_emb.weight
    model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count > 50_000_000:
        raise RuntimeError(f"El modelo supera 50M parámetros: {parameter_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_lr_multiplier(step, warmup_steps, max_updates,
                                          args.min_lr_ratio),
    )
    amp_enabled, amp_dtype = amp_settings(device, args.no_amp)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    update = tokens_seen = stream_epoch = 0
    previous_elapsed_seconds = 0.0
    previous_cost_usd = 0.0
    resume_wandb_id = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["config"] != cfg:
            raise ValueError("La configuración del checkpoint no coincide con el modelo")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        update = int(checkpoint.get("update", checkpoint.get("update_step", 0)))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        stream_epoch = int(checkpoint.get("stream_epoch", 0))
        resume_wandb_id = checkpoint.get("wandb_run_id")
        previous_elapsed_seconds = float(checkpoint.get("training_time_seconds", 0.0))
        previous_cost_usd = float(checkpoint.get("estimated_cost_usd", 0.0))

    print(f"Dispositivo: {device} | AMP: {amp_enabled} ({amp_dtype})")
    print(f"Parámetros: {parameter_count:,}")
    print(f"Batch efectivo: {args.micro_batch_size * args.gradient_accumulation} "
          f"secuencias = {tokens_per_update:,} tokens/update")
    print(f"Objetivo: {args.target_tokens:,} tokens | {max_updates:,} updates")
    if args.dry_run:
        print("Dry run correcto; no se abrió el stream ni se entrenó.")
        return

    hourly_cost = get_hourly_cost(args.hourly_cost)
    print(f"Coste indicado: ${hourly_cost:.4f} USD/hora")

    # Los loaders de evaluación se materializan una vez para que comparar
    # checkpoints no cambie el estado del stream de entrenamiento.
    train_loader, val_loader = create_stream_loaders(
        args, dataset, args.seed + stream_epoch,
    )
    _, fixed_val_loader = make_fixed_eval_loaders(
        train_loader, val_loader, max_train_batches=1,
        max_val_batches=args.eval_batches,
    )

    wandb = None
    run = None
    if args.wandb:
        import wandb as wandb_module
        wandb = wandb_module
        run = wandb.init(project=args.wandb_project, name=args.run_name,
                         id=resume_wandb_id,
                         resume="must" if resume_wandb_id else None,
                         config=vars(args))
        wandb.define_metric("update")
        wandb.define_metric("*", step_metric="update")

    benchmark_names = ("hellaswag", "arc_easy", "piqa", "winogrande")
    tokenizer = tiktoken.get_encoding("gpt2")
    started = time.perf_counter()
    initial_tokens_seen = tokens_seen
    model.train()
    optimizer.zero_grad(set_to_none=True)
    microbatches_accumulated = 0

    try:
        while update < max_updates and tokens_seen < args.target_tokens:
            stream_epoch += 1
            produced_batch = False
            for inputs, targets in train_loader:
                produced_batch = True
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                amp = (torch.autocast("cuda", dtype=amp_dtype) if amp_enabled
                       else contextlib.nullcontext())
                with amp:
                    logits = model(inputs)
                    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                    scaled_loss = loss / args.gradient_accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Loss no finita en update {update + 1}")
                scaler.scale(scaled_loss).backward()

                # Cuenta tokens realmente procesados, incluidos microbatches.
                tokens_seen += inputs.numel()
                microbatches_accumulated += 1
                end_update = (microbatches_accumulated == args.gradient_accumulation
                              or tokens_seen >= args.target_tokens)
                if not end_update:
                    continue

                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update += 1
                microbatches_accumulated = 0

                metrics = {
                    "update": update, "tokens_seen": tokens_seen,
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": float(grad_norm),
                }
                if wandb:
                    wandb.log(metrics)
                if update % args.log_every == 0 or update == 1:
                    session_elapsed = time.perf_counter() - started
                    rate = ((tokens_seen - initial_tokens_seen)
                            / max(session_elapsed, 1e-9))
                    print(f"{update:>7,}/{max_updates:,} | {tokens_seen:>13,} tok | "
                          f"loss {loss.item():.4f} | {rate:,.0f} tok/s")

                if update % args.eval_every == 0:
                    val_loss = evaluate(model, fixed_val_loader, args.eval_batches,
                                        device, amp_enabled, amp_dtype)
                    print(f"Validación @ {update:,}: loss {val_loss:.4f}")
                    if wandb:
                        wandb.log({"update": update, "validation/loss": val_loss})

                if args.benchmark_every and update % args.benchmark_every == 0:
                    results = evaluate_benchmark_suite(
                        model=model, tokenizer=tokenizer, device=device,
                        names=benchmark_names,
                        max_examples=args.benchmark_max_examples,
                        context_length=args.context_length,
                        autocast_dtype=amp_dtype if amp_enabled else None,
                    )
                    benchmark_log = {
                        f"benchmark/{name}_accuracy": values["accuracy"]
                        for name, values in results.items()
                    }
                    print("Benchmarks:", benchmark_log)
                    if wandb:
                        wandb.log({"update": update, **benchmark_log})

                if update % args.save_every == 0:
                    # Se sobrescribe para no llenar un disco de 16 GB con los
                    # estados (modelo + Adam) de decenas de checkpoints.
                    path = args.checkpoint_dir / "latest.pt"
                    session_elapsed = time.perf_counter() - started
                    total_elapsed = previous_elapsed_seconds + session_elapsed
                    total_cost = previous_cost_usd + session_elapsed / 3600 * hourly_cost
                    atomic_save(checkpoint_payload(
                        model, optimizer, scheduler, scaler, cfg, update,
                        tokens_seen, stream_epoch, run.id if run else None,
                        total_elapsed, total_cost,
                    ), path)
                    print(f"Checkpoint: {path}")

                if update >= max_updates or tokens_seen >= args.target_tokens:
                    break

            if not produced_batch:
                raise RuntimeError(f"{dataset} no produjo ningún batch")
            if update < max_updates and tokens_seen < args.target_tokens:
                # Nueva permutación si se agotase el stream antes del objetivo.
                train_loader, _ = create_stream_loaders(
                    args, dataset, args.seed + stream_epoch,
                )
    finally:
        session_elapsed = time.perf_counter() - started
        total_elapsed = previous_elapsed_seconds + session_elapsed
        total_cost = previous_cost_usd + session_elapsed / 3600 * hourly_cost
        final_path = args.checkpoint_dir / "final.pt"
        atomic_save(checkpoint_payload(
            model, optimizer, scheduler, scaler, cfg, update, tokens_seen,
            stream_epoch, run.id if run else None, total_elapsed, total_cost,
        ), final_path)
        summary = {
            "dataset": dataset,
            "tokens_seen": tokens_seen,
            "updates": update,
            "target_tokens": args.target_tokens,
            "training_time_seconds": total_elapsed,
            "training_time_hours": total_elapsed / 3600,
            "hourly_cost_usd_last_session": hourly_cost,
            "estimated_cost_usd": total_cost,
            "target_reached": tokens_seen >= args.target_tokens,
            "checkpoint": str(final_path),
        }
        summary_path = args.checkpoint_dir / "training_summary.json"
        atomic_save_json(summary, summary_path)
        if wandb:
            wandb.log({
                "update": update,
                "training/total_time_hours": total_elapsed / 3600,
                "training/estimated_cost_usd": total_cost,
            })
            wandb.finish()

    prompt = text_to_token_ids("Artificial intelligence", tokenizer).to(device)
    sample_ids = generate_text_simple(model, prompt, max_new_tokens=30,
                                      context_size=args.context_length)
    print("Muestra:", token_ids_to_text(sample_ids.cpu(), tokenizer))
    print(f"Tiempo total: {total_elapsed / 3600:.3f} horas")
    print(f"Coste estimado acumulado: ${total_cost:.2f} USD")
    print(f"Resumen guardado en: {summary_path}")
    print(f"Entrenamiento terminado: {tokens_seen:,} tokens; checkpoint {final_path}")


if __name__ == "__main__":
    main()
