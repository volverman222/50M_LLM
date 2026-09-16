"""Entrenamiento base para autoresearch, igual al notebook v4.8.

La métrica que se informa al final es ``test_loss``: la pérdida media de
entropía cruzada sobre ``data/smollm_local/validation.txt``. Menor es mejor.
"""

import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsi_architecture import build_model
from dynamic_observatory import capture_observations
from dynamic_observatory.wandb_io import log_observations_to_wandb
from llm_mini_lab.training import (
    LOOPED_GPT_CONFIG,
    LocalTextDataset,
    cosine_lr_multiplier,
    evaluate,
    init_xavier,
    load_tokenizer,
    tokenizer_vocab_size,
)

# ---------------------------------------------------------------------------
# Esta configuración y rsi_architecture/ son la superficie editable del agente.
# ---------------------------------------------------------------------------
SEED = 123
MAX_LENGTH = 128
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
MAX_TOKENS = 1_000_000
EVAL_INTERVAL = 10
TEST_BATCHES = 10

MODEL_CONFIG = {
    "n_unique_layers": 3,
    "n_heads": 8,
    "emb_dim": 1024,
    "positional_encoding": "rope",
}
# ---------------------------------------------------------------------------

TOKENIZER_NAME = "sp16384"
TOKENIZER_MODEL = PROJECT_ROOT / "tokenizers" / "fineweb_16384_bpe.model"
DATA_DIR = PROJECT_ROOT / "data" / "smollm_local"
TRAIN_PATH = DATA_DIR / "train.txt"
TEST_PATH = DATA_DIR / "validation.txt"

DYNAMICS_ENV = "AUTORESEARCH_DYNAMICS"


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _dynamic_module_names() -> list[str] | None:
    raw = os.getenv("AUTORESEARCH_DYNAMICS_MODULES", "").strip()
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if not TOKENIZER_MODEL.is_file():
        raise FileNotFoundError(f"No se encontró el tokenizador: {TOKENIZER_MODEL}")
    if not TRAIN_PATH.is_file() or not TEST_PATH.is_file():
        raise FileNotFoundError(f"Se esperan {TRAIN_PATH} y {TEST_PATH}")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    tokenizer = load_tokenizer(TOKENIZER_NAME, TOKENIZER_MODEL)
    vocab_size = tokenizer_vocab_size(tokenizer)
    train_dataset = LocalTextDataset(TRAIN_PATH, MAX_LENGTH, encoder=tokenizer)
    test_dataset = LocalTextDataset(TEST_PATH, MAX_LENGTH, encoder=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    dynamics_enabled = _env_enabled(DYNAMICS_ENV)
    dynamics_probe_inputs = None
    if dynamics_enabled:
        dynamics_probe_inputs = next(iter(test_loader))[0][:1].to(device)

    max_updates = MAX_TOKENS // (BATCH_SIZE * MAX_LENGTH)
    if max_updates < 1:
        raise ValueError("MAX_TOKENS debe cubrir al menos un lote")
    cfg = {
        **LOOPED_GPT_CONFIG,
        **MODEL_CONFIG,
        "context_length": MAX_LENGTH,
        "vocab_size": vocab_size,
        "tokenizer_name": TOKENIZER_NAME,
    }
    model = build_model(cfg).to(device)
    model.apply(init_xavier)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    if n_params > 50_000_000:
        raise ValueError(f"El modelo excede 50M de parámetros: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_lr_multiplier(step, max(1, int(0.05 * max_updates)), max_updates, 0.1),
    )
    print(f"device={device.type} vocab_size={vocab_size} parameters={n_params}")
    print(f"train_blocks={len(train_dataset)} test_blocks={len(test_dataset)} updates={max_updates}")

    run = wandb.init(
        project="gpt2-50M",
        name=f"rsi-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            **cfg,
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_tokens": MAX_TOKENS,
            "max_updates": max_updates,
            "test_batches": TEST_BATCHES,
            "parameters": n_params,
        },
    )
    wandb.define_metric("update")
    wandb.define_metric("*", step_metric="update")

    update = tokens_seen = 0
    started_at = time.perf_counter()
    model.train()
    while update < max_updates:
        for inputs, targets in train_loader:
            update += 1
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            train_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            if not torch.isfinite(train_loss):
                raise FloatingPointError(f"Pérdida no finita en el paso {update}")
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            tokens_seen += inputs.numel()
            wandb.log({
                "update": update,
                "tokens_seen": tokens_seen,
                "train/loss": train_loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            })

            if update % EVAL_INTERVAL == 0:
                test_loss = evaluate(model, test_loader, TEST_BATCHES, device)
                wandb.log({"update": update, "test/loss": test_loss})
                print(f"step={update}/{max_updates} train_loss={train_loss.item():.6f} test_loss={test_loss:.6f}")
            if update >= max_updates:
                break

    test_loss = evaluate(model, test_loader, TEST_BATCHES, device)
    elapsed = time.perf_counter() - started_at
    wandb.log({
        "update": update,
        "tokens_seen": tokens_seen,
        "test/loss": test_loss,
        "run/elapsed_seconds": elapsed,
    })
    run.summary["test_loss"] = test_loss
    run.summary["tokens_seen"] = tokens_seen
    run.summary["elapsed_seconds"] = elapsed
    if dynamics_enabled and dynamics_probe_inputs is not None:
        observations = capture_observations(
            model,
            dynamics_probe_inputs,
            run_id=str(run.id or run.name),
            checkpoint_tokens=tokens_seen,
            probe_id=os.getenv("AUTORESEARCH_DYNAMICS_PROBE_ID", "validation-head-v1"),
            module_names=_dynamic_module_names(),
        )
        run.summary["dynamics/observation_count"] = len(observations)
        if observations:
            out_dir = Path(__file__).resolve().parent / ".autoresearch" / "dynamics" / str(run.id or "run") / str(tokens_seen)
            log_observations_to_wandb(run, observations, out_dir, update=update)
    run.finish()
    print(f"tokens_seen={tokens_seen} elapsed_seconds={elapsed:.1f}")
    print(f"test_loss={test_loss:.6f}")


if __name__ == "__main__":
    main()
