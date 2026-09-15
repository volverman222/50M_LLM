"""Opt-in scalar export; local JSONL remains the canonical training evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

_UPDATE = frozenset(
    "tokens_seen loss lr grad_norm grad_rms clipped parameter_rms update_rms update_to_parameter_rms step_ms tokens_per_second peak_allocated_gb peak_reserved_gb".split()
)
_BLOCK = frozenset(
    "attention_rms mlp_rms hidden_rms delta_rms cosine_in_out mean std abs_max all_finite ff_hidden_width gate_activation_rms gate_negative_fraction gate_sigmoid_extreme_fraction ff_value_rms ff_product_rms".split()
)
_LOOP = frozenset("loop_entry_rms loop_exit_rms loop_delta_rms loop_cosine".split())
_CONFIG_NUMERIC = frozenset(
    "vocab_size context_length emb_dim n_heads n_kv_heads n_unique_layers num_loops ff_hidden_dim drop_rate qkv_bias tie_embeddings seed target_tokens micro_batch_size gradient_accumulation lr learning_rate weight_decay grad_clip warmup_ratio min_lr_ratio telemetry_every parameter_count".split()
)
_CONFIG_CHOICES = {
    "dataset": {"fineweb", "smollm"},
    "tokenizer_name": {"gpt2", "sp16384"},
    "tokenizer": {"gpt2", "sp16384"},
    "positional_encoding": {"rope", "learned"},
    "ff_activation": {"swiglu", "gelu"},
}


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _index(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def public_config(record: dict[str, Any]) -> dict[str, Any]:
    """Allowlist model/training values; never forward raw args, paths or credentials."""
    result: dict[str, Any] = {}
    for source in (record.get("config", {}), record.get("args", {}), record):
        for key, value in source.items():
            if key in _CONFIG_NUMERIC and _numeric(value):
                result[key] = value
            elif (
                key in _CONFIG_CHOICES
                and isinstance(value, str)
                and value in _CONFIG_CHOICES[key]
            ):
                result[key] = value
            elif (
                key == "smollm_config"
                and isinstance(value, str)
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value)
            ):
                result[key] = value
    return result


def to_wandb_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One payload per optimizer update, with separate loop/block namespaces."""
    payload: dict[str, Any] = {}
    for record in records:
        kind = record.get("kind")
        if kind not in {"update", "block", "loop", "validation"}:
            continue
        update = _index(record.get("update"), "update")
        if "update" in payload and payload["update"] != update:
            raise ValueError("One payload must contain exactly one optimizer update")
        payload["update"] = update
        if kind == "block":
            loop, block = (
                _index(record.get("loop"), "loop"),
                _index(record.get("block"), "block"),
            )
            prefix, allowed = f"loop_{loop}_block_{block}", _BLOCK
        elif kind == "loop":
            prefix, allowed = f"loop_{_index(record.get('loop'), 'loop')}", _LOOP
        else:
            prefix = kind
            allowed = _UPDATE if kind == "update" else {"loss", "tokens_seen"}
        nonfinite_count = sum(
            (isinstance(record.get(key), float) and not math.isfinite(record[key]))
            or key in record.get("nonfinite_fields", [])
            for key in allowed
        )
        if nonfinite_count:
            payload[f"telemetry/{prefix}.nonfinite_count"] = nonfinite_count
        for key in allowed:
            value = record.get(key)
            if not _numeric(value):
                continue
            metric = f"telemetry/{prefix}.{key}"
            if metric in payload and payload[metric] != value:
                raise ValueError(f"Conflicting metric at update {update}: {metric}")
            payload[metric] = value
    return payload


def load_export(path: Path | str) -> dict[str, Any]:
    """Validate a complete bounded snapshot before any SDK or network operation."""
    with Path(path).open("rb") as stream:
        raw = stream.read(128 * 1024 * 1024 + 1)
    if len(raw) > 128 * 1024 * 1024:
        raise ValueError("Telemetry snapshot exceeds the 128 MiB export limit")
    if not raw.endswith(b"\n"):
        raise ValueError("Unterminated final record; export a completed JSONL snapshot")
    config, config_seen, payloads, group = {}, False, [], []
    last_update = -1
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid JSON at line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected object record at line {line_number}")
        if record.get("kind") == "config":
            if config_seen or last_update >= 0:
                raise ValueError("Config must occur once, before measurement records")
            if record.get("schema_version", 1) != 1:
                raise ValueError("Unsupported telemetry schema version")
            config, config_seen = public_config(record), True
            continue
        if record.get("kind") not in {"update", "block", "loop", "validation"}:
            raise ValueError(f"Unsupported record kind at line {line_number}")
        update = _index(record.get("update"), "update")
        if update < last_update:
            raise ValueError("Optimizer update order moved backwards")
        if group and update != last_update:
            payloads.append(to_wandb_metrics(group))
            group = []
        group.append(record)
        last_update = update
    if group:
        payloads.append(to_wandb_metrics(group))
    if not payloads:
        raise ValueError("No measurement records to export")
    return {
        "config": config,
        "payloads": payloads,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _save_receipt(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def export_telemetry(
    path: Path | str,
    *,
    project: str,
    output_dir: Path | str,
    receipt: Path | str,
    entity: str | None = None,
    name: str | None = None,
    mode: str = "offline",
    allow_online: bool = False,
    api_key_env: str | None = None,
    sdk: Any = None,
) -> dict[str, Any]:
    """Export once. Online operation requires explicit approval; no model artifacts."""
    if mode not in {"offline", "online"}:
        raise ValueError("mode must be offline or online")
    if mode == "online" and not allow_online:
        raise ValueError("Online export requires allow_online=True")
    if not project.strip():
        raise ValueError("A nonempty W&B project is required")
    data = load_export(path)
    receipt, output_dir = Path(receipt), Path(output_dir)
    if receipt.resolve() == Path(path).resolve():
        raise ValueError("Receipt must not replace input evidence")
    key = None
    if mode == "online" and api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError("The requested credential environment variable is not set")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "started",
        "source_sha256": data["source_sha256"],
        "mode": mode,
        "project": project,
        "entity": entity,
        "payload_count": 0,
    }
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    run = None
    finished = False
    try:
        if sdk is None:
            import wandb as sdk
        output_dir.mkdir(parents=True, exist_ok=True)
        settings = {
            "disable_code": True,
            "disable_git": True,
            "console": "off",
            "x_disable_stats": True,
            "x_disable_meta": True,
            "x_save_requirements": False,
        }
        if key is not None:
            settings["api_key"] = key
        config = {**data["config"], "telemetry_source_sha256": data["source_sha256"]}
        kwargs = dict(
            project=project,
            entity=entity,
            name=name,
            mode=mode,
            dir=str(output_dir),
            config=config,
            settings=sdk.Settings(**settings),
        )
        if mode == "online":
            kwargs.update(id="telemetry-" + data["source_sha256"][:24], resume="never")
        run = sdk.init(**kwargs)
        run.define_metric("update")
        run.define_metric("telemetry/*", step_metric="update")
        for payload in data["payloads"]:
            run.log(payload)
            result["payload_count"] += 1
        run.finish(exit_code=0)
        finished = True
        result.update(status="completed", wandb_run_id=run.id)
        _save_receipt(receipt, result)
        return result
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__)
        _save_receipt(receipt, result)
        raise
    finally:
        if run is not None and not finished:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass  # Keep the original failure and durable failed receipt.
