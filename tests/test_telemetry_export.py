import importlib
import importlib.util
import json
from types import SimpleNamespace
import pytest


def module():
    assert importlib.util.find_spec("llm_mini_lab.telemetry_export"), (
        "Export adapter not implemented"
    )
    return importlib.import_module("llm_mini_lab.telemetry_export")


def records():
    return [
        {
            "kind": "config",
            "config": {"emb_dim": 32, "tokenizer_name": "sp16384"},
            "args": {
                "seed": 123,
                "tokenizer_model": "/private/model",
                "api_key": "do-not-export",
            },
        },
        {
            "kind": "update",
            "update": 1,
            "tokens_seen": 32,
            "loss": 2.5,
            "grad_norm": float("nan"),
        },
        {
            "kind": "block",
            "update": 1,
            "loop": 0,
            "block": 0,
            "hidden_rms": 1.2,
            "shape": [1, 8, 32],
            "device": "cuda:0",
            "secret": 123,
        },
        {"kind": "loop", "update": 1, "loop": 0, "loop_delta_rms": 0.3},
        {"kind": "validation", "update": 1, "tokens_seen": 32, "loss": 2.4},
        {"kind": "update", "update": 2, "tokens_seen": 64, "loss": 2.3},
    ]


def source(tmp_path, rows=None):
    p = tmp_path / "telemetry.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in (rows or records())))
    return p


def test_export_groups_updates_and_only_exports_approved_values(tmp_path):
    m = module()
    data = m.load_export(source(tmp_path))
    assert len(data["payloads"]) == 2
    first = data["payloads"][0]
    assert first["update"] == 1 and first["telemetry/update.loss"] == 2.5
    assert first["telemetry/loop_0_block_0.hidden_rms"] == 1.2
    assert first["telemetry/validation.loss"] == 2.4
    assert "telemetry/update.grad_norm" not in first
    text = json.dumps(data)
    assert (
        "do-not-export" not in text and "/private" not in text and "secret" not in text
    )
    assert data["config"]["seed"] == 123


@pytest.mark.parametrize(
    "tail",
    [
        [{"kind": "update", "update": 0, "loss": 2.0}],
        [{"kind": "update", "update": 2, "loss": 4.0}],
        [{"kind": "config", "args": {"seed": 5}}],
    ],
)
def test_rejects_conflicting_or_rewound_records(tmp_path, tail):
    m = module()
    with pytest.raises(ValueError):
        m.load_export(source(tmp_path, records() + tail))


def test_rejects_incomplete_jsonl_instead_of_losing_last_record(tmp_path):
    m = module()
    p = tmp_path / "partial.jsonl"
    p.write_text('{"kind":"update"')
    with pytest.raises(ValueError, match="line|record"):
        m.load_export(p)


def test_export_is_offline_and_receipt_prevents_duplicate_attempts(tmp_path):
    m = module()
    calls = []
    run = SimpleNamespace(
        id="local-test",
        url=None,
        define_metric=lambda *a, **k: None,
        log=lambda data: calls.append(data),
        finish=lambda **k: calls.append(k),
    )
    sdk = SimpleNamespace(
        Settings=lambda **kw: kw, init=lambda **kw: (calls.append(kw) or run)
    )
    path = source(tmp_path)
    receipt = tmp_path / "receipt.json"
    result = m.export_telemetry(
        path, project="test", output_dir=tmp_path / "wb", receipt=receipt, sdk=sdk
    )
    assert (
        calls[0]["mode"] == "offline" and calls[0]["settings"]["disable_code"] is True
    )
    assert calls[0]["settings"]["disable_git"] is True
    assert result["status"] == "completed" and result["payload_count"] == 2
    with pytest.raises(FileExistsError):
        m.export_telemetry(
            path, project="test", output_dir=tmp_path / "wb", receipt=receipt, sdk=sdk
        )


def test_online_requires_explicit_permission(tmp_path):
    m = module()
    with pytest.raises(ValueError, match="allow_online"):
        m.export_telemetry(
            source(tmp_path),
            project="test",
            mode="online",
            output_dir=tmp_path / "wb",
            receipt=tmp_path / "receipt.json",
        )


def test_failed_sink_is_not_reported_successful(tmp_path):
    m = module()
    finished = []

    def fail(data):
        raise RuntimeError("simulated sensitive message")

    run = SimpleNamespace(
        id="test",
        url=None,
        define_metric=lambda *a, **k: None,
        log=fail,
        finish=lambda **k: finished.append(k),
    )
    sdk = SimpleNamespace(Settings=lambda **k: k, init=lambda **k: run)
    receipt = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError):
        m.export_telemetry(
            source(tmp_path),
            project="test",
            output_dir=tmp_path / "wb",
            receipt=receipt,
            sdk=sdk,
        )
    r = json.loads(receipt.read_text())
    assert r["status"] == "failed" and r["error_type"] == "RuntimeError"
    assert "sensitive" not in receipt.read_text() and finished == [{"exit_code": 1}]


def test_preview_has_no_side_effects(tmp_path):
    import subprocess, sys
    from pathlib import Path

    output = tmp_path / "not-created"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/export_telemetry_wandb.py"),
        str(source(tmp_path)),
        "--project",
        "test",
        "--output-dir",
        str(output),
        "--dry-run",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "preview"
    assert not output.exists()


def test_credential_variable_not_consumed_in_offline_mode(tmp_path, monkeypatch):
    m = module()
    calls = []
    monkeypatch.setenv("EXAMPLE_TEST_KEY", "do-not-log-this")
    run = SimpleNamespace(
        id="test",
        define_metric=lambda *a, **k: None,
        log=lambda *a: None,
        finish=lambda **k: None,
    )
    sdk = SimpleNamespace(
        Settings=lambda **k: (calls.append(k) or k), init=lambda **k: run
    )
    result = m.export_telemetry(
        source(tmp_path),
        project="test",
        output_dir=tmp_path / "wb",
        receipt=tmp_path / "r.json",
        api_key_env="EXAMPLE_TEST_KEY",
        sdk=sdk,
    )
    assert "api_key" not in calls[0] and "do-not-log-this" not in json.dumps(result)


def test_rejects_unknown_schema_version(tmp_path):
    m = module()
    rows = records()
    rows[0]["schema_version"] = 999
    with pytest.raises(ValueError, match="schema"):
        m.load_export(source(tmp_path, rows))


def test_dataset_label_and_nonfinite_count_remain_visible(tmp_path):
    m = module()
    rows = records()
    rows[0]["dataset"] = "smollm"
    rows[0]["args"]["smollm_config"] = "cosmopedia-v2"
    data = m.load_export(source(tmp_path, rows))
    assert data["config"]["dataset"] == "smollm"
    assert data["config"]["smollm_config"] == "cosmopedia-v2"
    assert data["payloads"][0]["telemetry/update.nonfinite_count"] == 1
