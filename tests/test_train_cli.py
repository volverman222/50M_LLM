from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_accepts_instrumented_1024_looped_configuration(tmp_path):
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(ROOT / "scripts" / "train_pretrain_1b.py"),
        "--target-tokens",
        "2048",
        "--context-length",
        "128",
        "--micro-batch-size",
        "1",
        "--gradient-accumulation",
        "1",
        "--tokenizer",
        "sp16384",
        "--tokenizer-model",
        str(ROOT / "tokenizers" / "fineweb_16384_bpe.model"),
        "--emb-dim",
        "1024",
        "--n-heads",
        "8",
        "--n-unique-layers",
        "3",
        "--num-loops",
        "2",
        "--positional-encoding",
        "rope",
        "--ff-hidden-dim",
        "1376",
        "--telemetry-jsonl",
        str(tmp_path / "telemetry.jsonl"),
        "--telemetry-every",
        "10",
        "--device",
        "cpu",
        "--dry-run",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Dry run" in result.stdout
    assert "Parámetros:" in result.stdout
