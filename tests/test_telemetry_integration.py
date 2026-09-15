import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from llm_mini_lab.models import LoopedGPTModel
from llm_mini_lab.telemetry import LoopTelemetryCollector

ROOT = Path(__file__).resolve().parents[1]


def trainer():
    spec = importlib.util.spec_from_file_location(
        "telemetry_test_trainer", ROOT / "scripts/train_pretrain_1b.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("kv_heads", [4, 2])
@pytest.mark.parametrize("activation", ["swiglu", "gelu"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_observer_preserves_gqa_mha_dropout_gradients_and_rng(
    kv_heads, activation, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(44)
    cfg = dict(
        vocab_size=64,
        context_length=8,
        emb_dim=32,
        n_heads=4,
        n_kv_heads=kv_heads,
        n_unique_layers=2,
        num_loops=2,
        drop_rate=0.15,
        qkv_bias=False,
        positional_encoding="rope",
        ff_activation=activation,
        ff_hidden_dim=48,
        tie_embeddings=True,
    )
    model = LoopedGPTModel(cfg).to(device).train()
    tokens = torch.arange(8, device=device).reshape(1, 8)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    reference = model(tokens)
    reference.square().sum().backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    expected_cpu = torch.get_rng_state().clone()
    expected_cuda = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng)
    observer = LoopTelemetryCollector(model)
    try:
        observer.start_step()
        actual = model(tokens)
        observer.stop_step()
        actual.square().sum().backward()
        assert torch.equal(reference, actual)
        assert all(
            torch.equal(g, p.grad) for g, p in zip(gradients, model.parameters())
        )
        assert torch.equal(torch.get_rng_state(), expected_cpu)
        if expected_cuda is not None:
            assert torch.equal(torch.cuda.get_rng_state(), expected_cuda)
        assert len(observer.block_records()) == 4
        assert len(observer.loop_records()) == 2
        assert model.out_head.weight is model.tok_emb.weight
    finally:
        observer.close()
    assert not model.trf_blocks[0]._forward_hooks
    assert not model.trf_blocks[0]._forward_pre_hooks


def test_jsonl_preserves_nonfinite_observation_as_null(tmp_path):
    mod = trainer()
    path = tmp_path / "telemetry.jsonl"
    mod.append_jsonl(
        path, [{"kind": "update", "loss": float("nan"), "grad_norm": float("inf")}]
    )
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    record = json.loads(text)
    assert record["loss"] is None and record["grad_norm"] is None
    assert record["nonfinite_fields"] == ["loss", "grad_norm"]


def test_cli_exposes_kv_heads_without_changing_defaults(monkeypatch):
    mod = trainer()
    monkeypatch.setattr(sys, "argv", ["trainer", "--n-kv-heads", "2"])
    args = mod.parse_args()
    assert args.n_kv_heads == 2
    monkeypatch.setattr(sys, "argv", ["trainer"])
    args = mod.parse_args()
    assert args.n_kv_heads is None and args.telemetry_jsonl is None


@pytest.mark.parametrize("kv_heads", [4, 2])
def test_synthetic_train_equivalence_and_no_telemetry_overwrite(
    tmp_path, monkeypatch, kv_heads
):
    mod = trainer()
    from torch.utils.data import DataLoader, TensorDataset

    tokens = torch.arange(64).reshape(8, 8) % 63
    dataset = TensorDataset(tokens, (tokens + 1) % 64)
    monkeypatch.setattr(mod, "load_tokenizer", lambda *a: object())
    monkeypatch.setattr(mod, "tokenizer_vocab_size", lambda *a: 64)
    monkeypatch.setattr(
        mod,
        "create_stream_loaders",
        lambda *a, **k: (
            DataLoader(dataset, batch_size=2),
            DataLoader(dataset, batch_size=2),
        ),
    )
    monkeypatch.setattr(mod, "generate_text_simple", lambda model, idx, **k: idx)
    monkeypatch.setattr(mod, "text_to_token_ids", lambda *a: torch.tensor([[1]]))
    monkeypatch.setattr(mod, "token_ids_to_text", lambda *a: "synthetic smoke")
    common = [
        "trainer",
        "--target-tokens",
        "64",
        "--context-length",
        "8",
        "--emb-dim",
        "32",
        "--n-heads",
        "4",
        "--n-kv-heads",
        str(kv_heads),
        "--n-unique-layers",
        "2",
        "--num-loops",
        "2",
        "--ff-hidden-dim",
        "48",
        "--micro-batch-size",
        "2",
        "--gradient-accumulation",
        "2",
        "--device",
        "cpu",
        "--hourly-cost",
        "0",
        "--save-every",
        "100",
        "--benchmark-every",
        "0",
        "--hf-upload-every",
        "0",
        "--eval-every",
        "1",
        "--eval-batches",
        "1",
        "--log-every",
        "1",
    ]
    logs = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(
        sys, "argv", common + ["--checkpoint-dir", str(tmp_path / "reference")]
    )
    mod.main("smollm")
    monkeypatch.setattr(
        sys,
        "argv",
        common
        + [
            "--checkpoint-dir",
            str(tmp_path / "observed"),
            "--telemetry-jsonl",
            str(logs),
            "--telemetry-every",
            "1",
        ],
    )
    mod.main("smollm")
    a = torch.load(
        tmp_path / "reference/final.pt", map_location="cpu", weights_only=True
    )
    b = torch.load(
        tmp_path / "observed/final.pt", map_location="cpu", weights_only=True
    )
    assert all(
        torch.equal(a["model_state_dict"][k], v)
        for k, v in b["model_state_dict"].items()
    )
    records = [json.loads(s) for s in logs.read_text().splitlines()]
    assert len([r for r in records if r["kind"] == "update"]) == 2
    assert len([r for r in records if r["kind"] == "block"]) == 8
    before = logs.read_bytes()

    def forbidden_attach(_):
        raise AssertionError(
            "Existing evidence must be rejected before attaching observers"
        )

    monkeypatch.setattr(mod, "LoopTelemetryCollector", forbidden_attach)
    with pytest.raises(FileExistsError):
        mod.main("smollm")
    assert logs.read_bytes() == before
