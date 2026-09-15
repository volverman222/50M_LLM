import torch

from llm_mini_lab.models import LoopedGPTModel
from llm_mini_lab.telemetry import LoopTelemetryCollector


def tiny_config():
    return {
        "vocab_size": 64,
        "context_length": 16,
        "emb_dim": 32,
        "n_heads": 4,
        "n_unique_layers": 2,
        "num_loops": 3,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "positional_encoding": "rope",
        "ff_activation": "swiglu",
        "ff_hidden_dim": 48,
    }


def test_collector_maps_reused_blocks_to_loop_and_block_indices():
    torch.manual_seed(7)
    model = LoopedGPTModel(tiny_config()).eval()
    collector = LoopTelemetryCollector(model)
    tokens = torch.randint(0, 64, (2, 8))
    collector.start_step()
    model(tokens)
    rows = collector.block_records()
    assert [(row["loop"], row["block"]) for row in rows] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    ]
    assert all(row["hidden_rms"] > 0 for row in rows)
    assert all(row["delta_rms"] >= 0 for row in rows)
    assert all(-1.0 <= row["cosine_in_out"] <= 1.0 for row in rows)
    collector.close()


def test_collector_builds_one_summary_per_loop_and_is_observational():
    torch.manual_seed(11)
    model = LoopedGPTModel(tiny_config()).eval()
    tokens = torch.randint(0, 64, (1, 8))
    baseline = model(tokens)
    collector = LoopTelemetryCollector(model)
    collector.start_step()
    observed = model(tokens)
    loops = collector.loop_records()
    assert torch.allclose(observed, baseline)
    assert [row["loop"] for row in loops] == [0, 1, 2]
    assert all(row["loop_delta_rms"] >= 0 for row in loops)
    assert all(-1.0 <= row["loop_cosine"] <= 1.0 for row in loops)
    collector.start_step()
    assert collector.block_records() == []
    assert collector.loop_records() == []
    collector.close()


def test_parameter_metrics_report_gradient_and_true_update_size():
    from llm_mini_lab.telemetry import (
        gradient_rms,
        parameter_rms,
        parameter_update_rms,
        snapshot_parameters,
    )

    layer = torch.nn.Linear(3, 2, bias=False)
    before = snapshot_parameters(layer)
    loss = layer(torch.ones(1, 3)).sum()
    loss.backward()
    assert gradient_rms(layer) > 0
    assert parameter_rms(layer) > 0
    with torch.no_grad():
        layer.weight.add_(0.25)
    update_rms = parameter_update_rms(layer, before)
    assert abs(update_rms - 0.25) < 1e-6
