import torch
from llm_mini_lab.models import LoopedGPTModel
from llm_mini_lab.telemetry import LoopTelemetryCollector


def make_model():
    return LoopedGPTModel(
        dict(
            vocab_size=32,
            context_length=8,
            emb_dim=16,
            n_heads=2,
            n_unique_layers=3,
            num_loops=2,
            drop_rate=0.0,
            qkv_bias=False,
            positional_encoding="rope",
            ff_activation="swiglu",
            ff_hidden_dim=24,
        )
    )


def test_boundary_types_and_feedforward_observations():
    model = make_model()
    collector = LoopTelemetryCollector(model)
    collector.start_step()
    model(torch.tensor([[1, 2, 3, 4]]))
    collector.stop_step()
    rows = collector.block_records()
    assert len(rows) == 6
    for row in rows:
        assert row["shape"] == [1, 4, 16]
        assert row["dtype"] == "torch.float32"
        assert row["all_finite"] is True
        assert row["ff_hidden_width"] == 24
        assert row["gate_activation_rms"] >= 0
        assert 0 <= row["gate_sigmoid_extreme_fraction"] <= 1
        assert row["ff_product_rms"] >= 0
    collector.close()


def test_observer_preserves_outputs_gradients_and_rng():
    torch.manual_seed(11)
    model = make_model()
    x = torch.tensor([[1, 2, 3, 4]])
    ref = model(x)
    ref.sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    rng = torch.random.get_rng_state().clone()
    collector = LoopTelemetryCollector(model)
    collector.start_step()
    actual = model(x)
    actual.sum().backward()
    collector.stop_step()
    assert torch.equal(actual, ref)
    for p, grad in zip(model.parameters(), grads):
        assert torch.equal(p.grad, grad)
    assert torch.equal(torch.random.get_rng_state(), rng)
    collector.close()
