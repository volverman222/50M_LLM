import pytest
import torch

from llm_mini_lab.models.gpt import GPTModel, LoopedGPTModel


def _config(**overrides):
    config = {
        "vocab_size": 128,
        "context_length": 16,
        "emb_dim": 32,
        "n_heads": 4,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
        "positional_encoding": "learned",
        "tie_embeddings": True,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("model_class", "config"),
    [
        (GPTModel, _config()),
        (LoopedGPTModel, _config(n_unique_layers=2, num_loops=2)),
    ],
)
def test_input_and_output_embeddings_share_the_same_parameter(model_class, config):
    model = model_class(config)

    assert model.out_head.weight is model.tok_emb.weight

    with torch.no_grad():
        model.tok_emb.weight[0, 0] = 123.0
    assert model.out_head.weight[0, 0].item() == 123.0


def test_weight_tying_removes_one_vocabulary_matrix_from_parameter_count():
    tied = GPTModel(_config(tie_embeddings=True))
    untied = GPTModel(_config(tie_embeddings=False))

    tied_count = sum(parameter.numel() for parameter in tied.parameters())
    untied_count = sum(parameter.numel() for parameter in untied.parameters())

    assert untied_count - tied_count == 128 * 32
