import torch

from llm_mini_lab.hellaswag import evaluate_hellaswag, render_hellaswag_example


class CharacterTokenizer:
    def encode(self, text):
        return [ord(char) % 32 + 1 for char in text]


class CandidateModel(torch.nn.Module):
    def __init__(self, preferred_token):
        super().__init__()
        self.preferred_token = preferred_token

    def forward(self, tokens):
        logits = torch.zeros(*tokens.shape, 64, device=tokens.device)
        logits[..., self.preferred_token] = 10.0
        return logits


def test_render_truncates_context_before_completion():
    tokenizer = CharacterTokenizer()
    example = {"ctx": "a" * 20, "endings": ["b", "c", "d", "e"], "label": 0}
    tokens, mask, label = render_hellaswag_example(
        example, tokenizer, context_length=8
    )
    assert tokens.shape == (4, 8)
    assert mask.sum(dim=1).tolist() == [2.0, 2.0, 2.0, 2.0]
    assert label == 0


def test_evaluate_returns_accuracy_and_restores_training_state():
    tokenizer = CharacterTokenizer()
    preferred = tokenizer.encode(" a")[-1]
    model = CandidateModel(preferred).train()
    examples = [
        {"ctx": "x", "endings": ["a", "b", "c", "d"], "label": 0},
        {"ctx": "y", "endings": ["b", "a", "c", "d"], "label": 1},
    ]
    metrics = evaluate_hellaswag(
        model, tokenizer, "cpu", examples=examples, context_length=16
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["correct"] == 2
    assert metrics["total"] == 2
    assert model.training
