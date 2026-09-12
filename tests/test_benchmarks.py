import pytest
import torch

from llm_mini_lab.evaluation.suite import (
    adapt_arc_easy,
    adapt_piqa,
    adapt_winogrande,
    evaluate_benchmark,
)


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


def test_arc_adapter_resolves_non_positional_answer_key():
    adapted = adapt_arc_easy(
        {
            "question": "Which one?",
            "choices": {"text": ["wrong", "right"], "label": ["B", "D"]},
            "answerKey": "D",
        }
    )
    assert adapted.label == 1
    assert adapted.context == "Question: Which one?\nAnswer:"
    assert adapted.choices == [" wrong", " right"]


def test_piqa_adapter_uses_zero_based_label():
    adapted = adapt_piqa({"goal": "Do it", "sol1": "bad", "sol2": "good", "label": 1})
    assert adapted.label == 1
    assert adapted.choices == [" bad", " good"]


def test_winogrande_scores_option_and_sentence_suffix():
    adapted = adapt_winogrande(
        {"sentence": "The trophy does not fit in the _ because it is large.",
         "option1": "suitcase", "option2": "trophy", "answer": "2"}
    )
    assert adapted.context == "The trophy does not fit in the "
    assert adapted.choices[1] == "trophy because it is large."
    assert adapted.label == 1


@pytest.mark.parametrize(
    ("name", "example"),
    [
        ("arc_easy", {"question": "q", "choices": {"text": ["a", "b"], "label": ["A", "B"]}, "answerKey": "A"}),
        ("piqa", {"goal": "q", "sol1": "a", "sol2": "b", "label": 0}),
        ("winogrande", {"sentence": "q _", "option1": "a", "option2": "b", "answer": "1"}),
    ],
)
def test_evaluate_injected_examples_without_downloading(name, example):
    tokenizer = CharacterTokenizer()
    preferred = tokenizer.encode(" a")[-1]
    model = CandidateModel(preferred).train()
    result = evaluate_benchmark(
        name, model, tokenizer, "cpu", examples=[example], context_length=64
    )
    assert result["accuracy"] == 1.0
    assert result["total"] == 1
    assert model.training


def test_unknown_benchmark_has_clear_error():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        evaluate_benchmark("unknown", object(), object(), "cpu", examples=[])
