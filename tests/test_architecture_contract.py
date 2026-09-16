import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


class ArchitectureContractTests(unittest.TestCase):
    def test_editable_architecture_builds_language_model_contract(self):
        from rsi_architecture import build_model

        cfg = {
            "vocab_size": 128,
            "context_length": 16,
            "emb_dim": 32,
            "n_heads": 4,
            "n_unique_layers": 2,
            "num_loops": 2,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "positional_encoding": "rope",
            "ff_activation": "gelu",
            "ff_hidden_dim": 64,
            "tie_embeddings": True,
        }
        model = build_model(cfg)
        tokens = torch.randint(0, cfg["vocab_size"], (2, cfg["context_length"]))
        logits = model(tokens)
        self.assertEqual(logits.shape, (2, cfg["context_length"], cfg["vocab_size"]))
        self.assertLessEqual(sum(p.numel() for p in model.parameters()), 50_000_000)


    def test_train_uses_editable_architecture_factory(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn("from rsi_architecture import build_model", source)
        self.assertIn("model = build_model(cfg)", source)
        self.assertNotIn("LoopedGPTModel(cfg)", source)
        self.assertNotIn("model.out_head.weight = model.tok_emb.weight", source)

    def test_program_unlocks_only_candidate_architecture(self):
        program = (ROOT / "program.md").read_text(encoding="utf-8")
        self.assertIn("`rsi_architecture/`", program)
        self.assertIn("Code under `src/llm_mini_lab/`", program)
        self.assertIn("may create additional Python modules inside `rsi_architecture/`", program)


if __name__ == "__main__":
    unittest.main()
