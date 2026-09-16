import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


class ToyLoop(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.step.weight.copy_(torch.eye(2))

    def forward(self, x):
        for _ in range(4):
            x = self.step(x)
        return x


class CaptureTests(unittest.TestCase):
    def test_repeated_module_trace_is_passive(self):
        from dynamic_observatory.capture import TraceRecorder

        model = ToyLoop()
        x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        expected = model(x).detach().clone()
        with TraceRecorder(model, module_names=["step"]) as recorder:
            actual = model(x)

        self.assertTrue(torch.equal(actual, expected))
        trajectory = recorder.trajectory("step")
        self.assertEqual(tuple(trajectory.shape), (4, 2))
        self.assertTrue(torch.allclose(trajectory, torch.tensor([[0.5, 0.5]] * 4)))


class MetricTests(unittest.TestCase):
    def test_periodic_rotation_has_low_closure_error(self):
        from dynamic_observatory.metrics import analyze_trajectory

        states = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
            [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],
        ])
        report = analyze_trajectory(states, projection_basis=torch.eye(2), max_period=5)
        self.assertLess(abs(report.radius_drift), 1e-6)
        self.assertGreater(report.phase_coherence, 0.99)
        self.assertLess(report.closure_errors[4], 1e-6)
        self.assertAlmostEqual(report.dominant_frequency, 0.25, places=6)
        self.assertEqual(tuple(report.deltas.shape), (7, 2))
        self.assertEqual(tuple(report.second_deltas.shape), (6, 2))


class ArtifactTests(unittest.TestCase):
    def test_round_trip_preserves_provenance_and_rejects_simulated_source(self):
        from dynamic_observatory.artifact import DynamicObservation, load_observation, write_observation

        observation = DynamicObservation(
            run_id="run-001",
            checkpoint_tokens=2_000_000_000,
            probe_id="fixed-probe-v1",
            trace_name="recurrent.block.0",
            source="DERIVED",
            states=[[1.0, 0.0], [0.0, 1.0]],
            metrics={"phase_coherence": 1.0},
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "observation.json"
            write_observation(path, observation)
            loaded = load_observation(path)
            raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw["schema"], "devpost.dynamic_observation.v1")
        self.assertEqual(loaded.source, "DERIVED")
        self.assertEqual(loaded.checkpoint_tokens, 2_000_000_000)
        with self.assertRaises(ValueError):
            DynamicObservation(
                run_id="r", checkpoint_tokens=1, probe_id="p", trace_name="t",
                source="SIM", states=[], metrics={},
            )


class BoundaryTests(unittest.TestCase):
    def test_dynamic_observer_is_fixed_measurement_apparatus(self):
        program = (ROOT / "program.md").read_text(encoding="utf-8")
        self.assertIn("`src/dynamic_observatory/`", program)
        self.assertIn("fixed measurement apparatus", program.lower())
        self.assertNotIn("may edit `src/dynamic_observatory/`", program.lower())


if __name__ == "__main__":
    unittest.main()


class ProbeIntegrationTests(unittest.TestCase):
    def test_discovers_outermost_repeated_module(self):
        from dynamic_observatory.probe import discover_repeated_modules

        model = ToyLoop()
        x = torch.ones(1, 2, 2)
        names = discover_repeated_modules(model, x, min_calls=2)
        self.assertEqual(names, ["step"])

    def test_capture_restores_training_mode_and_emits_derived_record(self):
        from dynamic_observatory.probe import capture_observations

        model = ToyLoop()
        model.train()
        records = capture_observations(
            model,
            torch.ones(1, 2, 2),
            run_id="run-1",
            checkpoint_tokens=100,
            probe_id="probe-v1",
            module_names=["step"],
        )
        self.assertTrue(model.training)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, "DERIVED")
        self.assertEqual(records[0].trace_name, "step")


class TrainingHookContractTests(unittest.TestCase):
    def test_train_keeps_dynamics_observer_opt_in_and_test_loss_authoritative(self):
        source = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn("AUTORESEARCH_DYNAMICS", source)
        self.assertIn("capture_observations", source)
        self.assertIn('print(f"test_loss={test_loss:.6f}")', source)
        self.assertNotIn("dynamic_score", source)
