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


class ReferenceFrameTests(unittest.TestCase):
    def test_fixed_reference_frame_preserves_cross_run_displacement(self):
        from dynamic_observatory.reference import build_reference_frame, project_states

        baseline = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        frame = build_reference_frame(
            baseline,
            probe_id="probe-v1",
            trace_name="step",
            run_id="baseline",
            checkpoint_tokens=100,
        )
        shifted = baseline + torch.tensor([4.0, 0.0, 0.0])
        p0 = project_states(baseline, frame)
        p1 = project_states(shifted, frame)
        displacement = torch.linalg.vector_norm(p1.mean(0) - p0.mean(0))
        self.assertGreater(float(displacement), 1.0)
        self.assertEqual(frame.schema, "devpost.dynamic_reference_frame.v1")
    def test_reference_frame_round_trip(self):
        from dynamic_observatory.reference import (
            build_reference_frame,
            load_reference_frame,
            write_reference_frame,
        )

        states = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        frame = build_reference_frame(
            states,
            probe_id="probe-v1",
            trace_name="step",
            run_id="baseline",
            checkpoint_tokens=123,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "frame.json"
            write_reference_frame(path, frame)
            loaded = load_reference_frame(path)
        self.assertEqual(loaded.run_id, "baseline")
        self.assertEqual(loaded.checkpoint_tokens, 123)
        self.assertEqual(loaded.trace_name, "step")
        self.assertEqual(len(loaded.center), 2)


class ViewerContractTests(unittest.TestCase):
    def test_offline_3d_viewer_has_local_file_workflow_and_no_network_fetch(self):
        viewer = ROOT / "tools" / "dynamic_observatory_viewer" / "index.html"
        self.assertTrue(viewer.is_file())
        source = viewer.read_text(encoding="utf-8")
        self.assertIn('id="scene"', source)
        self.assertIn('id="observation-files"', source)
        self.assertIn("requestAnimationFrame", source)
        self.assertIn("FileReader", source)
        self.assertIn("FIXED_REFERENCE", source)
        self.assertNotIn("fetch(", source)
        self.assertNotIn("SIM", source)


class FixedFrameIntegrationTests(unittest.TestCase):
    def test_analyzer_uses_reference_center_not_per_run_center(self):
        from dynamic_observatory.metrics import analyze_trajectory

        states = torch.tensor([[5.0, 0.0], [5.0, 1.0], [5.0, 2.0]])
        report = analyze_trajectory(
            states,
            projection_basis=torch.eye(2),
            projection_center=torch.zeros(2),
        )
        self.assertGreater(float(report.projected[:, 0].mean()), 4.9)

    def test_capture_accepts_typed_reference_frame(self):
        from dynamic_observatory.probe import capture_observations
        from dynamic_observatory.reference import build_reference_frame

        model = ToyLoop()
        probe = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        baseline = torch.tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        frame = build_reference_frame(
            baseline + torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [-0.1, 0.0]]),
            probe_id="probe-v1", trace_name="step", run_id="base", checkpoint_tokens=1,
        )
        records = capture_observations(
            model,
            probe,
            run_id="candidate",
            checkpoint_tokens=2,
            probe_id="probe-v1",
            module_names=["step"],
            reference_frames={"step": frame},
        )
        self.assertEqual(records[0].metrics["projection_basis"], "FIXED_REFERENCE")
        self.assertEqual(records[0].metrics["reference_run_id"], "base")
        self.assertEqual(records[0].metrics["reference_checkpoint_tokens"], 1)


class ReferenceCliTests(unittest.TestCase):
    def test_reference_command_builds_frame_from_observation(self):
        from dynamic_observatory.artifact import DynamicObservation, write_observation
        from dynamic_observatory.cli import main as cli_main
        from dynamic_observatory.reference import load_reference_frame

        obs = DynamicObservation(
            run_id="baseline", checkpoint_tokens=100, probe_id="probe-v1",
            trace_name="step", source="DERIVED",
            states=[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], metrics={},
        )
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "obs.json"
            dst = Path(td) / "frame.json"
            write_observation(src, obs)
            rc = cli_main(["reference", str(src), str(dst)])
            frame = load_reference_frame(dst)
        self.assertEqual(rc, 0)
        self.assertEqual(frame.run_id, "baseline")
        self.assertEqual(frame.probe_id, "probe-v1")
