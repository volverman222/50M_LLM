import hashlib
import hmac
import json
import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch_feedback.contract import ContractError, parse_external_analysis
from autoresearch_feedback.security import verify_wandb_signature
from autoresearch_feedback.store import AnalysisConflict, AnalysisInbox
from autoresearch_feedback.server import FeedbackService


def payload(analysis_id="aria-001", objective="lower validation loss"):
    return {
        "schema": "devpost.autoresearch.external_analysis.v1",
        "source": "wandb.aria",
        "entity": "volverman22-universidad-aut-noma-de-madrid-org",
        "project": "gpt2-50M",
        "run_id": "abc123",
        "run_name": "baseline-r00",
        "analysis_id": analysis_id,
        "analysis_version": 1,
        "baseline_refs": ["baseline-r00"],
        "observations": ["test_loss improved"],
        "anomalies": [],
        "regressions": [],
        "improvements": ["lower test_loss"],
        "hypotheses": [
            {"claim": "smaller LR may improve late training", "evidence": ["loss tail"], "confidence": 0.65}
        ],
        "recommended_experiment": {
            "objective": objective,
            "independent_variables": {"learning_rate": 0.0002},
            "controlled_variables": {"max_tokens": 1000000},
            "expected_observation": "lower test_loss",
            "falsification_condition": "test_loss is not lower than baseline",
        },
    }


class SignatureTests(unittest.TestCase):
    def test_exact_body_hmac_sha256(self):
        body = b'{"hello":"world"}'
        secret = "shared-secret"
        sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        self.assertTrue(verify_wandb_signature(body, sig, secret))
        self.assertTrue(verify_wandb_signature(body, "sha256=" + sig, secret))
        self.assertFalse(verify_wandb_signature(body + b" ", sig, secret))
        self.assertFalse(verify_wandb_signature(body, "00" * 32, secret))


class ContractTests(unittest.TestCase):
    def test_parses_typed_advisory_proposal(self):
        analysis = parse_external_analysis(payload())
        self.assertEqual(analysis.analysis_id, "aria-001")
        self.assertEqual(analysis.recommended_experiment.independent_variables["learning_rate"], 0.0002)
        self.assertEqual(analysis.authority, "advisory_only")

    def test_rejects_wrong_schema_or_source(self):
        bad = payload()
        bad["schema"] = "other.v1"
        with self.assertRaises(ContractError):
            parse_external_analysis(bad)
        bad = payload()
        bad["source"] = "untrusted.agent"
        with self.assertRaises(ContractError):
            parse_external_analysis(bad)


class StoreTests(unittest.TestCase):
    def test_idempotent_duplicate_and_conflict(self):
        with tempfile.TemporaryDirectory() as td:
            inbox = AnalysisInbox(Path(td))
            first = inbox.ingest(payload())
            duplicate = inbox.ingest(payload())
            self.assertEqual(first.status, "accepted")
            self.assertEqual(duplicate.status, "duplicate")
            loaded = inbox.load(first.digest)
            self.assertEqual(loaded.analysis_id, "aria-001")
            self.assertEqual([r.digest for r in inbox.list_receipts()], [first.digest])
            changed = payload(objective="different proposal")
            with self.assertRaises(AnalysisConflict):
                inbox.ingest(changed)

    def test_analysis_id_cannot_escape_inbox(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "inbox"
            inbox = AnalysisInbox(root)
            receipt = inbox.ingest(payload(analysis_id="../../escape"))
            self.assertTrue((root / "analyses" / f"{receipt.digest}.json").is_file())
            self.assertFalse((Path(td) / "escape").exists())
            for path in root.rglob("*"):
                self.assertTrue(path.resolve().is_relative_to(root.resolve()))


class ServiceTests(unittest.TestCase):
    def test_signed_request_is_stored_but_not_executed(self):
        with tempfile.TemporaryDirectory() as td:
            body = json.dumps(payload(), sort_keys=True, separators=(",", ":")).encode()
            secret = "shared-secret"
            sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            service = FeedbackService(AnalysisInbox(Path(td)), webhook_secret=secret)
            status, response = service.process(
                path="/v1/wandb/aria",
                headers={"content-type": "application/json", "x-wandb-signature": sig},
                body=body,
            )
            self.assertEqual(status, 202)
            self.assertEqual(response["authority"], "advisory_only")
            self.assertEqual(response["status"], "accepted")
            self.assertNotIn("execute", response)

    def test_rejects_bad_signature_and_optional_bearer(self):
        with tempfile.TemporaryDirectory() as td:
            body = json.dumps(payload()).encode()
            service = FeedbackService(
                AnalysisInbox(Path(td)), webhook_secret="secret", bearer_token="token"
            )
            status, _ = service.process(
                path="/v1/wandb/aria",
                headers={"content-type": "application/json", "x-wandb-signature": "bad"},
                body=body,
            )
            self.assertEqual(status, 401)
            sig = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
            status, _ = service.process(
                path="/v1/wandb/aria",
                headers={"content-type": "application/json", "x-wandb-signature": sig},
                body=body,
            )
            self.assertEqual(status, 401)
            status, _ = service.process(
                path="/v1/wandb/aria",
                headers={
                    "content-type": "application/json",
                    "x-wandb-signature": sig,
                    "authorization": "Bearer token",
                },
                body=body,
            )
            self.assertEqual(status, 202)

    def test_optional_entity_and_project_allowlist(self):
        with tempfile.TemporaryDirectory() as td:
            body = json.dumps(payload()).encode()
            secret = "secret"
            sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            service = FeedbackService(
                AnalysisInbox(Path(td)),
                webhook_secret=secret,
                expected_entity="expected-entity",
                expected_project="expected-project",
            )
            status, response = service.process(
                path="/v1/wandb/aria",
                headers={"content-type": "application/json", "x-wandb-signature": sig},
                body=body,
            )
            self.assertEqual(status, 403)
            self.assertEqual(response["error"], "source_scope_mismatch")


if __name__ == "__main__":
    unittest.main()
