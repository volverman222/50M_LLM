from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contract import ExternalAnalysis, parse_external_analysis


class AnalysisConflict(RuntimeError):
    """Raised when one analysis_id is reused for different content."""


@dataclass(frozen=True)
class IngestReceipt:
    analysis_id: str
    digest: str
    status: str
    received_at: str
    authority: str = "advisory_only"

    def as_dict(self) -> dict[str, str]:
        return {
            "analysis_id": self.analysis_id,
            "digest": self.digest,
            "status": self.status,
            "received_at": self.received_at,
            "authority": self.authority,
        }


class AnalysisInbox:
    """Append-only evidence inbox. It never executes experiment proposals."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.analyses = self.root / "analyses"
        self.ids = self.root / "ids"
        self.receipts = self.root / "receipts"
        for directory in (self.analyses, self.ids, self.receipts):
            directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _canonical_bytes(analysis: ExternalAnalysis) -> bytes:
        return json.dumps(
            analysis.as_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    @staticmethod
    def _write_new(path: Path, data: bytes) -> None:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    def ingest(self, payload: Mapping[str, Any] | ExternalAnalysis) -> IngestReceipt:
        analysis = payload if isinstance(payload, ExternalAnalysis) else parse_external_analysis(payload)
        canonical = self._canonical_bytes(analysis)
        digest = hashlib.sha256(canonical).hexdigest()
        id_key = hashlib.sha256(analysis.analysis_id.encode("utf-8")).hexdigest()
        id_path = self.ids / f"{id_key}.json"
        analysis_path = self.analyses / f"{digest}.json"
        receipt_path = self.receipts / f"{digest}.json"

        with self._lock:
            if id_path.exists():
                marker = json.loads(id_path.read_text(encoding="utf-8"))
                if marker.get("digest") != digest:
                    raise AnalysisConflict(
                        f"analysis_id {analysis.analysis_id!r} already refers to different content"
                    )
                return IngestReceipt(
                    analysis_id=analysis.analysis_id,
                    digest=digest,
                    status="duplicate",
                    received_at=str(marker["received_at"]),
                )

            received_at = datetime.now(timezone.utc).isoformat()
            if not analysis_path.exists():
                self._write_new(analysis_path, canonical + b"\n")
            marker = {
                "analysis_id": analysis.analysis_id,
                "digest": digest,
                "received_at": received_at,
            }
            self._write_new(
                id_path,
                (json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            )
            receipt = IngestReceipt(
                analysis_id=analysis.analysis_id,
                digest=digest,
                status="accepted",
                received_at=received_at,
            )
            self._write_new(
                receipt_path,
                (json.dumps(receipt.as_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode(),
            )
            return receipt

    def load(self, digest: str) -> ExternalAnalysis:
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("digest must be a lowercase SHA-256 hex digest")
        path = self.analyses / f"{digest}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return parse_external_analysis(payload)

    def list_receipts(self) -> list[IngestReceipt]:
        receipts: list[IngestReceipt] = []
        for path in self.receipts.glob("*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            receipts.append(
                IngestReceipt(
                    analysis_id=str(value["analysis_id"]),
                    digest=str(value["digest"]),
                    status=str(value["status"]),
                    received_at=str(value["received_at"]),
                    authority=str(value.get("authority", "advisory_only")),
                )
            )
        return sorted(receipts, key=lambda item: (item.received_at, item.digest))
