from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .evidence import EvidenceStore


class Ledger:
    def __init__(self, db_path: str | Path, evidence: EvidenceStore):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence = evidence
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    campaign_id TEXT,
                    profile TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    executor TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    result_json TEXT
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    PRIMARY KEY (run_id, name)
                );
"""
            )
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    PRIMARY KEY (run_id, path)
                );
                """
            )

    @staticmethod
    def _status(loaded: dict[str, Any]) -> str:
        if loaded["result"] is not None:
            return loaded["result"]["status"]
        for event in reversed(loaded["events"]):
            if event.get("to_state"):
                return event["to_state"]
        return "queued"

    def project_run(self, run_id: str) -> None:
        loaded = self.evidence.load_run(run_id)
        manifest = loaded["manifest"]
        result = loaded["result"]
        status = self._status(loaded)
        priority = manifest.get("priority", 0)
        for event in loaded["events"]:
            if event["kind"] == "priority":
                priority = event["detail"]["priority"]
        manifest_json = json.dumps(manifest, sort_keys=True)
        result_json = json.dumps(result, sort_keys=True) if result is not None else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(run_id,campaign_id,profile,status,priority,executor,created_at,manifest_json,result_json)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                    campaign_id=excluded.campaign_id,
                    profile=excluded.profile,
                    status=excluded.status,
                    priority=excluded.priority,
                    executor=excluded.executor,
                    created_at=excluded.created_at,
                    manifest_json=excluded.manifest_json,
                    result_json=excluded.result_json
                """,
                (
                    run_id, manifest.get("campaign_id"), manifest["profile"], status,
                    priority, manifest.get("executor", "native"),
                    manifest["created_at"], manifest_json, result_json,
                ),
            )
            conn.execute("DELETE FROM metrics WHERE run_id=?", (run_id,))
            if result is not None:
                for name, value in sorted(result.get("metrics", {}).items()):
                    conn.execute(
                        "INSERT INTO metrics(run_id,name,value) VALUES(?,?,?)",
                        (run_id, name, float(value)),
                    )
            conn.execute("DELETE FROM artifacts WHERE run_id=?", (run_id,))
            for artifact in loaded["artifacts"]:
                conn.execute(
                    "INSERT INTO artifacts(run_id,path,sha256,size_bytes) VALUES(?,?,?,?)",
                    (run_id, artifact["path"], artifact["sha256"], artifact["size_bytes"]),
                )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row is not None else None

    def query_runs(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = filters or {}
        allowed = {"run_id", "campaign_id", "profile", "status", "executor"}
        unknown = set(filters) - allowed
        if unknown:
            raise ValueError(f"unsupported run filters: {sorted(unknown)}")
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in filters.items():
            clauses.append(f"{key}=?")
            params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = "SELECT * FROM runs" + where + " ORDER BY priority DESC, created_at ASC, run_id ASC"
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def rebuild(self, evidence_root: str | Path | None = None) -> None:
        root = Path(evidence_root) if evidence_root is not None else self.evidence.root
        with self._connect() as conn:
            conn.execute("DELETE FROM metrics")
            conn.execute("DELETE FROM artifacts")
            conn.execute("DELETE FROM runs")
        for child in sorted(root.iterdir(), key=lambda path: path.name):
            if child.is_dir() and (child / "manifest.json").exists():
                self.project_run(child.name)
