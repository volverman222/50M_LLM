from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .domain import RunEvent, RunResult, RunSpec, RunState


TERMINAL_STATES = {
    RunState.COMPLETED,
    RunState.FAILED,
    RunState.CANCELLED,
    RunState.BLOCKED,
}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_run_dir(root: str | Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id):
        raise ValueError("unsafe run_id")
    stem = run_id.split(".", 1)[0].upper()
    if run_id.endswith(".") or stem in {"CON","PRN","AUX","NUL", *[f"COM{i}" for i in range(1,10)], *[f"LPT{i}" for i in range(1,10)]}:
        raise ValueError("reserved/unsafe run_id")
    path = Path(root) / run_id
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("run path escapes evidence root")
    return path



class EvidenceStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return safe_run_dir(self.root, run_id)

    def create_run(self, spec: RunSpec) -> Path:
        run_dir = self.run_dir(spec.run_id)
        run_dir.mkdir(parents=False, exist_ok=False)
        for dirname in ("telemetry", "metrics", "artifacts", "checkpoints"):
            (run_dir / dirname).mkdir()
        _atomic_json(run_dir / "manifest.json", spec.model_dump(mode="json"))
        digest = hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()
        (run_dir / "manifest.sha256").write_text(digest, encoding="ascii")
        _atomic_json(run_dir / "source.json", {})
        _atomic_json(run_dir / "environment.json", {})
        _atomic_json(run_dir / "hardware.json", {})
        for filename in ("events.jsonl", "stdout.log", "stderr.log"):
            (run_dir / filename).touch(exist_ok=False)
        (run_dir / "artifacts" / "index.jsonl").touch(exist_ok=False)
        return run_dir

    def append_event(self, run_id: str, event: RunEvent) -> None:
        _append_jsonl(
            self.run_dir(run_id) / "events.jsonl",
            event.model_dump(mode="json"),
        )

    def append_log(self, run_id: str, stream: str, text: str) -> None:
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be stdout or stderr")
        path = self.run_dir(run_id) / f"{stream}.log"
        with path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()

    def register_artifact(
        self, run_id: str, source: str | Path, relative_path: str
    ) -> dict[str, Any]:
        source_path = Path(source)
        destination = self.run_dir(run_id) / "artifacts" / relative_path
        if not destination.resolve().is_relative_to((self.run_dir(run_id) / "artifacts").resolve()):
            raise ValueError("artifact path escapes run")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copy2(source_path, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        record = {
            "path": relative_path.replace("\\", "/"),
            "sha256": digest,
            "size_bytes": destination.stat().st_size,
        }
        _append_jsonl(self.run_dir(run_id) / "artifacts" / "index.jsonl", record)
        return record

    def seal_result(self, run_id: str, result: RunResult) -> Path:
        if result.run_id != run_id:
            raise ValueError("result run_id does not match evidence directory")
        if result.status not in TERMINAL_STATES:
            raise ValueError("result status must be terminal")
        path = self.run_dir(run_id) / "result.json"
        if path.exists():
            raise FileExistsError(path)
        _atomic_json(path, result.model_dump(mode="json"))
        return path

    def load_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.run_dir(run_id)
        seal = run_dir / "manifest.sha256"
        if not seal.is_file():
            raise ValueError("manifest seal is missing")
        if seal.read_text(encoding="ascii") != hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest():
            raise ValueError("manifest digest mismatch")
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if "data_protocol" in manifest.get("metadata", {}):
            protocol_file = run_dir / "artifacts" / "data_protocol.json"
            if not protocol_file.is_file() or json.loads(protocol_file.read_text(encoding="utf-8")) != manifest["metadata"]["data_protocol"]:
                raise ValueError("data protocol evidence mismatch")
        events = [
            json.loads(line)
            for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result_path = run_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        artifact_path = run_dir / "artifacts" / "index.jsonl"
        artifacts = [
            json.loads(line)
            for line in artifact_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] if artifact_path.exists() else []
        return {"manifest": manifest, "events": events, "result": result, "artifacts": artifacts}
