import json

import pytest
from pydantic import ValidationError

from exp_orchestrator.domain import (
    CommandSpec,
    ResourceRequest,
    RunEvent,
    RunResult,
    RunSpec,
    RunState,
)


def make_spec() -> RunSpec:
    return RunSpec(
        run_id="run-001",
        profile="generic",
        command=CommandSpec(argv=["python", "job.py"]),
        resources=ResourceRequest(gpu_count=1, min_vram_mb=8000),
    )


def test_runspec_round_trips_json_and_is_frozen():
    spec = make_spec()
    assert RunSpec.model_validate_json(spec.model_dump_json()) == spec
    with pytest.raises(ValidationError):
        spec.profile = "changed"


def test_resource_request_rejects_negative_limits():
    with pytest.raises(ValidationError):
        ResourceRequest(gpu_count=-1)
    with pytest.raises(ValidationError):
        ResourceRequest(min_vram_mb=-1)
    with pytest.raises(ValidationError):
        ResourceRequest(process_limit=0)


def test_command_rejects_empty_argv_and_duplicate_dependencies():
    with pytest.raises(ValidationError):
        CommandSpec(argv=[])
    with pytest.raises(ValidationError):
        RunSpec(
            run_id="dup",
            profile="generic",
            command=CommandSpec(argv=["python", "job.py"]),
            depends_on=["a", "a"],
        )


def test_run_state_values_are_declared_and_events_results_validate():
    assert {state.value for state in RunState} == {
        "queued", "admitted", "running", "completed", "failed", "cancelled", "blocked"
    }
    event = RunEvent.transition("queued", "admitted", actor="scheduler")
    assert event.from_state == RunState.QUEUED
    assert event.to_state == RunState.ADMITTED
    result = RunResult(run_id="run-001", status=RunState.COMPLETED, exit_code=0)
    payload = json.loads(result.model_dump_json())
    assert payload["status"] == "completed"
