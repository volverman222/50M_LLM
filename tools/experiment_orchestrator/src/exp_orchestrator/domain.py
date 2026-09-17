from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunState(str, Enum):
    QUEUED = "queued"
    ADMITTED = "admitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ResourceRequest(FrozenModel):
    gpu_count: int = Field(default=0, ge=0)
    min_vram_mb: int = Field(default=0, ge=0)
    process_limit: int = Field(default=1, gt=0)
    min_disk_free_mb: int = Field(default=0, ge=0)


class CommandSpec(FrozenModel):
    argv: list[str]
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("argv")
    @classmethod
    def argv_must_not_be_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("argv must not be empty")
        if any(not isinstance(item, str) or not item for item in value):
            raise ValueError("argv entries must be non-empty strings")
        return value


class RunSpec(FrozenModel):
    run_id: str
    campaign_id: str | None = None
    profile: str = "generic"
    command: CommandSpec
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    priority: int = 0
    depends_on: list[str] = Field(default_factory=list)
    executor: str = "native"
    timeout_seconds: float | None = Field(default=None, gt=0)
    max_retries: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("depends_on")
    @classmethod
    def dependencies_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("depends_on contains duplicate run IDs")
        return value


class RunEvent(FrozenModel):
    kind: str = "transition"
    at: datetime = Field(default_factory=utc_now)
    actor: str
    from_state: RunState | None = None
    to_state: RunState | None = None
    detail: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def transition(
        cls, from_state: RunState | str, to_state: RunState | str, *, actor: str,
        detail: dict[str, Any] | None = None,
    ) -> "RunEvent":
        return cls(
            actor=actor,
            from_state=RunState(from_state),
            to_state=RunState(to_state),
            detail=detail or {},
        )


class RunResult(FrozenModel):
    run_id: str
    status: RunState
    exit_code: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    message: str | None = None
    finished_at: datetime = Field(default_factory=utc_now)
