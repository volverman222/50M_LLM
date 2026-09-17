from __future__ import annotations

from typing import Any, Protocol

from ..executors.base import Executor


class HostAgent(Protocol):
    def capabilities(self) -> dict[str, Any]: ...

    def executor(self, name: str) -> Executor: ...
