from exp_orchestrator.agents.local import LocalHostAgent
from exp_orchestrator.executors.native import NativeExecutor


def test_local_agent_reports_stable_capabilities_and_executors():
    agent = LocalHostAgent(executors={"native": NativeExecutor()})
    first = agent.capabilities()
    second = agent.capabilities()
    assert first["host_id"] == second["host_id"]
    assert first["cpu_count"] >= 1
    assert first["memory_total_mb"] > 0
    assert isinstance(first["gpus"], list)
    assert "native" in first["executors"]
    assert agent.executor("native") is not None


def test_local_agent_rejects_unknown_executor():
    agent = LocalHostAgent(executors={"native": NativeExecutor()})
    try:
        agent.executor("missing")
    except KeyError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing executor should raise KeyError")
