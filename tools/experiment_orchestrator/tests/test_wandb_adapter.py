import importlib
import importlib.util
import json
import pytest


def module():
    assert importlib.util.find_spec("exp_orchestrator.wandb_adapter"), (
        "Bridge not implemented"
    )
    return importlib.import_module("exp_orchestrator.wandb_adapter")


def fixture(tmp_path):
    run = tmp_path / "run"
    repo = tmp_path / "model"
    (run / "telemetry").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    (repo / "src").mkdir()
    (run / "result.json").write_text(
        json.dumps({"status": "completed", "exit_code": 0})
    )
    (run / "telemetry/training.jsonl").write_text(
        '{"kind":"update","update":1,"loss":3}\n'
    )
    (repo / "scripts/export_telemetry_wandb.py").write_text("# fixture")
    return run, repo, tmp_path / "export"


def test_builds_native_command_without_uploading_or_reading_secrets(tmp_path):
    m = module()
    run, repo, out = fixture(tmp_path)
    cmd = m.build_wandb_export_command(
        run, repo, out, project="team-test", python="python"
    )
    assert cmd.argv[:2] == ["python", "-u"]
    assert cmd.argv[cmd.argv.index("--mode") + 1] == "offline"
    assert cmd.env["WANDB_MODE"] == "offline"
    assert not out.exists()


def test_rejects_active_run(tmp_path):
    m = module()
    run, repo, out = fixture(tmp_path)
    (run / "result.json").unlink()
    with pytest.raises(ValueError):
        m.build_wandb_export_command(run, repo, out, project="test")


def test_online_requires_authorization_and_passes_variable_name_only(tmp_path):
    m = module()
    run, repo, out = fixture(tmp_path)
    with pytest.raises(ValueError):
        m.build_wandb_export_command(run, repo, out, project="test", mode="online")
    cmd = m.build_wandb_export_command(
        run,
        repo,
        out,
        project="test",
        mode="online",
        allow_online=True,
        api_key_env="WANDB_HACKATHON_API",
    )
    assert "--allow-online" in cmd.argv
    assert cmd.argv[-2:] == ["--api-key-env", "WANDB_HACKATHON_API"]
    assert "WANDB_HACKATHON_API" not in cmd.env


def test_rejects_output_inside_sealed_run(tmp_path):
    m = module()
    run, repo, out = fixture(tmp_path)
    with pytest.raises(ValueError):
        m.build_wandb_export_command(run, repo, run / "export", project="test")
