from exp_orchestrator.domain import CommandSpec, RunSpec
from exp_orchestrator.sweeps import expand_sweep


def base_spec():
    return RunSpec(
        run_id="baseline",
        profile="ml",
        command=CommandSpec(argv=["python", "train.py"]),
        metadata={"model": {"loops": 2, "width": 512}, "lr": 3e-4},
    )


def test_sweep_expansion_is_stable_unique_and_non_mutating():
    base = base_spec()
    original = base.model_dump(mode="json")
    grid = {
        "metadata.model.loops": [1, 2, 3],
        "metadata.lr": [1e-4, 3e-4],
    }
    first = expand_sweep(base, grid, campaign_id="cmp")
    second = expand_sweep(base, grid, campaign_id="cmp")
    assert [item.run_id for item in first] == [item.run_id for item in second]
    assert len({item.run_id for item in first}) == 6
    assert [item.metadata["sweep_changes"] for item in first] == [
        item.metadata["sweep_changes"] for item in second
    ]
    assert all(item.campaign_id == "cmp" for item in first)
    assert base.model_dump(mode="json") == original


def test_sweep_records_changed_keys_and_values():
    children = expand_sweep(base_spec(), {"metadata.model.loops": [1, 4]}, campaign_id="depth")
    assert children[0].metadata["model"]["loops"] == 1
    assert children[1].metadata["model"]["loops"] == 4
    assert children[0].metadata["sweep_changes"] == {"metadata.model.loops": 1}
