# Devpost Hackathon — 50M_LLM
## Local experiment orchestrator (owner-review candidate)

This package is part of the proposed contribution, not merely a W&B bridge. It provides a serial queue, process supervision, per-run evidence, a rebuildable SQLite ledger, resource sampling, browser log/config/result access and a live terminal monitor. It does not change the team's model architecture or publish anything.

**Review before publication:** do not push or update a PR until Thomas has reviewed this exact candidate and explicitly approved publication. The existing telemetry PR is not approval of these additions.

### Install from the repository root

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -e "tools/experiment_orchestrator[dev]"
# Only for actual ML jobs, install the model's dependencies separately:
python -m pip install -e ".[training,dev]"
```

The generic orchestrator does not depend on PyTorch. The optional ML recipe compiler imports the model package only when used. Paths below are examples chosen by the operator, not hardcoded workstation directories.

### Review, queue, monitor

```bash
expctl --root ./experiment_runs prepare-ml ./recipe.json --output ./reviewed_run.json
# Inspect reviewed_run.json before submitting it.
expctl --root ./experiment_runs submit ./reviewed_run.json
expctl --root ./experiment_runs worker
# Separate terminals while the worker is running:
expctl --root ./experiment_runs serve --port 8766
expctl --root ./experiment_runs watch --run-id your-run-id --stream stdout
expctl --root ./experiment_runs logs your-run-id --stream data-protocol
```

The browser binds only to `http://127.0.0.1:8766`. Buttons show Logs, Errors, Telemetry, Resources, Config, Results, Data / splits and Events. Views refresh every two seconds and are bounded to the latest 16 KiB; complete logs remain on disk.

The terminal view refreshes live and uses literal text, not markup from log content. It is a **read-only terminal monitor**, not a finished keyboard-navigation/cancellation TUI. Use the CLI or explicitly enabled browser controls for changes.

### Optional browser controls

`serve --allow-control` explicitly enables local queue/cancel operations. The server prints a session token; paste it into the browser's control-token field. It is not embedded in the page or stored in manifests. Remote Host/Origin requests are rejected. Cancellation is a request until the worker confirms it.

This service launches trusted local commands. It is **not a process sandbox, remote service, or multi-user authorization system**. Do not expose it publicly. Run it under an appropriately restricted OS account. Never put API-key values, passwords or credential command arguments in a RunSpec; the admission layer rejects common credential fields.

### Scientific configuration

See `../../docs/DATA_PROTOCOL.md` and `../../configs/data_protocols/`. Dataset repository/configuration/revision, text column, sequence length, packing, EOS, batch handling, document ordering and role-specific split membership are explicit. Model-initialization and data-order/split seeds are distinct.

`prepare-ml` generates a candidate RunSpec; it does not enqueue or train. Submitted protocols are copied into run artifacts and compared with the sealed manifest before execution. Context/tokenizer mismatches and missing/tampered identities fail closed. GPU execution requires an explicit device and resource reservation; no unreserved `auto` device is accepted by the recipe compiler.

Each job may depend on successful predecessor runs. This configures **execution order**, not an implicit training curriculum: checkpoint-continuation, dataset mixing and cross-stage optimizer-state semantics are not implemented here.

### Evidence and boundaries

Run folders contain manifest/seal, events, stdout/stderr, environment/hardware, telemetry, metrics, artifacts and checkpoints. The database is an index; `rebuild` recreates it from evidence. Unknown running processes after a crash block new dispatch rather than causing a duplicate job. Recovery is explicit; automatic scientific retries are disabled.

The worker is deliberately serial. Resource telemetry is sampled and GPU counters include other processes. External completed-run W&B scalar export stays opt-in/offline-first. Neither running the worker nor opening the dashboard uploads model weights or creates PRs.

The inherited `adapters/` and baseline-specific module remain source-level reference utilities. The supported portable paths are the generic CLI and the explicit `prepare-ml` recipe, not automatic discovery of a workstation's archived baseline.

### Tests

From the repository root, install the model and orchestrator packages in an isolated environment, then run:

```bash
python -m pytest tools/experiment_orchestrator/tests -q
python -m pytest tests -q
```

The suite includes a 256-token synthetic CPU pipeline check on a tiny disposable model. That is a software test, not an additional hackathon experiment or a quality result. Benchmark-scoring validation, multi-worker split equivalence, exact data-cursor resume and a richer interactive TUI remain separate acceptance gates.

### Standalone release and persistent local worker

See `docs/STANDALONE.md` for source-only release export and verification. The same package can be retained outside the model repository without carrying its datasets, checkpoints or Git history. A local release is never silently overwritten.

Use `worker --stay-alive` for a worker that accepts later submissions; plain `worker` retains drain-and-exit behavior. `worker-status` reports the worker separately from its jobs. The admission lock tolerates bounded submission contention without weakening the single-worker lock.

Recipe preparation now resolves the selected repository's protocol module directly. Optional `execution.python_executable` selects a separate model environment, and relative paths resolve beside the recipe JSON. No model package import or PyTorch installation is required for generic orchestration.

For optional ML tests in a separated source package, set `DEVPOST_MODEL_REPO` to the reviewed model repository and use that model's test environment. Without it, model-dependent tests explicitly skip. This does not suppress generic queue, packaging, browser, ledger or lifecycle tests.
