# Devpost Hackathon — 50M_LLM
## Repository integration and standalone operation

Source remains at `tools/experiment_orchestrator/`. A standalone release is a verified, versioned copy of that package, not an independently edited fork. Existing experiment folders are not moved or adopted automatically.

### Export and check parity

```text
expctl --root unused snapshot-package tools/experiment_orchestrator /path/to/new-release
expctl --root unused verify-package /path/to/new-release
expctl --root unused verify-package /path/to/new-release --source tools/experiment_orchestrator
```

Export accepts a new destination only, stages the copy, verifies it and records a file manifest. Runtime files, datasets, checkpoints, virtual environments and Git internals are excluded. Never edit a recorded release to synchronize it; make and review another release. Local hashes are not signed custody or authenticity guarantees.

### Independent Python environments

The core needs Pydantic, psutil and Rich, not PyTorch or an installed model package. `prepare-ml` loads the protocol module from the exact trusted `model_repo` selected in the recipe, not another installed checkout. This imports trusted local Python code and is not a sandbox.

An optional `execution.python_executable` selects the model's Python environment separately. The default is the current interpreter. Set it explicitly when the core environment lacks model dependencies. Interpreter, source identities and commands are recorded in the RunSpec.

Relative model, tokenizer and local-dataset paths resolve against the recipe JSON location. Preparation creates no evidence directory or training process. Submission and execution remain separate operations.

### Worker lifetime

`worker` drains a queue and exits. `worker --stay-alive` explicitly waits idle for later submissions. It does not start at boot. Ctrl+C cancels owned active child processes and ends the worker; unrelated processes are not killed.

Worker lifecycle is distinct from run state in `worker-status`, browser and terminal views. Receipts include PID/start identity and heartbeat time. Stale receipts are not evidence of live processes; process liveness does not prove responsiveness or successful training.
