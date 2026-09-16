# Devpost Hackathon — 50M_LLM

## Opt-in training telemetry

This contribution adds measurements to the existing training path; it does not
replace the team's model, optimizer, data loader or experiment methodology.
Telemetry is disabled unless `--telemetry-jsonl` is supplied. Existing defaults
are preserved. The new architecture CLI overrides are optional; in particular,
`--n-kv-heads` forwards the existing upstream GQA setting without enabling it by default.

```bash
python scripts/train_pretrain_smollm_1b.py \
  --target-tokens 100000000 --seed 123 \
  --telemetry-jsonl output/run-a/training.jsonl --telemetry-every 50 \
  --checkpoint-dir output/run-a/checkpoints --hf-upload-every 0
```

The example retains the trainer's other defaults; it is not a new agreed data or
model recipe. Supply the team's exact tokenizer, model, batch and evaluation
arguments when reproducing an experiment. The Hugging Face upload control is
independent from telemetry and W&B; no existing trainer default is changed here.

### Local record contract

A new JSONL file starts with a `config` record (`schema_version: 1`, dataset label,
resolved model configuration, CLI arguments, parameter count and device).
The file must not exist. For a resumed checkpoint, use a new telemetry filename;
previous evidence is never truncated. Checkpoint files are managed by the existing
trainer, independently from this telemetry file.

| Record | Scope and meaning |
|---|---|
| `update` | Update 1 and every Nth optimizer update: last-microbatch loss, gradients, actual parameter displacement, timings and allocator peaks |
| `block` | Each application of a shared block on the last microbatch of a sampled update, indexed by zero-based pass and block |
| `loop` | Entry/exit RMS, displacement RMS and cosine for each complete shared-block pass |
| `validation` | The trainer's existing periodic held-out loss and processed-token count |

For SwiGLU, block records additionally include gate/value/product magnitudes and
the fraction with sigmoid(gate preactivation) outside [0.01, 0.99]. This fraction
is not a general diagnosis of saturation or dead features. GELU remains supported
without SwiGLU-only fields. Nonfinite scalar values are stored as JSON `null` and
listed in `nonfinite_fields`, not encoded as nonstandard JSON NaN/Infinity.

Gradient measurements are taken after unscaling and before clipping. Parameter
RMS and displacement are measured after the optimizer step. `lr` is recorded
after `scheduler.step()` and therefore describes the next scheduled rate.
`loss` is the final microbatch, not the average across accumulated microbatches.
Allocator memory values use decimal GB; they are not whole-device VRAM usage.

Sampling includes device synchronization and parameter snapshots on instrumented
updates. Do not treat their `step_ms` as an uninstrumented performance benchmark.
An entry/exit cosine is not a cosine between successive update vectors. Sparse
samples from changing minibatches are not a fixed-input dynamical trajectory.
These are descriptive/defensive measurements, not assertions of a model defect
or evidence that a particular harmonic/oscillatory mechanism is present.

## W&B without replacing the local ledger

With both `--wandb` and `--telemetry-jsonl`, the trainer also sends sampled scalar
metrics into its existing W&B run. Raw CLI paths are kept in local JSONL rather
than forwarded as W&B configuration. W&B remains an optional training dependency.

A separate exporter lets an external orchestrator mirror a *completed snapshot*
without importing the orchestrator into this repository:

```bash
python scripts/export_telemetry_wandb.py output/run-a/training.jsonl \
  --project YOUR_PROJECT --output-dir output/run-a-export --dry-run

python scripts/export_telemetry_wandb.py output/run-a/training.jsonl \
  --project YOUR_PROJECT --output-dir output/run-a-export --mode offline
```

Preview validates and prints only the approved configuration/metric payloads; it
neither imports W&B nor writes files. Export defaults to **offline**, records a
source hash and terminal receipt, and groups metrics by optimizer update. It
rejects incomplete lines, conflicting values, backwards update indices, unknown
schema versions and multiple config records. The v1 exporter also accepts older
single-run logs without a `schema_version` field. Snapshot size is limited to
128 MiB; split larger histories deliberately rather than silently truncate them.

Online export requires both `--mode online` and `--allow-online`. An optional
`--api-key-env WANDB_HACKATHON_API` names an existing credential variable; never
put the credential itself in the command. The adapter does not write login
credentials. Project/entity/name are supplied explicitly or resolved by the SDK;
no team account is embedded in the source.

The exporter allowlists scalar metric and model/training fields; it does not call
`watch`, upload checkpoints, register model artifacts, or copy raw logs. SDK code,
Git, console, requirements and automatic system-metadata capture are disabled for
this standalone export. Its SDK privacy settings are validated by W&B; an
incompatible SDK fails rather than silently dropping those controls. Tested with
W&B 0.30.0. Nonfinite measurements are omitted from charts with an explicit
nonfinite count, while the original local evidence remains unchanged.

A receipt path is exclusive: a repeated attempt using that path is refused even
if an earlier export failed. Online runs use a source-hash-derived ID with
`resume="never"`. An operator must inspect failed/partial exports before retrying;
the tool does not silently resume, overwrite or deduplicate conflicting histories.

## Boundaries and validation

The experiment queue, resource monitor, immutable run manifests, SQLite ledger,
reports and lab notebook remain in the external orchestrator. That orchestrator
may launch this exporter as an ordinary subprocess after a successful run.
This repository receives no run data, credentials, local filesystem bindings or
weights from the author's workstation.

```bash
PYTHONPATH=src python -m pytest tests -q -rs
```

The tests include CPU/CUDA MHA and GQA observation, dropout/RNG/gradient
noninterference, tiny synthetic end-to-end training with telemetry on versus off,
no-overwrite behavior, JSON validity, scalar-export privacy and failed-sink handling.
They are small synthetic checks, not new 100M-token experiments or benchmark claims.
The existing forced-Flash test skips only when the installed PyTorch was not built
with that backend; it still executes where Flash is compiled in. A skip is not
FlashAttention validation. On the native Windows test build, the other CUDA tests
exercise the available SDPA backend.

SDK contracts used: the W&B `Run.log`/`define_metric`, `init(mode=...)` and `Settings`
APIs. Local source hashes and full-precision records, rather than rounded W&B
charts, remain the reference for reproducibility comparisons.
