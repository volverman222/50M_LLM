# W&B / ARIA feedback boundary

This module is the inbound evidence boundary between W&B/ARIA and the Devpost autoresearch loop.
It deliberately does **not** execute experiments. External analysis is advisory evidence; only the
orchestrator may turn an accepted proposal into an experiment specification and schedule it.

## Invariants

- Accept only `devpost.autoresearch.external_analysis.v1` from `source = "wandb.aria"`.
- Verify `X-Wandb-Signature` over the exact request bytes with HMAC-SHA256.
- Optionally require a bearer token and an exact W&B entity/project allowlist.
- Store accepted analyses and receipts append-only under `.autoresearch/aria-inbox/` by SHA-256.
- Replaying identical content is idempotent; reusing an `analysis_id` for different content is a conflict.
- Never derive filesystem paths from the remote `analysis_id`.
- Never execute commands, edit research code, or change experiment state from this endpoint.
- Bind to loopback by default. Non-loopback binding requires an explicit command-line override.

## Receiver

PowerShell example:

```powershell
$env:WANDB_WEBHOOK_SECRET = "<team webhook secret>"
$env:WANDB_WEBHOOK_BEARER = "<optional bearer token>"
$env:AUTORESEARCH_WANDB_ENTITY = "<entity>"
$env:AUTORESEARCH_WANDB_PROJECT = "<project>"
uv run python -m autoresearch_feedback.server
```

Endpoints:

- `GET /healthz`
- `POST /v1/wandb/aria`

The receiver listens on `127.0.0.1:8787` by default. Do not expose it directly to the internet.
If W&B Cloud must reach the receiver, terminate public HTTPS/authentication in a controlled relay or
reverse proxy and forward only the signed request to this loopback service.

## Analysis contract

```json
{
  "schema": "devpost.autoresearch.external_analysis.v1",
  "source": "wandb.aria",
  "entity": "<wandb entity>",
  "project": "<wandb project>",
  "run_id": "<wandb run id>",
  "run_name": "<wandb run name>",
  "analysis_id": "<stable unique analysis id>",
  "analysis_version": 1,
  "baseline_refs": ["<comparison run>"],
  "observations": ["<observation>"],
  "anomalies": [],
  "regressions": [],
  "improvements": [],
  "hypotheses": [
    {
      "claim": "<testable claim>",
      "evidence": ["<metric or run evidence>"],
      "confidence": 0.0
    }
  ],
  "recommended_experiment": {
    "objective": "<one objective>",
    "independent_variables": {},
    "controlled_variables": {},
    "expected_observation": "<what should change if the hypothesis is right>",
    "falsification_condition": "<observation that rejects the hypothesis>"
  }
}
```

`authority` is added locally as `advisory_only`; the sender cannot elevate it.

## ARIA finished-run instruction

Use the finished-run automation already configured in W&B, but make the output contract explicit:

```text
Analyze the finished run in its current W&B project. Summarize the key metrics and compare the run
with recent relevant runs under the same experimental methodology. Identify anomalies, regressions,
and improvements. Form only testable hypotheses supported by run evidence. Recommend one smallest
next experiment that preserves the fixed dataset, tokenizer, evaluation method, 1,000,000-token
budget, and 50M-parameter ceiling.

Return one machine-readable JSON object conforming exactly to
`devpost.autoresearch.external_analysis.v1`. Include the W&B entity, project, run id, run name, a
stable unique analysis id, baseline references, observations, anomalies, regressions, improvements,
hypotheses with confidence in [0,1], and one recommended experiment containing objective,
independent variables, controlled variables, expected observation, and falsification condition.
Do not launch runs, edit code, mutate W&B state, or treat the recommendation as execution authority.
```

## Transport status

W&B supports signed webhook automations for W&B events, but this integration does not assume that an
ARIA chat/analysis completion is itself a documented webhook event. The receiving contract above is
therefore transport-independent: wire the ARIA output to it only through a W&B-supported or otherwise
explicitly authenticated callback/relay that has been verified in the account. Until that final hop is
verified, keep ARIA analysis enabled as an observer and keep experiment execution under the existing
orchestrator.

## Auto Researcher consumption

`AnalysisInbox.list_receipts()` returns immutable accepted receipts in receive order and
`AnalysisInbox.load(digest)` returns the validated typed analysis. A consumer may use those records to
update its evidence state and create a candidate experiment specification. Acceptance of a proposal is
not permission to run it; normal orchestrator validation, invariants, and scheduling still apply.
