# Screenshot provenance

These images document operator-facing behavior. They are **not model-quality evidence**.

- `dyn3d-fixed-reference.png` — DYN-3D with a local documentation fixture loaded together with a compatible reference frame. The observation is explicitly tagged `source = N/A`; the screenshot demonstrates the `FIXED_REFERENCE` UI state and provenance panel only.
- `dyn3d-local-pca-warning.png` — the same documentation fixture without a reference frame, demonstrating the visible `LOCAL_PCA_UNCALIBRATED` guard.
- `aria-health-advisory-only.png` — the local receiver's real `/healthz` response while running with an ephemeral documentation-only secret and inbox. It demonstrates that the service exposes `authority = advisory_only`.

The temporary JSON fixture and documentation-only receiver inbox are intentionally not committed. Do not cite these screenshots as experiment results, benchmark evidence, or scientific measurements.
- `orchestrator-dashboard-owner-review.png` — existing browser-qualification capture from `X:/tfs.experiments/50M_LLM/reviews/standalone-integration-20260916T071922Z/browser/desktop.png`. It shows the **adjacent local owner-review candidate**, including worker state, run list and evidence views such as Data / splits, Inputs consumed, Validation identity and Source identity. Its inclusion here documents the current operator surface; it does not imply that the orchestrator/data-protocol contribution is merged or publication-approved.
