# Screenshot provenance

These images document operator-facing behavior. They are **not model-quality evidence**. Captures are cropped to the relevant application/output region and contain no desktop, taskbar, browser chrome, or development-tool UI.

- `dyn3d-fixed-reference.png` — DYN-3D with a local documentation fixture loaded together with a compatible reference frame. The observation is explicitly tagged `source = N/A`; the screenshot demonstrates the `FIXED_REFERENCE` UI state and provenance panel only.
- `dyn3d-local-pca-warning.png` — the same documentation fixture without a reference frame, demonstrating the visible `LOCAL_PCA_UNCALIBRATED` guard.
- `aria-health-advisory-only.png` — the local receiver's real `/healthz` response while running with an ephemeral documentation-only secret and inbox. It demonstrates that the service exposes `authority = advisory_only`.

The temporary JSON fixture and documentation-only receiver inbox are intentionally not committed. Do not cite these screenshots as experiment results, benchmark evidence, or scientific measurements.
- `orchestrator-dashboard.png` ? browser-qualification capture of the adjacent orchestrator contribution, showing worker state, run list, and evidence views such as Data / splits, Inputs consumed, Validation identity, and Source identity. Its inclusion documents the operator surface; it does not imply that the contribution is merged into `rsi_exp`.
