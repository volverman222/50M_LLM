# Dynamic Observatory

The Dynamic Observatory is a passive measurement layer for the Devpost Hackathon autoresearch loop.
It captures reduced hidden-state trajectories from repeated model modules and derives dynamical
observables without changing the candidate architecture, optimizer, token budget, evaluation, or
`test_loss` selection rule.

## Boundary

- `src/dynamic_observatory/` is fixed measurement apparatus.
- `rsi_architecture/` remains the agent-editable architecture surface.
- Observation is opt-in with `AUTORESEARCH_DYNAMICS=1`; the default training path is unchanged.
- External/dynamic evidence is advisory. `test_loss` remains the keep/discard metric.
- Research provenance tags are `LIVE`, `REPLAY`, `DERIVED`, and `N/A` only.
- Plausible substitute or simulated research measurements are prohibited.

## Capture path

At the end of an opted-in run, `train.py` takes the first deterministic validation block as the fixed
probe input. If module names are not supplied, the observer performs one inference pass to find
modules invoked at least twice and keeps only the outermost repeated modules. A second passive pass
records each selected module's output, reduced across batch/token axes while preserving hidden width.

For the current LoopedGPT baseline, automatic discovery resolves the recurrent transformer blocks, for
example `trf_blocks.0`, `trf_blocks.1`, rather than their nested attention/norm/FFN descendants.
Training/evaluation mode is restored after every probe.

## Derived observables

For a captured trajectory `h[0..k]`, the observer records the reduced states and derives:

- first differences `Δh` and second differences `Δ²h`;
- phase velocity and phase coherence in the projected dynamical plane;
- radial drift in that plane;
- normalized return/closure error for periods `q = 1..Q`;
- dominant recurrence frequency from the temporal FFT;
- normalized spectral entropy.

The JSON record schema is `devpost.dynamic_observation.v1`. Each record carries run id, token
checkpoint, probe id, trace name, source provenance, states, and metrics. W&B receives scalar metrics
under `dynamics/<trace>/...` plus the JSON records as a `dynamic-observation` artifact.

## Projection comparability

When no reference projection is supplied, each trace is tagged
`LOCAL_PCA_UNCALIBRATED`. That view is useful for within-trajectory structure but must not be treated
as a fixed coordinate system across checkpoints or runs.

For cross-run comparison, build one persisted reference frame from the chosen baseline observation:

```powershell
uv run python -m dynamic_observatory.cli reference baseline-observation.json baseline-frame.json
```

The frame schema is `devpost.dynamic_reference_frame.v1` and stores the baseline center and deterministic
PCA/SVD basis. Reusing both fixes the coordinate system across checkpoints and preserves real displacement
instead of re-centering every candidate independently.

## Enable the final-run probe

```powershell
$env:AUTORESEARCH_DYNAMICS = "1"
$env:AUTORESEARCH_DYNAMICS_PROBE_ID = "validation-head-v1"
uv run train.py
```

Optional explicit module selection:

```powershell
$env:AUTORESEARCH_DYNAMICS_MODULES = "trf_blocks.0,trf_blocks.1,trf_blocks.2"
```

## DYN-3D viewer

Open `tools/dynamic_observatory_viewer/index.html` in a browser. The viewer is deliberately offline:
it does not fetch remote data and accepts research evidence only through local JSON file selection or
drag-and-drop.

Load one or more `devpost.dynamic_observation.v1` files. For cross-run overlays, also load the matching
`devpost.dynamic_reference_frame.v1` file. Traces that match the fixed probe/trace frame are marked
`FIXED_REFERENCE`; unmatched traces fall back to `LOCAL_PCA_UNCALIBRATED` and display a comparison
warning.

The viewer provides orbit/dolly interaction, trajectory overlays, state points, delta-vector display,
axes, selected-trace provenance, closure metrics, phase/coherence metrics, spectral metrics, and token
checkpoint context. It never feeds a visual score back into the keep/discard rule.
