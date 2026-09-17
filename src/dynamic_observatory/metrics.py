from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class DynamicsReport:
    projected: torch.Tensor
    deltas: torch.Tensor
    second_deltas: torch.Tensor
    phase_velocity: float
    phase_coherence: float
    radius_drift: float
    closure_errors: dict[int, float]
    dominant_frequency: float
    spectral_entropy: float

    def metrics_dict(self) -> dict[str, object]:
        return {
            "phase_velocity": self.phase_velocity,
            "phase_coherence": self.phase_coherence,
            "radius_drift": self.radius_drift,
            "closure_errors": {str(k): v for k, v in self.closure_errors.items()},
            "dominant_frequency": self.dominant_frequency,
            "spectral_entropy": self.spectral_entropy,
        }


def _states_2d(states: torch.Tensor) -> torch.Tensor:
    states = torch.as_tensor(states, dtype=torch.float32, device="cpu")
    if states.ndim != 2:
        raise ValueError("states must have shape [steps, dimensions]")
    if states.shape[0] < 2 or states.shape[1] < 1:
        raise ValueError("states require at least 2 steps and 1 dimension")
    if not torch.isfinite(states).all():
        raise ValueError("states must be finite")
    return states


def fit_projection_basis(states: torch.Tensor, components: int = 3) -> torch.Tensor:
    states = _states_2d(states)
    centered = states - states.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    count = min(max(1, components), vh.shape[0])
    basis = vh[:count].T.contiguous()
    for col in range(basis.shape[1]):
        vec = basis[:, col]
        pivot = torch.argmax(vec.abs())
        if vec[pivot] < 0:
            basis[:, col] = -vec
    return basis


def _phase_stats(projected: torch.Tensor) -> tuple[float, float, float]:
    if projected.shape[1] < 2:
        return 0.0, 0.0, 0.0
    xy = projected[:, :2]
    angles = torch.atan2(xy[:, 1], xy[:, 0])
    raw = angles[1:] - angles[:-1]
    wrapped = torch.atan2(torch.sin(raw), torch.cos(raw))
    c = torch.cos(wrapped).mean()
    s = torch.sin(wrapped).mean()
    coherence = torch.sqrt(c * c + s * s)
    velocity = torch.atan2(s, c)
    radius = torch.linalg.vector_norm(xy, dim=1)
    drift = (radius[1:] - radius[:-1]).mean()
    return float(velocity), float(coherence), float(drift)


def _spectral_stats(states: torch.Tensor) -> tuple[float, float]:
    centered = states - states.mean(dim=0, keepdim=True)
    coeff = torch.fft.rfft(centered, dim=0)
    power = (coeff.real.square() + coeff.imag.square()).sum(dim=1)
    if power.numel() <= 1:
        return 0.0, 0.0
    power = power.clone()
    power[0] = 0.0
    total = power.sum()
    if total <= 0:
        return 0.0, 0.0
    idx = int(torch.argmax(power))
    freq = float(torch.fft.rfftfreq(states.shape[0], d=1.0)[idx])
    probs = power[1:] / total
    positive = probs[probs > 0]
    entropy = float(-(positive * torch.log(positive)).sum())
    denom = math.log(max(1, probs.numel()))
    return freq, (entropy / denom if denom > 0 else 0.0)


def analyze_trajectory(
    states: torch.Tensor,
    *,
    projection_basis: torch.Tensor | None = None,
    max_period: int = 8,
) -> DynamicsReport:
    states = _states_2d(states)
    centered = states - states.mean(dim=0, keepdim=True)
    basis = fit_projection_basis(states) if projection_basis is None else torch.as_tensor(
        projection_basis, dtype=torch.float32, device="cpu"
    )
    if basis.ndim != 2 or basis.shape[0] != states.shape[1]:
        raise ValueError("projection_basis must have shape [dimensions, components]")
    projected = centered @ basis
    deltas = states[1:] - states[:-1]
    second_deltas = deltas[1:] - deltas[:-1]
    phase_velocity, phase_coherence, radius_drift = _phase_stats(projected)
    scale = float(torch.linalg.vector_norm(centered, dim=1).mean()) + 1e-12
    closure_errors: dict[int, float] = {}
    for period in range(1, min(max_period, states.shape[0] - 1) + 1):
        error = torch.linalg.vector_norm(states[period:] - states[:-period], dim=1).mean()
        closure_errors[period] = float(error) / scale
    dominant_frequency, spectral_entropy = _spectral_stats(states)
    return DynamicsReport(
        projected=projected,
        deltas=deltas,
        second_deltas=second_deltas,
        phase_velocity=phase_velocity,
        phase_coherence=phase_coherence,
        radius_drift=radius_drift,
        closure_errors=closure_errors,
        dominant_frequency=dominant_frequency,
        spectral_entropy=spectral_entropy,
    )
