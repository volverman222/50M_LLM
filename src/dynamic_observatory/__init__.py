"""Passive dynamical-state observation for Devpost autoresearch experiments."""

from .artifact import DynamicObservation, load_observation, write_observation
from .capture import TraceRecorder
from .metrics import DynamicsReport, analyze_trajectory, fit_projection_basis
from .probe import capture_observations, discover_repeated_modules
from .reference import (
    DynamicReferenceFrame,
    build_reference_frame,
    load_reference_frame,
    project_states,
    write_reference_frame,
)

__all__ = [
    "DynamicObservation",
    "DynamicReferenceFrame",
    "DynamicsReport",
    "TraceRecorder",
    "analyze_trajectory",
    "build_reference_frame",
    "capture_observations",
    "discover_repeated_modules",
    "fit_projection_basis",
    "load_observation",
    "load_reference_frame",
    "project_states",
    "write_observation",
    "write_reference_frame",
]
