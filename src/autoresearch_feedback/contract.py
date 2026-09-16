from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

SCHEMA = "devpost.autoresearch.external_analysis.v1"
SOURCE = "wandb.aria"
AUTHORITY = "advisory_only"


class ContractError(ValueError):
    """Raised when an external analysis does not satisfy the ingestion contract."""


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    if len(value) > 2048:
        raise ContractError(f"{field} is too long")
    return value


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ContractError(f"{field} must be a list")
    return tuple(_string(item, f"{field}[]") for item in value)


@dataclass(frozen=True)
class Hypothesis:
    claim: str
    evidence: tuple[str, ...]
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ExperimentProposal:
    objective: str
    independent_variables: Mapping[str, Any]
    controlled_variables: Mapping[str, Any]
    expected_observation: str
    falsification_condition: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "independent_variables": dict(self.independent_variables),
            "controlled_variables": dict(self.controlled_variables),
            "expected_observation": self.expected_observation,
            "falsification_condition": self.falsification_condition,
        }


@dataclass(frozen=True)
class ExternalAnalysis:
    entity: str
    project: str
    run_id: str
    run_name: str
    analysis_id: str
    analysis_version: int
    baseline_refs: tuple[str, ...]
    observations: tuple[str, ...]
    anomalies: tuple[str, ...]
    regressions: tuple[str, ...]
    improvements: tuple[str, ...]
    hypotheses: tuple[Hypothesis, ...]
    recommended_experiment: ExperimentProposal
    schema: str = SCHEMA
    source: str = SOURCE
    authority: str = AUTHORITY

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source,
            "authority": self.authority,
            "entity": self.entity,
            "project": self.project,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "analysis_id": self.analysis_id,
            "analysis_version": self.analysis_version,
            "baseline_refs": list(self.baseline_refs),
            "observations": list(self.observations),
            "anomalies": list(self.anomalies),
            "regressions": list(self.regressions),
            "improvements": list(self.improvements),
            "hypotheses": [item.as_dict() for item in self.hypotheses],
            "recommended_experiment": self.recommended_experiment.as_dict(),
        }


def _parse_hypothesis(value: Any) -> Hypothesis:
    if not isinstance(value, dict):
        raise ContractError("hypotheses[] must be an object")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ContractError("hypotheses[].confidence must be a number")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ContractError("hypotheses[].confidence must be between 0 and 1")
    return Hypothesis(
        claim=_string(value.get("claim"), "hypotheses[].claim"),
        evidence=_string_list(value.get("evidence"), "hypotheses[].evidence"),
        confidence=confidence,
    )


def _parse_proposal(value: Any) -> ExperimentProposal:
    if not isinstance(value, dict):
        raise ContractError("recommended_experiment must be an object")
    independent = value.get("independent_variables")
    controlled = value.get("controlled_variables")
    if not isinstance(independent, dict):
        raise ContractError("recommended_experiment.independent_variables must be an object")
    if not isinstance(controlled, dict):
        raise ContractError("recommended_experiment.controlled_variables must be an object")
    return ExperimentProposal(
        objective=_string(value.get("objective"), "recommended_experiment.objective"),
        independent_variables=dict(independent),
        controlled_variables=dict(controlled),
        expected_observation=_string(
            value.get("expected_observation"), "recommended_experiment.expected_observation"
        ),
        falsification_condition=_string(
            value.get("falsification_condition"),
            "recommended_experiment.falsification_condition",
        ),
    )


def parse_external_analysis(payload: Mapping[str, Any]) -> ExternalAnalysis:
    if not isinstance(payload, Mapping):
        raise ContractError("payload must be an object")
    if payload.get("schema") != SCHEMA:
        raise ContractError(f"schema must be {SCHEMA!r}")
    if payload.get("source") != SOURCE:
        raise ContractError(f"source must be {SOURCE!r}")
    version = payload.get("analysis_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ContractError("analysis_version must be a positive integer")
    hypotheses_value = payload.get("hypotheses")
    if not isinstance(hypotheses_value, list):
        raise ContractError("hypotheses must be a list")
    return ExternalAnalysis(
        entity=_string(payload.get("entity"), "entity"),
        project=_string(payload.get("project"), "project"),
        run_id=_string(payload.get("run_id"), "run_id"),
        run_name=_string(payload.get("run_name"), "run_name"),
        analysis_id=_string(payload.get("analysis_id"), "analysis_id"),
        analysis_version=version,
        baseline_refs=_string_list(payload.get("baseline_refs"), "baseline_refs"),
        observations=_string_list(payload.get("observations"), "observations"),
        anomalies=_string_list(payload.get("anomalies"), "anomalies"),
        regressions=_string_list(payload.get("regressions"), "regressions"),
        improvements=_string_list(payload.get("improvements"), "improvements"),
        hypotheses=tuple(_parse_hypothesis(item) for item in hypotheses_value),
        recommended_experiment=_parse_proposal(payload.get("recommended_experiment")),
    )
