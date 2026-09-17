from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

DEVPOST_PROJECT_LABEL = "Devpost Hackathon — 50M_LLM"
DEVPOST_ISSUER = "Devpost Hackathon"
_PROHIBITED = ("torafirma", "tora firma")


class ProjectIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    project_label: Literal["Devpost Hackathon — 50M_LLM"] = DEVPOST_PROJECT_LABEL
    issuer: str = DEVPOST_ISSUER

    @field_validator("issuer")
    @classmethod
    def issuer_must_not_use_torafirma_brand(cls, value: str) -> str:
        lowered = value.casefold()
        if any(term in lowered for term in _PROHIBITED):
            raise ValueError("Devpost project documents must not use Torafirma branding")
        return value


def assert_project_document_identity(text: str) -> None:
    lowered = text.casefold()
    if any(term in lowered for term in _PROHIBITED):
        raise ValueError("Torafirma branding is prohibited for this project")
    first_heading = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if DEVPOST_PROJECT_LABEL not in first_heading:
        raise ValueError(f"First document heading must identify {DEVPOST_PROJECT_LABEL}")
