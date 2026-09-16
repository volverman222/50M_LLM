import pytest
from pydantic import ValidationError

from exp_orchestrator.project_identity import (
    DEVPOST_PROJECT_LABEL,
    ProjectIdentity,
    assert_project_document_identity,
)


def test_devpost_identity_is_the_only_default_for_this_project():
    identity = ProjectIdentity()
    assert identity.project_label == "Devpost Hackathon — 50M_LLM"
    assert DEVPOST_PROJECT_LABEL == identity.project_label


def test_torafirma_cannot_be_used_as_project_issuer_or_label():
    with pytest.raises(ValidationError):
        ProjectIdentity(project_label="ToraFirma Systems")
    with pytest.raises(ValidationError):
        ProjectIdentity(issuer="Torafirma Systems")


def test_project_facing_title_must_identify_devpost_hackathon():
    assert_project_document_identity("# Devpost Hackathon — 50M_LLM\n\nAnalysis Report")
    with pytest.raises(ValueError):
        assert_project_document_identity("# TFS 50M LLM\n\nAnalysis Report")
    with pytest.raises(ValueError):
        assert_project_document_identity("# 50M LLM Analysis Report\n\nTorafirma Systems")
