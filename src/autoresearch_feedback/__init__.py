"""Signed advisory feedback ingestion for the Devpost autoresearch loop."""

from .contract import ExternalAnalysis, ExperimentProposal, parse_external_analysis
from .store import AnalysisInbox, IngestReceipt

__all__ = [
    "AnalysisInbox",
    "ExternalAnalysis",
    "ExperimentProposal",
    "IngestReceipt",
    "parse_external_analysis",
]
