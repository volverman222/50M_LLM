from llm_mini_lab.models import LoopedGPTModel


def build_model(cfg):
    """Build the current candidate architecture. Agents may replace this implementation."""
    return LoopedGPTModel(cfg)
