"""
👃 SniffTest — Misinformation Investigation Gym

An OpenEnv-compliant RL environment where an agent investigates claims,
gathers evidence through tool calls, and submits a smell-test verdict.
"""

from .client import SniffTestEnv
from .models import ActionLog, InvestigateAction, SniffTestObservation, SourceSummary

__all__ = [
    "InvestigateAction",
    "SniffTestObservation",
    "SourceSummary",
    "ActionLog",
    "SniffTestEnv",
]
