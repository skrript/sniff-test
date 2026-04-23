"""
RewardEngine — dense step-level rewards throughout the investigation trajectory.

Rewards good investigative practice; penalises redundancy, repeated queries,
and premature verdicts. Terminal grading remains in TaskGrader.
"""

from typing import List, Set

try:
    from ..models import InvestigateAction
    from .world_state import ClaimScenario
except ImportError:
    from models import InvestigateAction
    from server.world_state import ClaimScenario


REASONING_SIGNALS = [
    "because",
    "however",
    "therefore",
    "despite",
    "although",
    "confirms",
    "contradicts",
    "reveals",
    "suggests",
    "indicates",
    "evidence shows",
    "according to",
    "this means",
    "which means",
    "shows that",
    "proves",
    "demonstrates",
    "inconsistent",
    "unreliable",
]


def justification_quality(justification: str, scenario: ClaimScenario) -> float:
    """Reward justifications that cite evidence and connect it with reasoning."""
    justification_lower = (justification or "").lower()
    has_reasoning = any(signal in justification_lower for signal in REASONING_SIGNALS)
    source_count = sum(
        1
        for src in scenario.sources
        if src.source_id.lower() in justification_lower
        or src.domain.lower() in justification_lower
    )
    if source_count > 0 and has_reasoning:
        return min(1.0, source_count / 2.0)
    if source_count > 0:
        return 0.1
    return 0.0


def format_reward(action: InvestigateAction) -> float:
    """Reward structurally valid verdict submissions and penalise trivial ones."""
    if action.action_type != "submit_verdict":
        return 0.0
    justification = action.justification or ""
    if len(justification.strip()) < 50:
        return -0.2
    if action.verdict is None or action.confidence is None:
        return -0.3
    return 0.1


class RewardEngine:
    """Computes step rewards for one episode."""

    def __init__(self, scenario: ClaimScenario) -> None:
        self.scenario = scenario
        self._key_ids: Set[str] = set(scenario.key_evidence_source_ids)
        self._opened_sources: Set[str] = set()
        self._actions_taken: List[str] = []
        self._previous_queries: Set[str] = set()

    def compute_step_reward(
        self,
        action: InvestigateAction,
        tool_result: str,
        step_count: int,
    ) -> float:
        """Return the shaped reward for a single step (excluding terminal reward)."""
        reward = 0.0
        atype = action.action_type

        if atype == "search":
            query = (action.query or "").strip().lower()
            if query and query in self._previous_queries:
                reward = -0.15
            else:
                if query:
                    self._previous_queries.add(query)
                reward = 0.05

        elif atype == "open_source":
            sid = action.source_id or ""
            if sid in self._opened_sources:
                reward = -0.1  # redundant action penalty
            else:
                self._opened_sources.add(sid)
                reward = 0.3 if sid in self._key_ids else 0.05

        elif atype == "cross_reference":
            ids = set(action.source_ids or [])
            reward = 0.2 if (ids & self._key_ids) else 0.05

        elif atype == "trace_origin":
            reward = 0.15

        elif atype == "check_metadata":
            sid = action.source_id or ""
            reward = 0.15 if sid in self._key_ids else 0.05

        elif atype == "submit_verdict":
            reward = -0.3 if step_count < 3 else 0.0

        # Penalise spamming the same action type repeatedly
        if atype != "submit_verdict" and self._actions_taken[-3:].count(atype) >= 2:
            reward -= 0.05

        self._actions_taken.append(atype)
        return round(reward, 4)
