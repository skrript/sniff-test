"""
RewardEngine — dense step-level rewards throughout the investigation trajectory.

Rewards good investigative practice; penalises redundancy and lazy loops.
Terminal reward is handled by TaskGrader — not here.
"""

from typing import List, Set

try:
    from ..models import InvestigateAction
    from .world_state import ClaimScenario
except ImportError:
    from models import InvestigateAction
    from server.world_state import ClaimScenario


class RewardEngine:
    """Computes step rewards for one episode."""

    def __init__(self, scenario: ClaimScenario) -> None:
        self.scenario = scenario
        self._key_ids: Set[str] = set(scenario.key_evidence_source_ids)
        self._opened_sources: Set[str] = set()
        self._actions_taken: List[str] = []

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
            reward = 0.0  # terminal reward computed by grader

        # Penalise spamming the same action type repeatedly
        if atype != "submit_verdict" and self._actions_taken[-3:].count(atype) >= 2:
            reward -= 0.05

        self._actions_taken.append(atype)
        return round(reward, 4)
