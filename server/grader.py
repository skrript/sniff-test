"""
TaskGrader — deterministic, LLM-free episode scoring.

Called once per episode after submit_verdict (or on timeout).
Returns a dict with per-dimension scores and a weighted final_score in [0, 1].
"""

from typing import List

try:
    from ..models import ActionLog
    from .world_state import ClaimScenario
except ImportError:
    from models import ActionLog
    from server.world_state import ClaimScenario


class TaskGrader:
    """Grades a completed investigation episode on four dimensions."""

    # Scoring weights (must sum to 1.0)
    WEIGHT_ACCURACY = 0.50
    WEIGHT_EVIDENCE = 0.25
    WEIGHT_REASONING = 0.15
    WEIGHT_EFFICIENCY = 0.10

    def grade(
        self,
        scenario: ClaimScenario,
        verdict: str,
        confidence: float,
        justification: str,
        action_history: List[ActionLog],
        step_count: int,
        total_step_reward: float,
    ) -> dict:
        """Return a dict of per-dimension scores plus weighted final_score.

        Args:
            scenario: The ClaimScenario used in this episode.
            verdict: Agent's submitted verdict label.
            confidence: Agent's self-reported confidence (0.0–1.0).
            justification: Agent's written reasoning.
            action_history: Full ordered log of actions taken.
            step_count: Total steps used.
            total_step_reward: Sum of step rewards (for reference only).

        Returns:
            dict with keys: accuracy, evidence_alignment, reasoning_depth,
            efficiency, final_score, scenario_id, difficulty, truth_label,
            agent_verdict, total_step_reward.
        """
        scores: dict = {}

        # 1. Accuracy — binary: correct label or not
        scores["accuracy"] = 1.0 if verdict == scenario.truth_label else 0.0

        # 2. Evidence alignment — fraction of key sources opened before verdict
        # ActionLog.result_summary for open_source is always "Opened: <source_id>"
        opened: set = set()
        for log in action_history:
            if log.action_type == "open_source" and log.result_summary.startswith(
                "Opened: "
            ):
                sid = log.result_summary[len("Opened: ") :].strip()
                opened.add(sid)

        key_ids = set(scenario.key_evidence_source_ids)
        coverage = len(opened & key_ids) / max(len(key_ids), 1)
        scores["evidence_alignment"] = round(coverage, 4)

        # 3. Reasoning depth — did justification cite source IDs or domains?
        justification_lower = (justification or "").lower()
        source_mentions = sum(
            1
            for src in scenario.sources
            if src.source_id.lower() in justification_lower
            or src.domain.lower() in justification_lower
        )
        scores["reasoning_depth"] = min(1.0, source_mentions / 2.0)

        # 4. Efficiency — fewer steps is better; bonus for finishing in ≤6
        efficiency = max(0.0, 1.0 - (step_count - 3) / 7.0)
        scores["efficiency"] = round(efficiency, 4)

        # Weighted final score
        final = (
            scores["accuracy"] * self.WEIGHT_ACCURACY
            + scores["evidence_alignment"] * self.WEIGHT_EVIDENCE
            + scores["reasoning_depth"] * self.WEIGHT_REASONING
            + scores["efficiency"] * self.WEIGHT_EFFICIENCY
        )
        scores["final_score"] = round(final, 4)

        # Metadata
        scores["scenario_id"] = scenario.scenario_id
        scores["difficulty"] = scenario.difficulty
        scores["truth_label"] = scenario.truth_label
        scores["agent_verdict"] = verdict
        scores["total_step_reward"] = round(total_step_reward, 4)

        return scores
