"""
SniffTest Environment — core OpenEnv-compliant environment.

Episode lifecycle:
1. reset(task_level) — selects a ClaimScenario, initialises ToolEngine + RewardEngine,
   returns initial observation with claim and 3 visible sources.
2. step(action) — dispatches to ToolEngine, computes step reward, updates observation.
3. Episode ends when submit_verdict is called OR step_count >= max_steps.
   On timeout: -0.5 penalty, verdict recorded as "timeout" (scores 0.0 accuracy).

Architecture notes (from BUILD_NOTES.md):
- Reliability scores are hidden from the agent in SniffTestObservation.
- _record_episode_to_tracker() is called in BOTH submit_verdict AND timeout paths.
- AdversarialGenerator is lazy: called only when WeaknessTracker has confirmed weaknesses.
"""

import json
import random
import uuid
from pathlib import Path
from typing import Any, List, Literal, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ActionLog,
        InvestigateAction,
        SniffTestObservation,
        SourceSummary,
    )
    from .adversarial import AdversarialGenerator, EpisodeResult, WeaknessTracker
    from .grader import TaskGrader
    from .reward import RewardEngine, format_reward
    from .tools import ToolEngine
    from .world_state import ClaimScenario
except (ImportError, ModuleNotFoundError):
    from models import ActionLog, InvestigateAction, SniffTestObservation
    from server.adversarial import AdversarialGenerator, EpisodeResult, WeaknessTracker
    from server.grader import TaskGrader
    from server.reward import RewardEngine, format_reward
    from server.tools import ToolEngine
    from server.world_state import ClaimScenario

_DATASET_PATH = Path(__file__).parent.parent / "data" / "claims_dataset.json"
_MAX_STEPS = 10
_TIMEOUT_PENALTY = -0.5


class SniffTestEnvironment(Environment):
    """Multi-step misinformation investigation environment.

    An agent receives a claim, uses tool calls to gather evidence, and submits
    a verdict against a hidden deterministic world state. Three difficulty tiers
    (easy / medium / hard) with increasing epistemic complexity.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()
        self._all_scenarios: List[dict] = self._load_dataset()
        self._weakness_tracker = WeaknessTracker()
        self._adversarial_gen = AdversarialGenerator(self._weakness_tracker)
        self._grader = TaskGrader()

        # Episode state (initialised in reset)
        self._episode_id: str = str(uuid.uuid4())
        self._step_count: int = 0
        self._total_step_reward: float = 0.0
        self._action_history: List[ActionLog] = []
        self._grade_result: Optional[dict] = None
        self._timed_out: bool = False
        self._is_adversarial_episode: bool = False
        self._current_scenario: Optional[ClaimScenario] = None
        self._tool_engine: Optional[ToolEngine] = None
        self._reward_engine: Optional[RewardEngine] = None
        self._last_obs: Optional[SniffTestObservation] = None
        self._last_reward_components: Optional[dict[str, float]] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_level: Literal["easy", "medium", "hard"] = "easy",
        **kwargs: Any,
    ) -> SniffTestObservation:
        """Reset the environment and start a new investigation episode.

        Args:
            seed: Optional random seed for reproducibility.
            episode_id: Optional custom episode identifier.
            task_level: Difficulty of the scenario to load ("easy"/"medium"/"hard").

        Returns:
            Initial SniffTestObservation with claim and 3 visible sources.
        """
        if seed is not None:
            random.seed(seed)

        # Try adversarial scenario first (only activates after enough episodes)
        self._adversarial_gen.maybe_generate()
        adv_scenario = self._adversarial_gen.pop_scenario()

        if adv_scenario:
            self._current_scenario = ClaimScenario(**adv_scenario)
            self._is_adversarial_episode = True
        else:
            candidates = [
                s for s in self._all_scenarios if s.get("difficulty") == task_level
            ]
            if not candidates:
                candidates = self._all_scenarios
            self._current_scenario = ClaimScenario(**random.choice(candidates))
            self._is_adversarial_episode = False

        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._total_step_reward = 0.0
        self._action_history = []
        self._grade_result = None
        self._timed_out = False
        self._tool_engine = ToolEngine(self._current_scenario)
        self._reward_engine = RewardEngine(self._current_scenario)
        self._last_reward_components = None

        obs = self._build_observation(
            message="🔍 New claim incoming. Does it pass the sniff test?"
        )
        self._last_obs = obs
        return obs

    def step(
        self,
        action: InvestigateAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SniffTestObservation:
        """Execute one investigative action and return the updated observation.

        Returns:
            SniffTestObservation with reward and done set appropriately.
        """
        if self._current_scenario is None or self._tool_engine is None:
            # Auto-reset with default easy level when step is called on a fresh
            # instance (e.g. via stateless HTTP /step without a prior /reset).
            self.reset()

        self._step_count += 1

        # Dispatch action to the tool engine
        tool_result, obs_kwargs = self._dispatch_action(action)

        # Log the action
        result_summary = self._make_result_summary(action, tool_result)
        self._action_history.append(
            ActionLog(
                step=self._step_count,
                action_type=action.action_type,
                result_summary=result_summary,
            )
        )

        # Compute step reward
        step_reward = self._reward_engine.compute_step_reward(
            action, tool_result, self._step_count
        )

        done = False
        final_reward = step_reward
        reward_components = None

        if action.action_type == "submit_verdict":
            # Grade the episode and compute terminal reward
            grade = self._grader.grade(
                scenario=self._current_scenario,
                verdict=action.verdict or "unverifiable",
                confidence=action.confidence or 0.0,
                justification=action.justification or "",
                action_history=self._action_history,
                step_count=self._step_count,
                total_step_reward=self._total_step_reward,
            )
            self._grade_result = grade
            reward_components = {
                "accuracy": round(grade["accuracy"] * self._grader.WEIGHT_ACCURACY, 4),
                "evidence": round(
                    grade["evidence_alignment"] * self._grader.WEIGHT_EVIDENCE, 4
                ),
                "reasoning": round(
                    grade["reasoning_depth"] * self._grader.WEIGHT_REASONING, 4
                ),
                "efficiency": round(
                    grade["efficiency"] * self._grader.WEIGHT_EFFICIENCY, 4
                ),
                "format": round(format_reward(action), 4),
                "anti_hack": round(step_reward, 4),
            }
            final_reward = round(sum(reward_components.values()), 4)
            done = True
            self._record_episode_to_tracker(grade, timed_out=False)
            obs_kwargs["message"] = self._verdict_message(action.verdict or "", grade)
            obs_kwargs["reward_components"] = reward_components

        elif self._step_count >= _MAX_STEPS:
            # Timeout: penalise and end
            timeout_grade = self._grader.grade(
                scenario=self._current_scenario,
                verdict="timeout",
                confidence=0.0,
                justification="",
                action_history=self._action_history,
                step_count=self._step_count,
                total_step_reward=self._total_step_reward,
            )
            self._grade_result = timeout_grade
            final_reward = _TIMEOUT_PENALTY
            done = True
            self._timed_out = True
            self._record_episode_to_tracker(timeout_grade, timed_out=True)
            obs_kwargs["message"] = (
                "⏰ Ran out of time. The smell test is inconclusive."
            )

        self._total_step_reward += step_reward
        self._last_reward_components = reward_components

        obs = self._build_observation(
            tool_result=tool_result,
            action=action,
            done=done,
            reward=final_reward,
            **obs_kwargs,
        )
        self._last_obs = obs
        return obs

    @property
    def state(self) -> State:
        """Current environment state (exposed in /state endpoint and web UI)."""
        weaknesses = self._weakness_tracker.get_weaknesses()
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count,
            is_adversarial_episode=self._is_adversarial_episode,
            confirmed_weaknesses=list(weaknesses.keys()),
            episodes_completed=len(self._weakness_tracker.history),
            adversarial_cache_size=len(self._adversarial_gen._cache),
            current_difficulty=(
                self._current_scenario.difficulty if self._current_scenario else None
            ),
            grade_result=self._grade_result,
        )

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata

        return EnvironmentMetadata(
            name="snifftest-env",
            description=(
                "A multi-step misinformation investigation environment where an agent "
                "sniffs out false claims, gathers evidence through tool calls, and submits "
                "a smell-test verdict. Tests epistemic reasoning under uncertainty."
            ),
            version="0.1.0",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dataset(self) -> List[dict]:
        if not _DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found at {_DATASET_PATH}. "
                "Run scripts/generate_dataset.py to generate it."
            )
        with open(_DATASET_PATH, "r") as f:
            data = json.load(f)
        if not data:
            raise ValueError("claims_dataset.json is empty.")
        return data

    def _build_observation(
        self,
        tool_result: str = "",
        action: Optional[InvestigateAction] = None,
        done: bool = False,
        reward: Optional[float] = None,
        message: str = "",
        **kwargs,
    ) -> SniffTestObservation:
        assert self._current_scenario is not None
        assert self._tool_engine is not None

        opened_content = None
        cross_reference_result = None
        trace_result = None
        metadata_result = None

        if action:
            atype = action.action_type
            if atype == "open_source":
                opened_content = tool_result
            elif atype == "cross_reference":
                cross_reference_result = tool_result
            elif atype == "trace_origin":
                trace_result = tool_result
            elif atype == "check_metadata":
                metadata_result = tool_result

        return SniffTestObservation(
            claim=self._current_scenario.claim,
            available_sources=self._tool_engine.visible_sources,
            opened_content=opened_content,
            cross_reference_result=cross_reference_result,
            trace_result=trace_result,
            metadata_result=metadata_result,
            action_history=list(self._action_history),
            step_count=self._step_count,
            steps_remaining=max(0, _MAX_STEPS - self._step_count),
            message=message,
            done=done,
            reward=reward,
            reward_components=kwargs.get("reward_components"),
        )

    def _dispatch_action(self, action: InvestigateAction) -> tuple:
        """Route the action to the appropriate ToolEngine method."""
        te = self._tool_engine
        atype = action.action_type
        extra: dict = {}

        if atype == "search":
            query = action.query or ""
            results = te.search(query)
            tool_result = (
                f"Search for '{query}' returned {len(results)} sources:\n"
                + "\n".join(
                    f"  [{s.source_id}] {s.title} ({s.domain}): {s.snippet}"
                    for s in results
                )
            )

        elif atype == "open_source":
            sid = action.source_id or ""
            tool_result = te.open_source(sid)
            if not tool_result.startswith("ERROR"):
                extra["message"] = (
                    "👃 You're onto something — this source has a strong scent."
                    if sid in self._current_scenario.key_evidence_source_ids
                    else "🔍 Source opened. Check the details carefully."
                )

        elif atype == "cross_reference":
            ids = action.source_ids or []
            if len(ids) != 2:
                tool_result = "ERROR: cross_reference requires exactly 2 source IDs."
            else:
                tool_result = te.cross_reference(ids[0], ids[1])
                if "CONTRADICTION" in tool_result:
                    extra["message"] = (
                        "💀 Foul odor detected — these sources reek of contradiction."
                    )
                elif "AGREE" in tool_result:
                    extra["message"] = "✅ Sources aligned. Smells clean so far."
                else:
                    extra["message"] = "⚠️ Sources diverge on some details."

        elif atype == "trace_origin":
            sid = action.source_id or ""
            tool_result = te.trace_origin(sid)

        elif atype == "check_metadata":
            sid = action.source_id or ""
            tool_result = te.check_metadata(sid)
            if "Low-credibility" in tool_result or "Anonymous" in tool_result:
                extra["message"] = "🤢 This one smells off. Check the metadata."

        elif atype == "submit_verdict":
            tool_result = f"Verdict submitted: {action.verdict}"

        else:
            tool_result = f"ERROR: Unknown action_type '{atype}'."

        return tool_result, extra

    def _make_result_summary(self, action: InvestigateAction, tool_result: str) -> str:
        """Build the result_summary string stored in ActionLog.

        IMPORTANT: open_source result_summary MUST be 'Opened: <source_id>'
        so the grader can reconstruct which sources were opened.
        """
        atype = action.action_type
        if atype == "open_source":
            sid = action.source_id or ""
            if not tool_result.startswith("ERROR"):
                return f"Opened: {sid}"
            return tool_result[:80]
        elif atype == "search":
            return f"Searched: '{action.query}'"
        elif atype == "cross_reference":
            return f"Cross-referenced: {action.source_ids}"
        elif atype == "trace_origin":
            return f"Traced origin for: {action.source_id}"
        elif atype == "check_metadata":
            return f"Checked metadata: {action.source_id}"
        elif atype == "submit_verdict":
            return f"Verdict: {action.verdict} (confidence={action.confidence})"
        return tool_result[:80]

    def _verdict_message(self, verdict: str, grade: dict) -> str:
        if grade["accuracy"] == 1.0:
            return "🏆 Nailed it. Your nose doesn't lie."
        return "❌ Wrong call. You got played — something slipped past your nose."

    def _record_episode_to_tracker(self, grade: dict, timed_out: bool) -> None:
        """Record episode outcome to WeaknessTracker for adversarial generation.

        Called in BOTH submit_verdict and timeout paths.
        """
        used_cross_ref = any(
            log.action_type == "cross_reference" for log in self._action_history
        )
        used_trace = any(
            log.action_type == "trace_origin" for log in self._action_history
        )
        used_metadata = any(
            log.action_type == "check_metadata" for log in self._action_history
        )

        result = EpisodeResult(
            scenario_id=self._current_scenario.scenario_id,  # type: ignore[union-attr]
            difficulty=self._current_scenario.difficulty,  # type: ignore[union-attr]
            manipulation_type=self._current_scenario.manipulation_type,  # type: ignore[union-attr]
            truth_label=self._current_scenario.truth_label,  # type: ignore[union-attr]
            agent_verdict=grade.get("agent_verdict", "timeout"),
            accuracy=grade.get("accuracy", 0.0),
            evidence_alignment=grade.get("evidence_alignment", 0.0),
            used_cross_reference=used_cross_ref,
            used_trace_origin=used_trace,
            used_check_metadata=used_metadata,
            step_count=self._step_count,
            timed_out=timed_out,
        )
        self._weakness_tracker.record(result)
