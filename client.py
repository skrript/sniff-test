"""SniffTest Environment Client."""

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ActionLog, InvestigateAction, SniffTestObservation, SourceSummary


class SniffTestEnv(EnvClient[InvestigateAction, SniffTestObservation, State]):
    """
    Client for the SniffTest misinformation investigation environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated session with isolated episode state.

    Example (async):
        >>> async with SniffTestEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task_level="easy")
        ...     obs = result.observation
        ...     print(obs.claim)
        ...
        ...     action = InvestigateAction(action_type="search", query="climate")
        ...     result = await env.step(action)
        ...     print(result.reward)

    Example (sync via context manager):
        >>> with SniffTestEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset(task_level="medium")
        ...     action = InvestigateAction(
        ...         action_type="submit_verdict",
        ...         verdict="false",
        ...         confidence=0.9,
        ...         justification="Based on src_001_b which contradicts the claim.",
        ...     )
        ...     result = env.step(action)
        ...     print(result.reward)
    """

    def _step_payload(self, action: InvestigateAction) -> Dict:
        """Convert InvestigateAction to JSON payload."""
        return action.model_dump(exclude_none=True, exclude={"metadata"})

    def _parse_result(self, payload: Dict) -> StepResult[SniffTestObservation]:
        """Parse server WebSocket response into StepResult[SniffTestObservation]."""
        obs_data = payload.get("observation", {})
        observation = self._parse_observation(obs_data, payload)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_observation(self, obs_data: Dict, payload: Dict) -> SniffTestObservation:
        """Reconstruct SniffTestObservation from the raw dict."""
        raw_sources: List[Dict] = obs_data.get("available_sources", [])
        available_sources = [
            SourceSummary(
                source_id=s["source_id"],
                title=s["title"],
                domain=s["domain"],
                snippet=s["snippet"],
                retrieved=s.get("retrieved", False),
            )
            for s in raw_sources
        ]

        raw_history: List[Dict] = obs_data.get("action_history", [])
        action_history = [
            ActionLog(
                step=h["step"],
                action_type=h["action_type"],
                result_summary=h["result_summary"],
            )
            for h in raw_history
        ]

        return SniffTestObservation(
            claim=obs_data.get("claim", ""),
            available_sources=available_sources,
            opened_content=obs_data.get("opened_content"),
            cross_reference_result=obs_data.get("cross_reference_result"),
            trace_result=obs_data.get("trace_result"),
            metadata_result=obs_data.get("metadata_result"),
            action_history=action_history,
            step_count=obs_data.get("step_count", 0),
            steps_remaining=obs_data.get("steps_remaining", 10),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State (extra fields allowed by State base class)."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            is_adversarial_episode=payload.get("is_adversarial_episode", False),
            confirmed_weaknesses=payload.get("confirmed_weaknesses", []),
            episodes_completed=payload.get("episodes_completed", 0),
            adversarial_cache_size=payload.get("adversarial_cache_size", 0),
        )
