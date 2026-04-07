"""
Data models for the SniffTest Environment.

InvestigateAction — the agent's tool calls.
SniffTestObservation — what the agent sees each step.
SourceSummary / ActionLog — sub-models used inside the observation.
"""

from typing import List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class InvestigateAction(Action):
    """One step of agentic investigation.

    The agent picks an action_type and fills in the relevant fields.
    Unused fields should be left as None.
    """

    action_type: Literal[
        "search",  # search for sources about the claim
        "open_source",  # read full content of a specific source
        "cross_reference",  # compare two sources for contradictions
        "trace_origin",  # trace how the claim propagated
        "check_metadata",  # inspect source credibility signals
        "submit_verdict",  # final verdict — ends the episode
    ] = Field(..., description="The type of investigative action to take")

    query: Optional[str] = Field(
        None, description="Search query (used with action_type='search')"
    )
    source_id: Optional[str] = Field(
        None,
        description="Source ID to act on (used with open_source, check_metadata, trace_origin)",
    )
    source_ids: Optional[List[str]] = Field(
        None,
        description="Exactly two source IDs to compare (used with cross_reference)",
    )
    verdict: Optional[Literal["true", "false", "misleading", "unverifiable"]] = Field(
        None, description="Verdict label (used with submit_verdict)"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in verdict 0.0–1.0"
    )
    justification: Optional[str] = Field(
        None,
        description="Reasoning for verdict, citing source IDs or domains (used with submit_verdict)",
    )


# ---------------------------------------------------------------------------
# Sub-models used inside the observation
# ---------------------------------------------------------------------------


class SourceSummary(BaseModel):
    """A source the agent can see in the observation.

    Reliability score is intentionally hidden — the agent must infer
    credibility from content and metadata signals.
    """

    source_id: str = Field(..., description="Unique source identifier")
    title: str = Field(..., description="Article or page title")
    domain: str = Field(..., description="Domain/publication name")
    snippet: str = Field(..., description="Short preview (2-3 sentences)")
    retrieved: bool = Field(
        default=False, description="Whether the agent has opened this source"
    )


class ActionLog(BaseModel):
    """Record of a single past action in the episode history."""

    step: int = Field(..., description="Step number")
    action_type: str = Field(..., description="Type of action taken")
    result_summary: str = Field(
        ...,
        description=(
            "Brief summary of the result. "
            "For open_source actions, always formatted as 'Opened: <source_id>'."
        ),
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class SniffTestObservation(Observation):
    """What the agent sees at each step of an investigation episode.

    Reliability scores are intentionally hidden — the agent must infer
    source credibility from content, metadata, and cross-referencing.
    """

    claim: str = Field(default="", description="The claim under investigation")
    available_sources: List[SourceSummary] = Field(
        default_factory=list,
        description="Sources discovered so far (grows as the agent searches)",
    )
    opened_content: Optional[str] = Field(
        None, description="Full text of the most recently opened source"
    )
    cross_reference_result: Optional[str] = Field(
        None, description="Result of the most recent cross_reference action"
    )
    trace_result: Optional[str] = Field(
        None, description="Result of the most recent trace_origin action"
    )
    metadata_result: Optional[str] = Field(
        None, description="Result of the most recent check_metadata action"
    )
    action_history: List[ActionLog] = Field(
        default_factory=list,
        description="Ordered log of all actions taken this episode",
    )
    step_count: int = Field(default=0, description="Steps taken so far")
    steps_remaining: int = Field(default=10, description="Steps left before timeout")
    message: str = Field(
        default="", description="Thematic feedback message for the agent"
    )
