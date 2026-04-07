"""
World state models — hidden ground truth never exposed to the agent.

ClaimScenario is the full server-side representation of one investigation task.
Agents see only SniffTestObservation; they never see reliability_score or
full_content until they call open_source.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class SourceRecord(BaseModel):
    """A full source record as stored in the dataset (server-side only)."""

    source_id: str
    title: str
    domain: str
    snippet: str
    full_content: str = Field(
        ..., description="Only revealed to the agent via open_source action"
    )
    reliability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Hidden from agent — 0.0 (junk) to 1.0 (authoritative)",
    )
    bias_type: Optional[str] = Field(
        None,
        description=(
            "e.g. 'cherry_picked', 'outdated_context', 'fabricated', "
            "'coordinated_campaign', 'misleading_framing'"
        ),
    )
    publish_date: str
    author: Optional[str] = None
    is_original_source: bool = Field(
        ..., description="Whether this source is the origin of the claim"
    )


class PropagationNode(BaseModel):
    """One node in the claim's propagation chain."""

    node_id: str
    platform: str
    timestamp: str
    reach: int = Field(..., ge=0, description="Estimated audience size")


class ClaimScenario(BaseModel):
    """Complete investigation scenario loaded from the dataset."""

    scenario_id: str
    difficulty: Literal["easy", "medium", "hard"]
    claim: str
    truth_label: Literal["true", "false", "misleading", "unverifiable"]
    manipulation_type: Optional[str] = None
    sources: List[SourceRecord]
    propagation_chain: List[PropagationNode]
    key_evidence_source_ids: List[str] = Field(
        ..., description="Source IDs that contain the decisive evidence"
    )
    grader_notes: str = Field(
        ..., description="Human-readable explanation of the correct answer"
    )
