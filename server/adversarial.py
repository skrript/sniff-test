"""
Adversarial scenario generator — tracks agent weaknesses and generates
targeted ClaimScenario batches using OpenAI when weaknesses are confirmed.

Design principles (from BUILD_NOTES.md):
- Lazy generation: OpenAI is called ONCE when a weakness threshold is crossed,
  producing a batch of 5 scenarios cached in memory. Per-episode calls = slow.
- Graceful fallback: If OPENAI_API_KEY is missing, silently falls back to
  the static dataset. No crashes.
- Same grader: Adversarially generated scenarios use the identical ClaimScenario
  schema, so TaskGrader works unchanged.
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EpisodeResult:
    """Outcome of a completed episode recorded for weakness analysis."""

    scenario_id: str
    difficulty: str
    manipulation_type: Optional[str]
    truth_label: str
    agent_verdict: str
    accuracy: float
    evidence_alignment: float
    used_cross_reference: bool
    used_trace_origin: bool
    used_check_metadata: bool
    step_count: int
    timed_out: bool


class WeaknessTracker:
    """Tracks agent performance across episodes and identifies patterns."""

    WEAKNESS_THRESHOLD = 0.35
    MIN_EPISODES = 5

    def __init__(self) -> None:
        self.history: List[EpisodeResult] = []

    def record(self, result: EpisodeResult) -> None:
        self.history.append(result)

    def get_weaknesses(self) -> Dict:
        """Analyse history and return confirmed weaknesses.

        Returns an empty dict if fewer than MIN_EPISODES have been played.
        """
        if len(self.history) < self.MIN_EPISODES:
            return {}

        weaknesses: Dict = {}

        # 1. Manipulation types the agent consistently fails on
        by_manipulation: Dict[str, List[float]] = defaultdict(list)
        for ep in self.history:
            if ep.manipulation_type:
                by_manipulation[ep.manipulation_type].append(ep.accuracy)

        weak_manipulations = [
            mtype
            for mtype, scores in by_manipulation.items()
            if len(scores) >= 2
            and (sum(scores) / len(scores)) < self.WEAKNESS_THRESHOLD
        ]
        if weak_manipulations:
            weaknesses["manipulation_types"] = weak_manipulations

        # 2. Verdict confusion: what does the agent mislabel as what?
        confusion: Dict[str, List[str]] = defaultdict(list)
        for ep in self.history:
            if ep.accuracy == 0.0:
                confusion[ep.truth_label].append(ep.agent_verdict)
        if confusion:
            weaknesses["verdict_confusion"] = {
                truth: max(set(preds), key=preds.count)
                for truth, preds in confusion.items()
                if preds
            }

        # 3. Tool underuse
        n = len(self.history)
        tool_rates = {
            "cross_reference": sum(1 for ep in self.history if ep.used_cross_reference)
            / n,
            "trace_origin": sum(1 for ep in self.history if ep.used_trace_origin) / n,
            "check_metadata": sum(1 for ep in self.history if ep.used_check_metadata)
            / n,
        }
        underused = [t for t, rate in tool_rates.items() if rate < 0.3]
        if underused:
            weaknesses["underused_tools"] = underused

        # 4. Evidence skipping
        avg_alignment = sum(ep.evidence_alignment for ep in self.history) / n
        if avg_alignment < 0.5:
            weaknesses["skips_key_evidence"] = True

        # 5. Frequent timeouts
        timeout_rate = sum(1 for ep in self.history if ep.timed_out) / n
        if timeout_rate > 0.4:
            weaknesses["frequent_timeouts"] = True

        return weaknesses

    def should_generate(self) -> bool:
        return bool(self.get_weaknesses())

    def summary_for_prompt(self) -> str:
        """Human-readable weakness description for the generation prompt."""
        w = self.get_weaknesses()
        if not w:
            return "No confirmed weaknesses yet."
        lines = []
        if "manipulation_types" in w:
            lines.append(
                f"- Fails on manipulation types: {', '.join(w['manipulation_types'])}"
            )
        if "verdict_confusion" in w:
            for truth, pred in w["verdict_confusion"].items():
                lines.append(f"- Mislabels '{truth}' claims as '{pred}'")
        if "underused_tools" in w:
            lines.append(f"- Rarely uses tools: {', '.join(w['underused_tools'])}")
        if w.get("skips_key_evidence"):
            lines.append(
                "- Frequently skips key evidence sources before submitting verdict"
            )
        if w.get("frequent_timeouts"):
            lines.append("- Often runs out of steps without submitting a verdict")
        return "\n".join(lines)


class AdversarialGenerator:
    """Generates targeted ClaimScenario batches using OpenAI.

    Scenarios are cached in memory and served until exhausted.
    Falls back to the static dataset gracefully if the API key is missing.
    """

    BATCH_SIZE = 5

    def __init__(self, tracker: WeaknessTracker) -> None:
        self.tracker = tracker
        self._cache: List[dict] = []
        self._api_key = os.environ.get("OPENAI_API_KEY")
        self._client = None
        if self._api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                pass

    def is_available(self) -> bool:
        return self._client is not None

    def maybe_generate(self) -> bool:
        """Trigger generation if conditions are met; return True if generated."""
        if not self.is_available():
            return False
        if self._cache:
            return False
        if not self.tracker.should_generate():
            return False
        try:
            self._generate_batch()
            return True
        except Exception as exc:
            print(
                f"[AdversarialGenerator] Generation failed (non-fatal): {exc}",
                flush=True,
            )
            return False

    def pop_scenario(self) -> Optional[dict]:
        """Return the next cached adversarial scenario dict, or None."""
        if not self._cache:
            return None
        return self._cache.pop(0)

    def _generate_batch(self) -> None:
        weakness_summary = self.tracker.summary_for_prompt()

        prompt = f"""You are generating adversarial test scenarios for a misinformation investigation RL environment.

The AI agent being tested has shown these specific weaknesses:
{weakness_summary}

Generate exactly {self.BATCH_SIZE} claim investigation scenarios that directly exploit these weaknesses.
Each scenario should make the agent's weakness the deciding factor.

Return ONLY a JSON array of {self.BATCH_SIZE} objects. Each object must match this exact schema:

{{
  "scenario_id": "adv_<unique_4_digit_number>",
  "difficulty": "hard",
  "claim": "A specific factual claim the agent must investigate",
  "truth_label": "true" | "false" | "misleading" | "unverifiable",
  "manipulation_type": "fabricated" | "cherry_picked" | "outdated_context" | "coordinated_campaign" | "misleading_framing" | null,
  "sources": [
    {{
      "source_id": "adv_<scenario_number>_<letter>",
      "title": "Article title",
      "domain": "example-domain.com",
      "snippet": "2-3 sentence preview",
      "full_content": "Full 6-8 sentence content with specific details",
      "reliability_score": 0.0-1.0,
      "bias_type": null | "cherry_picked" | "outdated_context" | "fabricated",
      "publish_date": "YYYY-MM-DD",
      "author": "Name or null",
      "is_original_source": true | false
    }}
  ],
  "propagation_chain": [
    {{"node_id": "p1", "platform": "platform_name", "timestamp": "ISO8601", "reach": integer}}
  ],
  "key_evidence_source_ids": ["source_id_1", "source_id_2"],
  "grader_notes": "Why this is the correct verdict and which sources prove it. Smell test verdict: ROTTEN|FRESH|MUSTY|STALE."
}}

Targeting rules:
- If agent fails on 'coordinated_campaign': use 4+ consistent false sources, one buried authoritative contradiction
- If agent mislabels 'misleading' as 'false': make claim technically true but critically missing context
- If agent underuses 'cross_reference': make verdict only determinable by comparing two specific sources
- If agent underuses 'trace_origin': hide the fabrication signal in the propagation chain origin
- If agent underuses 'check_metadata': put the key credibility signal in author/date metadata
- If agent skips key evidence: hide decisive evidence behind a non-obvious search query

Return ONLY the JSON array. No markdown, no preamble."""

        response = self._client.responses.create(
            model="gpt-5-mini",
            max_output_tokens=6000,
            input=[
                {
                    "role": "system",
                    "content": "You are generating synthetic datasets for AI research. Return only valid JSON arrays.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw = response.output_text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        scenarios = json.loads(raw)
        self._cache.extend(scenarios)
        print(
            f"[AdversarialGenerator] Generated {len(scenarios)} adversarial scenarios "
            f"targeting: {self.tracker.summary_for_prompt()}",
            flush=True,
        )
