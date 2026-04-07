"""
ToolEngine — deterministic tool implementations backed by a ClaimScenario.

All lookups are against the loaded scenario data — no live APIs, no randomness
at inference time. This ensures reproducible grading.
"""

from typing import List, Set

try:
    from ..models import SourceSummary
    from .world_state import ClaimScenario
except ImportError:
    from models import SourceSummary
    from server.world_state import ClaimScenario


class ToolEngine:
    """Dispatches tool calls for one episode against a fixed ClaimScenario."""

    def __init__(self, scenario: ClaimScenario) -> None:
        self.scenario = scenario
        self._source_map = {s.source_id: s for s in scenario.sources}
        self._opened_sources: Set[str] = set()
        self._discovered_source_ids: Set[str] = set()

        # Information unlocking: agent starts with only 3 sources visible
        # (the highest-reliability ones). More are discovered via search.
        initial = sorted(scenario.sources, key=lambda s: -s.reliability_score)[:3]
        for src in initial:
            self._discovered_source_ids.add(src.source_id)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def visible_sources(self) -> List[SourceSummary]:
        """Sources the agent has discovered so far (grows via search)."""
        return [
            SourceSummary(
                source_id=s.source_id,
                title=s.title,
                domain=s.domain,
                snippet=s.snippet,
                retrieved=s.source_id in self._opened_sources,
            )
            for s in self.scenario.sources
            if s.source_id in self._discovered_source_ids
        ]

    def search(self, query: str) -> List[SourceSummary]:
        """Return sources matching the query (keyword scoring).

        Only sources with at least one keyword match are returned —
        poor queries return fewer results, rewarding strategic search.
        New matches are added to the discovered set.
        """
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if len(w) > 2]

        scored = []
        for src in self.scenario.sources:
            haystack = (src.title + " " + src.snippet).lower()
            score = sum(1 for word in keywords if word in haystack)
            if score > 0:
                scored.append((score, src))

        if not scored:
            fallback = self.scenario.sources[:2]
            scored = [(0, s) for s in fallback]

        scored.sort(key=lambda x: -x[0])

        for _, src in scored:
            self._discovered_source_ids.add(src.source_id)

        return [
            SourceSummary(
                source_id=s.source_id,
                title=s.title,
                domain=s.domain,
                snippet=s.snippet,
                retrieved=s.source_id in self._opened_sources,
            )
            for _, s in scored
        ]

    def open_source(self, source_id: str) -> str:
        """Return the full content of a source (reveals hidden text)."""
        src = self._source_map.get(source_id)
        if not src:
            return f"ERROR: Source '{source_id}' not found."
        self._opened_sources.add(source_id)
        self._discovered_source_ids.add(source_id)
        return (
            f"[{src.title}]\n"
            f"Domain: {src.domain} | Published: {src.publish_date} | "
            f"Author: {src.author or 'Unknown'}\n\n"
            f"{src.full_content}"
        )

    def cross_reference(self, source_id_a: str, source_id_b: str) -> str:
        """Compare two sources and return a structured contradiction assessment.

        Uses reliability score difference — not brittle bias_type matching —
        to determine contradiction likelihood.
        """
        src_a = self._source_map.get(source_id_a)
        src_b = self._source_map.get(source_id_b)
        if not src_a:
            return f"ERROR: Source '{source_id_a}' not found."
        if not src_b:
            return f"ERROR: Source '{source_id_b}' not found."

        rel_diff = abs(src_a.reliability_score - src_b.reliability_score)

        if rel_diff > 0.5:
            low, high = (
                (src_a, src_b)
                if src_a.reliability_score < src_b.reliability_score
                else (src_b, src_a)
            )
            return (
                "💀 LIKELY CONTRADICTION DETECTED:\n"
                f"• {high.domain} ({high.source_id}): {high.snippet}\n"
                f"• {low.domain} ({low.source_id}): {low.snippet}\n"
                "Note: These sources present significantly different accounts. "
                "One appears substantially less credible."
            )
        elif rel_diff > 0.2:
            return (
                "⚠️ PARTIAL DISAGREEMENT:\n"
                f"• {src_a.domain} ({source_id_a}): {src_a.snippet}\n"
                f"• {src_b.domain} ({source_id_b}): {src_b.snippet}\n"
                "Note: Sources agree on some points but diverge on key details."
            )
        else:
            return (
                "✅ SOURCES BROADLY AGREE:\n"
                f"• {src_a.domain}: {src_a.snippet}\n"
                f"• {src_b.domain}: {src_b.snippet}"
            )

    def trace_origin(self, source_id: str) -> str:
        """Return the propagation chain showing how the claim spread."""
        chain = self.scenario.propagation_chain
        original = next(
            (s for s in self.scenario.sources if s.is_original_source), None
        )
        chain_str = " → ".join(
            f"{n.platform} ({n.timestamp[:10]}, reach: {n.reach:,})" for n in chain
        )
        origin_note = (
            f"\nOriginal source: {original.domain} — {original.title}"
            if original
            else ""
        )
        return f"Propagation chain:\n{chain_str}{origin_note}"

    def check_metadata(self, source_id: str) -> str:
        """Return metadata about a source with credibility flag analysis."""
        src = self._source_map.get(source_id)
        if not src:
            return f"ERROR: Source '{source_id}' not found."

        flags = []
        if src.reliability_score < 0.4:
            flags.append("⚠️ Low-credibility domain")
        if src.bias_type == "outdated_context":
            flags.append("⚠️ Article may be outdated or missing recent context")
        if src.bias_type == "fabricated":
            flags.append("⚠️ No verifiable author or institution")
        if not src.author:
            flags.append("⚠️ Anonymous authorship")
        if src.bias_type == "cherry_picked":
            flags.append("⚠️ May present selectively chosen statistics")

        flags_str = (
            "\n".join(flags) if flags else "✓ No major credibility flags detected"
        )
        return (
            f"Metadata for {source_id}:\n"
            f"Domain:    {src.domain}\n"
            f"Author:    {src.author or 'Unknown'}\n"
            f"Published: {src.publish_date}\n"
            f"Credibility flags:\n{flags_str}"
        )
