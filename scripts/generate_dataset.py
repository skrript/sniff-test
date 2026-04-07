#!/usr/bin/env python3
"""
Dataset generation script for SniffTest.

Calls the Anthropic API (claude-sonnet-4-20250514) to generate 20 claim
investigation scenarios: 7 easy, 7 medium, 6 hard.

Run once: python scripts/generate_dataset.py
Output:   data/claims_dataset.json

IMPORTANT (from BUILD_NOTES.md):
  After generating, manually open the JSON and eyeball 5 scenarios.
  Check that full_content has enough decisive detail and that
  key_evidence_source_ids actually contain the critical evidence.
  If quality looks off, tweak the PROMPT and regenerate.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT_PATH = ROOT / "data" / "claims_dataset.json"


SYSTEM = (
    "You are generating a synthetic dataset for a misinformation investigation "
    "RL environment. Return ONLY valid JSON, no markdown, no preamble."
)

PROMPT = """Generate 20 claim investigation scenarios for a fact-checking RL environment.

Return a JSON array of 20 objects. Each object MUST match this exact schema:

{
  "scenario_id": "scenario_001",
  "difficulty": "easy" | "medium" | "hard",
  "claim": "A specific factual claim an agent must investigate",
  "truth_label": "true" | "false" | "misleading" | "unverifiable",
  "manipulation_type": null | "fabricated" | "cherry_picked" | "outdated_context" | "coordinated_campaign" | "misleading_framing",
  "sources": [
    {
      "source_id": "src_001_a",
      "title": "Article title",
      "domain": "reuters-global.com",
      "snippet": "2-3 sentence preview of the article",
      "full_content": "Full 6-8 sentence article content with specific dates, stats, quotes, and named researchers or organisations",
      "reliability_score": 0.92,
      "bias_type": null,
      "publish_date": "2024-03-15",
      "author": "Jane Smith",
      "is_original_source": true
    }
  ],
  "propagation_chain": [
    {"node_id": "p1", "platform": "academic_journal", "timestamp": "2024-01-10T09:00:00Z", "reach": 500},
    {"node_id": "p2", "platform": "news_site", "timestamp": "2024-01-12T14:00:00Z", "reach": 50000},
    {"node_id": "p3", "platform": "twitter", "timestamp": "2024-01-13T08:00:00Z", "reach": 200000}
  ],
  "key_evidence_source_ids": ["src_001_b", "src_001_c"],
  "grader_notes": "The claim is [verdict] because [specific reasoning]. The key evidence is in [source_id] which shows [detail]. Smell test verdict: ROTTEN|FRESH|MUSTY|STALE."
}

Requirements:
- 7 EASY scenarios: clear true/false, 4-5 high-quality sources, minimal noise, obvious verdict
- 7 MEDIUM scenarios: conflicting sources, partial truths, verdict requires cross-referencing 2+ sources
- 6 HARD scenarios: coordinated misinformation, misleading_framing, or outdated_context; noisy sources; requires trace_origin + check_metadata

Topics: health, science, climate, technology, economics, food safety, public policy (no real people by name)
- Use fictional-but-realistic organisation and researcher names
- Unreliable sources: reliability_score < 0.4, appropriate bias_type
- Reliable sources: reliability_score > 0.75, author present, reputable domain
- Full content MUST contain enough detail to be decisive when combined correctly
- Each scenario: 4-6 sources (mix of reliable and unreliable)
- key_evidence_source_ids: 2-3 sources that together prove the correct verdict
- grader_notes must end with: "Smell test verdict: ROTTEN." (false/misleading) or "Smell test verdict: FRESH." (true) or "Smell test verdict: MUSTY." (unverifiable) or "Smell test verdict: STALE." (outdated_context)

Return ONLY the JSON array. No markdown fences, no comments, no extra text."""


def generate() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr
        )
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print(
            "ERROR: anthropic package not installed. Run: pip install anthropic",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Generating dataset via Claude API...")
    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        system=SYSTEM,
        messages=[{"role": "user", "content": PROMPT}],
    )

    raw = response.content[0].text.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Failed to parse JSON response: {exc}", file=sys.stderr)
        print("Raw response (first 500 chars):", raw[:500], file=sys.stderr)
        sys.exit(1)

    if not isinstance(data, list):
        print("ERROR: Expected a JSON array at the top level.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f, indent=2)

    # Summary
    by_diff = {"easy": 0, "medium": 0, "hard": 0}
    for s in data:
        by_diff[s.get("difficulty", "unknown")] = (
            by_diff.get(s.get("difficulty", "unknown"), 0) + 1
        )

    print(f"Generated {len(data)} scenarios → {OUTPUT_PATH}")
    print(
        f"  easy={by_diff.get('easy', 0)}  medium={by_diff.get('medium', 0)}  hard={by_diff.get('hard', 0)}"
    )
    print()
    print("Next step: manually inspect 5 scenarios in data/claims_dataset.json.")
    print(
        "Check that full_content is decisive and key_evidence_source_ids are correct."
    )


if __name__ == "__main__":
    generate()
