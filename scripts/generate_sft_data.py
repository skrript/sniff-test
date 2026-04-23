"""
Generate expert demonstration trajectories for SniffTest SFT warm start.

Usage:
    export OPENAI_API_KEY=...
    python3 scripts/generate_sft_data.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "data" / "claims_dataset.json"
OUTPUT_PATH = ROOT / "data" / "sft_trajectories.jsonl"
MODEL_NAME = os.getenv("OPENAI_SFT_MODEL", "gpt-5-mini")

SYSTEM_PROMPT = """You are an expert fact-checker demonstrating ideal investigation technique.
Given a claim and available sources, produce a precise step-by-step investigation.
Return ONLY a JSON array of actions. Each action must match exactly one of these schemas:
{"action_type": "search", "query": "string"}
{"action_type": "open_source", "source_id": "string"}
{"action_type": "cross_reference", "source_ids": ["string", "string"]}
{"action_type": "trace_origin", "source_id": "string"}
{"action_type": "check_metadata", "source_id": "string"}
{"action_type": "submit_verdict", "verdict": "true|false|misleading|unverifiable",
 "confidence": 0.0, "justification": "detailed reasoning citing specific sources"}
No preamble. No markdown. Only the JSON array."""


def _load_scenarios() -> list[dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text())


def _prompt_for_scenario(scenario: dict[str, Any]) -> str:
    visible_sources = "\n".join(
        f"- [{source['source_id']}] {source['title']} ({source['domain']}): {source['snippet']}"
        for source in scenario["sources"][:3]
    )
    return f"""Claim: {scenario['claim']}

Available sources:
{visible_sources}

Ground truth (for your reference only, not visible to agent):
- Truth label: {scenario['truth_label']}
- Key evidence sources: {scenario['key_evidence_source_ids']}
- Grader notes: {scenario['grader_notes']}

Produce an ideal 5-8 step investigation sequence that:
1. Starts with a targeted search.
2. Opens decisive evidence sources.
3. Uses cross-reference, metadata, or trace when useful.
4. Ends with a well-reasoned verdict citing concrete evidence."""


def _generate_trajectory(client: OpenAI, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _prompt_for_scenario(scenario)},
        ],
    )
    return json.loads(response.output_text)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to generate SFT trajectories.")

    client = OpenAI()
    scenarios = _load_scenarios()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with OUTPUT_PATH.open("w") as out_f:
        for scenario in scenarios:
            scenario_id = scenario["scenario_id"]
            print(f"Generating trajectory for {scenario_id}...")
            try:
                actions = _generate_trajectory(client, scenario)
            except Exception as exc:
                print(f"  Failed: {exc}")
                continue

            record = {
                "scenario_id": scenario_id,
                "difficulty": scenario["difficulty"],
                "claim": scenario["claim"],
                "actions": actions,
                "truth_label": scenario["truth_label"],
            }
            out_f.write(json.dumps(record) + "\n")
            written += 1

    print(f"Generated {written} trajectories -> {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
