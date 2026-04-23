"""
Generate expert demonstration trajectories for SniffTest SFT warm start.

Uses the dedicated SFT scenario file rather than the RL claims dataset.

Usage:
    export OPENAI_API_KEY=...
    python3 scripts/generate_sft_data.py
    python3 scripts/generate_sft_data.py --input data/sft_scenarios.json --output data/sft_trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = ROOT / "data" / "sft_scenarios.json"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "sft_trajectories.jsonl"
MODEL_NAME = os.getenv("OPENAI_SFT_MODEL", "gpt-5-mini")

ACTION_FIELDS_BY_TYPE = {
    "search": {"query"},
    "open_source": {"source_id"},
    "cross_reference": {"source_ids"},
    "trace_origin": {"source_id"},
    "check_metadata": {"source_id"},
    "submit_verdict": {"verdict", "confidence", "justification"},
}
VALID_VERDICTS = {"true", "false", "misleading", "unverifiable"}

SYSTEM_PROMPT = """You are an expert fact-checker demonstrating ideal investigation technique.
Given a claim and three visible sources, produce a precise step-by-step investigation.
Return ONLY a JSON array of actions. Each action must match exactly one of these schemas:
{"action_type": "search", "query": "string"}
{"action_type": "open_source", "source_id": "string"}
{"action_type": "cross_reference", "source_ids": ["string", "string"]}
{"action_type": "trace_origin", "source_id": "string"}
{"action_type": "check_metadata", "source_id": "string"}
{"action_type": "submit_verdict", "verdict": "true|false|misleading|unverifiable",
 "confidence": 0.0, "justification": "detailed reasoning citing specific sources"}
Constraints:
- Use only the listed source IDs.
- Produce 4-6 actions total.
- Start with search.
- End with submit_verdict.
- Prefer trajectories that teach clean JSON formatting and disciplined tool use.
No preamble. No markdown. Only the JSON array."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    scenarios = json.loads(path.read_text())
    if not isinstance(scenarios, list):
        raise ValueError(f"{path} must contain a top-level JSON array.")
    return scenarios


def _prompt_for_scenario(scenario: dict[str, Any]) -> str:
    visible_sources = "\n".join(
        f"- [{source['source_id']}] {source['title']} ({source['domain']}): {source['snippet']}"
        for source in scenario["sources"][:3]
    )
    teaching_focus = ", ".join(scenario.get("teaching_focus", [])) or "clean action formatting"
    return f"""Claim: {scenario['claim']}

Available sources:
{visible_sources}

Ground truth (for your reference only, not visible to agent):
- Truth label: {scenario['truth_label']}
- Teaching focus: {teaching_focus}
- Scenario notes: {scenario.get('notes', '')}

Produce an ideal investigation sequence that:
1. Starts with a targeted search.
2. Opens decisive sources from the visible list.
3. Uses cross_reference, metadata, or trace only when it improves the investigation.
4. Ends with a well-reasoned verdict citing concrete source IDs."""


def _validate_actions(scenario: dict[str, Any], actions: list[dict[str, Any]]) -> None:
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{scenario['scenario_id']}: generation did not return a non-empty action list.")

    valid_source_ids = {source["source_id"] for source in scenario["sources"][:3]}
    for idx, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError(f"{scenario['scenario_id']}: action {idx} is not an object.")
        action_type = action.get("action_type")
        if action_type not in ACTION_FIELDS_BY_TYPE:
            raise ValueError(f"{scenario['scenario_id']}: action {idx} has invalid action_type {action_type!r}.")

        required_fields = ACTION_FIELDS_BY_TYPE[action_type]
        missing_fields = sorted(field for field in required_fields if field not in action)
        if missing_fields:
            raise ValueError(
                f"{scenario['scenario_id']}: action {idx} missing required fields {missing_fields}."
            )

        if action_type == "search":
            if not isinstance(action["query"], str) or not action["query"].strip():
                raise ValueError(f"{scenario['scenario_id']}: action {idx} has invalid search query.")
        elif action_type in {"open_source", "trace_origin", "check_metadata"}:
            if action["source_id"] not in valid_source_ids:
                raise ValueError(
                    f"{scenario['scenario_id']}: action {idx} references non-visible source_id {action['source_id']}."
                )
        elif action_type == "cross_reference":
            source_ids = action["source_ids"]
            if (
                not isinstance(source_ids, list)
                or len(source_ids) != 2
                or len(set(source_ids)) != 2
                or any(source_id not in valid_source_ids for source_id in source_ids)
            ):
                raise ValueError(
                    f"{scenario['scenario_id']}: action {idx} cross_reference must include two distinct visible source_ids."
                )
        elif action_type == "submit_verdict":
            if action["verdict"] not in VALID_VERDICTS:
                raise ValueError(f"{scenario['scenario_id']}: action {idx} has invalid verdict.")
            confidence = action["confidence"]
            if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
                raise ValueError(f"{scenario['scenario_id']}: action {idx} has invalid confidence.")
            if not isinstance(action["justification"], str) or not action["justification"].strip():
                raise ValueError(f"{scenario['scenario_id']}: action {idx} justification must be non-empty.")

    if actions[0].get("action_type") != "search":
        raise ValueError(f"{scenario['scenario_id']}: first action must be search.")
    if actions[-1].get("action_type") != "submit_verdict":
        raise ValueError(f"{scenario['scenario_id']}: final action must be submit_verdict.")
    if actions[-1].get("verdict") != scenario["truth_label"]:
        raise ValueError(f"{scenario['scenario_id']}: final verdict must match truth_label.")


def _visible_sources_for_scenario(scenario: dict[str, Any]) -> list[dict[str, str]]:
    visible_sources: list[dict[str, str]] = []
    for source in scenario["sources"][:3]:
        visible_sources.append(
            {
                "source_id": source["source_id"],
                "title": source["title"],
                "domain": source["domain"],
                "snippet": source["snippet"],
            }
        )
    return visible_sources


def _generate_trajectory(client: OpenAI, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _prompt_for_scenario(scenario)},
        ],
    )
    actions = json.loads(response.output_text)
    _validate_actions(scenario, actions)
    return actions


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to generate SFT trajectories.")

    args = _parse_args()
    client = OpenAI()
    scenarios = _load_scenarios(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w") as out_f:
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
                "visible_sources": _visible_sources_for_scenario(scenario),
                "actions": actions,
                "truth_label": scenario["truth_label"],
            }
            out_f.write(json.dumps(record, ensure_ascii=True) + "\n")
            written += 1

    print(f"Generated {written} trajectories -> {args.output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
