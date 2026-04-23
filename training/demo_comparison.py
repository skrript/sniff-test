"""
Evaluate baseline, post-SFT, and post-RL models on a fixed test dataset.

This is the evaluation script for the demo story:
- SFT should improve json_valid_rate
- GRPO should improve verdict_accuracy and efficiency

Usage:
    python3 training/demo_comparison.py \
      --post-sft-dir training_outputs/sft_warm_start \
      --post-rl-dir training_outputs/final_merged
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train import load_unsloth_model, run_episode

DEFAULT_TEST_DATASET = ROOT / "data" / "test_dataset.json"
DEFAULT_OUTPUT_PATH = ROOT / "training_outputs" / "test_eval_metrics.json"


def _load_checkpoint(checkpoint_dir: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, tokenizer


def _load_test_scenarios(dataset_path: Path) -> list[dict[str, Any]]:
    scenarios = json.loads(dataset_path.read_text())
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"{dataset_path} must contain a non-empty array of scenarios.")
    return scenarios


def _episode_summary(result: dict[str, Any]) -> dict[str, Any]:
    grade = result.get("grade_result") or {}
    return {
        "scenario_id": result.get("scenario_id"),
        "truth_label": result.get("truth_label"),
        "final_verdict": result.get("final_verdict"),
        "json_valid_rate": round(float(result.get("json_valid_rate", 0.0)), 4),
        "step_count": int(result.get("step_count", 0)),
        "used_advanced_tool": bool(result.get("used_advanced_tool", False)),
        "verdict_correct": bool(grade.get("accuracy", 0.0) == 1.0),
        "total_reward": round(float(result.get("total_reward", 0.0)), 4),
        "timed_out": bool(result.get("timed_out", False)),
    }


def _aggregate(label: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    episode_count = len(episodes)
    if episode_count == 0:
        raise ValueError(f"{label}: no episodes to aggregate.")

    json_valid_steps = sum(ep["valid_json_steps"] for ep in episodes)
    total_steps = sum(ep["total_steps"] for ep in episodes)
    verdict_correct = sum(
        1 for ep in episodes if (ep.get("grade_result") or {}).get("accuracy", 0.0) == 1.0
    )
    avg_steps_to_verdict = sum(ep["step_count"] for ep in episodes) / episode_count
    tool_use_rate = sum(1 for ep in episodes if ep["used_advanced_tool"]) / episode_count

    return {
        "model_label": label,
        "episodes": episode_count,
        "json_valid_rate": round(json_valid_steps / max(total_steps, 1), 4),
        "verdict_accuracy": round(verdict_correct / episode_count, 4),
        "avg_steps_to_verdict": round(avg_steps_to_verdict, 4),
        "tool_use_rate": round(tool_use_rate, 4),
        "avg_total_reward": round(
            sum(float(ep.get("total_reward", 0.0)) for ep in episodes) / episode_count,
            4,
        ),
    }


def _evaluate_model(
    label: str,
    model: Any,
    tokenizer: Any,
    dataset_path: Path,
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    from server.snifftest_environment import SniffTestEnvironment

    env = SniffTestEnvironment(dataset_path=dataset_path, enable_adversarial=False)
    episodes: list[dict[str, Any]] = []

    print(f"\nEvaluating {label} on {len(scenarios)} test scenarios...")
    for idx, scenario in enumerate(scenarios, start=1):
        print(
            f"  [{idx:02d}/{len(scenarios):02d}] "
            f"{scenario['scenario_id']} ({scenario['difficulty']})"
        )
        result = run_episode(
            model=model,
            tokenizer=tokenizer,
            task_level=scenario["difficulty"],
            env=env,
            scenario_id=scenario["scenario_id"],
        )
        episodes.append(result)

    summary = _aggregate(label, episodes)
    print(
        f"  json_valid_rate={summary['json_valid_rate']:.4f}  "
        f"verdict_accuracy={summary['verdict_accuracy']:.4f}  "
        f"avg_steps_to_verdict={summary['avg_steps_to_verdict']:.2f}  "
        f"tool_use_rate={summary['tool_use_rate']:.4f}"
    )
    return {
        "summary": summary,
        "episodes": [_episode_summary(ep) for ep in episodes],
    }


def _add_model_spec(specs: list[tuple[str, Path | None]], label: str, path: Path | None) -> None:
    if path is not None:
        specs.append((label, path))


def main(
    dataset_path: Path,
    output_path: Path,
    post_sft_dir: Path | None,
    post_rl_dir: Path | None,
) -> None:
    scenarios = _load_test_scenarios(dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "scenario_count": len(scenarios),
        "models": [],
    }

    base_model, base_tokenizer = load_unsloth_model()
    results["models"].append(
        _evaluate_model(
            label="baseline",
            model=base_model,
            tokenizer=base_tokenizer,
            dataset_path=dataset_path,
            scenarios=scenarios,
        )
    )

    extra_models: list[tuple[str, Path | None]] = []
    _add_model_spec(extra_models, "post_sft", post_sft_dir)
    _add_model_spec(extra_models, "post_rl", post_rl_dir)

    for label, checkpoint_dir in extra_models:
        model, tokenizer = _load_checkpoint(checkpoint_dir)
        results["models"].append(
            _evaluate_model(
                label=label,
                model=model,
                tokenizer=tokenizer,
                dataset_path=dataset_path,
                scenarios=scenarios,
            )
        )

    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved test-set evaluation -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TEST_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--post-sft-dir",
        type=Path,
        default=None,
        help="Path to an inference-ready post-SFT checkpoint",
    )
    parser.add_argument(
        "--post-rl-dir",
        type=Path,
        default=None,
        help="Path to an inference-ready post-RL checkpoint",
    )
    args = parser.parse_args()
    main(
        dataset_path=args.dataset,
        output_path=args.output,
        post_sft_dir=args.post_sft_dir,
        post_rl_dir=args.post_rl_dir,
    )
