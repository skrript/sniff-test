"""
Before/after demo runner for pitch preparation.

Run this after training to compare a base model and a trained model on the same
difficulty tier, using the local SniffTest environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training.train import load_unsloth_model, run_episode


def demo_episode(model, tokenizer, env, task_level: str = "hard", label: str = "") -> None:
    print(f"\n{'=' * 60}")
    print(f"{label} - {task_level.upper()} scenario")
    print(f"{'=' * 60}")
    result = run_episode(model=model, tokenizer=tokenizer, task_level=task_level, env=env)
    print(f"Total reward: {result['total_reward']:.4f}")
    print(f"Reward components: {result.get('reward_components')}")
    print(f"Grade result: {result.get('grade_result')}")


def main(task_level: str, trained_dir: Path | None) -> None:
    from server.snifftest_environment import SniffTestEnvironment

    env = SniffTestEnvironment()

    base_model, base_tokenizer = load_unsloth_model()
    demo_episode(base_model, base_tokenizer, env, task_level=task_level, label="BASELINE")

    if trained_dir:
        # The finishing note prioritises merged-save output for inference.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        trained_tokenizer = AutoTokenizer.from_pretrained(trained_dir)
        trained_model = AutoModelForCausalLM.from_pretrained(
            trained_dir,
            torch_dtype="auto",
            device_map="auto",
        )
        demo_episode(
            trained_model,
            trained_tokenizer,
            env,
            task_level=task_level,
            label="TRAINED",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-level", default="hard")
    parser.add_argument(
        "--trained-dir",
        type=Path,
        default=None,
        help="Path to final_merged or another inference-ready checkpoint",
    )
    args = parser.parse_args()
    main(task_level=args.task_level, trained_dir=args.trained_dir)
