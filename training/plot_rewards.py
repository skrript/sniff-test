"""
Plot reward improvement across the easy/medium/hard curriculum.

Usage:
    python3 training/plot_rewards.py --input training_outputs/reward_history.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(input_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    results = json.loads(input_path.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"easy": "#22c55e", "medium": "#f59e0b", "hard": "#ef4444"}

    for ax, level in zip(axes, ["easy", "medium", "hard"]):
        episodes = [row for row in results if row["level"] == level]
        rewards = [row["reward"] for row in episodes]

        window = 20
        rolling = [
            sum(rewards[max(0, i - window) : i + 1]) / min(i + 1, window)
            for i in range(len(rewards))
        ]

        ax.plot(rewards, alpha=0.2, color=colors[level])
        ax.plot(rolling, color=colors[level], linewidth=2)
        ax.set_title(f"{level.capitalize()} Tasks", fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_ylim(0, 1)
        ax.axhline(
            y=rolling[0] if rolling else 0,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Baseline",
        )
        ax.legend()

    plt.suptitle(
        "SniffTest: Reward Improvement Across Curriculum",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("training_outputs/reward_history.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reward_curves.png"),
    )
    args = parser.parse_args()
    main(args.input, args.output)
