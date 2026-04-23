"""
Plot curriculum reward history alongside fixed test-set evaluation metrics.

Usage:
    python3 training/plot_rewards.py \
      --reward-input training_outputs/reward_history.json \
      --eval-input training_outputs/test_eval_metrics.json \
      --output training_outputs/training_and_eval_report.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _plot_reward_history(ax, results: list[dict[str, Any]], level: str, color: str) -> None:
    episodes = [row for row in results if row["level"] == level]
    rewards = [row["reward"] for row in episodes]

    window = 20
    rolling = [
        sum(rewards[max(0, i - window) : i + 1]) / min(i + 1, window)
        for i in range(len(rewards))
    ]

    ax.plot(rewards, alpha=0.2, color=color)
    ax.plot(rolling, color=color, linewidth=2)
    ax.set_title(f"{level.capitalize()} Tasks", fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_ylim(0, 1)
    ax.axhline(
        y=rolling[0] if rolling else 0,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Start",
    )
    ax.legend()


def _plot_eval_bars(ax, metrics: list[dict[str, Any]], metric_key: str, title: str) -> None:
    labels = [row["model_label"] for row in metrics]
    values = [row[metric_key] for row in metrics]
    colors = {
        "baseline": "#6b7280",
        "post_sft": "#2563eb",
        "post_rl": "#16a34a",
    }
    bar_colors = [colors.get(label, "#6b7280") for label in labels]
    bars = ax.bar(labels, values, color=bar_colors, width=0.6)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_key.replace("_", " "))

    if metric_key in {"json_valid_rate", "verdict_accuracy", "tool_use_rate"}:
        ax.set_ylim(0, 1)
    else:
        upper = max(values) if values else 1
        ax.set_ylim(0, max(upper * 1.2, 1))

    for bar, value in zip(bars, values):
        label = f"{value:.3f}" if isinstance(value, float) else str(value)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _maybe_get_eval_summaries(eval_results: dict[str, Any]) -> list[dict[str, Any]]:
    models = eval_results.get("models", [])
    summaries: list[dict[str, Any]] = []
    for model in models:
        summary = model.get("summary")
        if isinstance(summary, dict):
            summaries.append(summary)
    return summaries


def main(reward_input: Path, eval_input: Path | None, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to plot SniffTest reports. "
            "Install it in the training environment before running training/plot_rewards.py."
        ) from exc

    rewards = _load_json(reward_input)
    eval_summaries = _maybe_get_eval_summaries(_load_json(eval_input)) if eval_input and eval_input.exists() else []

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.4, 1.0])

    reward_axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    colors = {"easy": "#22c55e", "medium": "#f59e0b", "hard": "#ef4444"}
    for ax, level in zip(reward_axes, ["easy", "medium", "hard"]):
        _plot_reward_history(ax, rewards, level, colors[level])

    summary_ax = fig.add_subplot(grid[0, 3])
    summary_ax.axis("off")
    total_episodes = len(rewards)
    avg_reward = sum(row["reward"] for row in rewards) / max(total_episodes, 1)
    by_level = {
        level: len([row for row in rewards if row["level"] == level])
        for level in ["easy", "medium", "hard"]
    }
    summary_text = [
        "Training Summary",
        f"Episodes: {total_episodes}",
        f"Avg reward: {avg_reward:.3f}",
        f"Easy episodes: {by_level['easy']}",
        f"Medium episodes: {by_level['medium']}",
        f"Hard episodes: {by_level['hard']}",
    ]
    if eval_summaries:
        summary_text.extend(
            [
                "",
                "Test-set eval loaded:",
                f"Models: {', '.join(row['model_label'] for row in eval_summaries)}",
            ]
        )
    summary_ax.text(0.0, 1.0, "\n".join(summary_text), va="top", fontsize=11)

    eval_metric_specs = [
        ("json_valid_rate", "JSON Valid Rate"),
        ("verdict_accuracy", "Verdict Accuracy"),
        ("avg_steps_to_verdict", "Avg Steps To Verdict"),
        ("tool_use_rate", "Tool Use Rate"),
    ]
    eval_axes = [fig.add_subplot(grid[1, idx]) for idx in range(4)]
    for ax, (metric_key, title) in zip(eval_axes, eval_metric_specs):
        if eval_summaries:
            _plot_eval_bars(ax, eval_summaries, metric_key, title)
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No test-set eval file provided", ha="center", va="center")

    fig.suptitle(
        "SniffTest Training Curve And Fixed Test-Set Evaluation",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward-input",
        type=Path,
        default=Path("training_outputs/reward_history.json"),
    )
    parser.add_argument(
        "--eval-input",
        type=Path,
        default=Path("training_outputs/test_eval_metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_outputs/training_and_eval_report.png"),
    )
    args = parser.parse_args()
    main(
        reward_input=args.reward_input,
        eval_input=args.eval_input,
        output_path=args.output,
    )
