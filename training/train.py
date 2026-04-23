"""
SniffTest training entrypoint.

Follows docs/FINISHING_TOUCHES.md:
- Phase 1: SFT warm start on expert trajectories
- Phase 2: GRPO curriculum training (easy -> medium -> hard)
- Save with Unsloth's merged-save path, not merge_and_unload()

Examples:
    python3 training/train.py export-sft --output data/sft_chat_examples.jsonl
    python3 training/train.py train-sft
    python3 training/train.py train-grpo
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "data" / "claims_dataset.json"
SFT_TRAJECTORIES_PATH = ROOT / "data" / "sft_trajectories.jsonl"
DEFAULT_EXPORT_PATH = ROOT / "data" / "sft_chat_examples.jsonl"
DEFAULT_OUTPUT_DIR = Path(
    os.getenv("SNIFFTEST_TRAIN_OUTPUT_DIR", str(ROOT / "training_outputs"))
)

MODEL_NAME = os.getenv("SNIFFTEST_MODEL_NAME", "unsloth/Qwen2.5-1.5B-Instruct")
MAX_SEQ_LENGTH = int(os.getenv("SNIFFTEST_MAX_SEQ_LENGTH", "2048"))
LORA_RANK = int(os.getenv("SNIFFTEST_LORA_RANK", "16"))
MAX_TURNS = 10

SYSTEM_PROMPT = """You are an expert fact-checker investigating claims for accuracy.

Available actions (return as JSON, one action per response):
{"action_type": "search", "query": "your search query"}
{"action_type": "open_source", "source_id": "src_xxx"}
{"action_type": "cross_reference", "source_ids": ["src_xxx", "src_yyy"]}
{"action_type": "trace_origin", "source_id": "src_xxx"}
{"action_type": "check_metadata", "source_id": "src_xxx"}
{"action_type": "submit_verdict", "verdict": "true|false|misleading|unverifiable",
 "confidence": 0.0-1.0, "justification": "reasoning citing specific source IDs and domains"}

Investigation strategy:
1. Search with a targeted query to discover sources
2. Open the most relevant sources to read full content
3. Cross-reference conflicting sources
4. Check metadata on suspicious sources
5. Submit verdict with detailed justification

Return ONLY the JSON action. No other text."""

CURRICULUM = [
    {"task_level": "easy", "episodes": 300, "reward_threshold": 0.45},
    {"task_level": "medium", "episodes": 300, "reward_threshold": 0.30},
    {"task_level": "hard", "episodes": 200, "reward_threshold": None},
]

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _import_env_types():
    from models import InvestigateAction
    from server.snifftest_environment import SniffTestEnvironment

    return InvestigateAction, SniffTestEnvironment


def _load_inference_prompt() -> str:
    inference_path = ROOT / "inference.py"
    if not inference_path.exists():
        return SYSTEM_PROMPT

    spec = importlib.util.spec_from_file_location("snifftest_inference", inference_path)
    if spec is None or spec.loader is None:
        return SYSTEM_PROMPT

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return SYSTEM_PROMPT
    return getattr(module, "SYSTEM_PROMPT", SYSTEM_PROMPT)


def load_unsloth_model(
    model_name: str = MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LENGTH,
    lora_rank: int = LORA_RANK,
):
    """Load the QLoRA training model using Unsloth."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=TARGET_MODULES,
        lora_alpha=lora_rank * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_sft_trajectories(
    path: Path = SFT_TRAJECTORIES_PATH,
) -> list[dict[str, Any]]:
    """Load generated expert trajectories from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/generate_sft_data.py.")

    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _format_available_sources(visible: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- [{src['source_id']}] {src['title']} ({src['domain']}): {src['snippet']}"
        for src in visible
    )


def _make_sft_user_turn(
    record: dict[str, Any],
    prior_actions: list[dict[str, Any]],
) -> str:
    visible_sources = record.get("visible_sources", [])
    lines = [
        f"Investigate this claim: {record['claim']}",
        "",
        "Available sources:",
        _format_available_sources(visible_sources),
    ]
    if prior_actions:
        lines.extend(
            [
                "",
                "Previous expert actions:",
                *[
                    json.dumps(action, ensure_ascii=True, separators=(",", ":"))
                    for action in prior_actions
                ],
            ]
        )
    lines.extend(["", "Return the next best single JSON action."])
    return "\n".join(lines)


def build_sft_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand trajectories into per-step chat examples."""
    examples: list[dict[str, Any]] = []

    for record in records:
        visible_sources = record.get("visible_sources")
        if not isinstance(visible_sources, list) or len(visible_sources) < 3:
            continue

        actions = record.get("actions", [])
        for idx, action in enumerate(actions):
            prior_actions = actions[:idx]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _make_sft_user_turn(record, prior_actions),
                },
                {
                    "role": "assistant",
                    "content": json.dumps(action, ensure_ascii=True, separators=(",", ":")),
                },
            ]
            examples.append(
                {
                    "scenario_id": record["scenario_id"],
                    "difficulty": record["difficulty"],
                    "step_index": idx,
                    "messages": messages,
                }
            )

    return examples


def export_sft_examples(output_path: Path = DEFAULT_EXPORT_PATH) -> Path:
    """Write per-step chat-format SFT examples to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = build_sft_examples(load_sft_trajectories())

    with output_path.open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example) + "\n")

    print(f"Exported {len(examples)} examples -> {output_path}")
    return output_path


def train_sft(
    output_dir: Path = DEFAULT_OUTPUT_DIR / "sft_warm_start",
    epochs: int = 2,
    learning_rate: float = 2e-4,
    model_name: str = MODEL_NAME,
):
    """Warm-start the policy on expert trajectories before RL."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = load_unsloth_model(model_name=model_name)
    examples = build_sft_examples(load_sft_trajectories())

    rendered = []
    for example in examples:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        rendered.append({"text": text})

    dataset = Dataset.from_list(rendered)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        logging_steps=10,
        save_steps=50,
        bf16=True,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"SFT warm start saved -> {output_dir}")
    return model, tokenizer, trainer


def obs_to_text(obs: Any, step: int) -> str:
    """Render a SniffTest observation as a model user message."""
    lines = [
        f"Step {step}/{MAX_TURNS}  |  Steps remaining: {obs.steps_remaining}",
        "",
        f"CLAIM: {obs.claim}",
        "",
        f"AVAILABLE SOURCES ({len(obs.available_sources)} discovered):",
    ]
    for source in obs.available_sources:
        tag = " [READ]" if source.retrieved else ""
        lines.append(f"  [{source.source_id}] {source.title} ({source.domain}){tag}")
        lines.append(f"    {source.snippet}")
    if obs.opened_content:
        lines.extend(["", "LAST OPENED:", obs.opened_content[:500]])
    if obs.cross_reference_result:
        lines.extend(["", "CROSS-REFERENCE:", obs.cross_reference_result[:300]])
    if obs.trace_result:
        lines.extend(["", "TRACE RESULT:", obs.trace_result[:300]])
    if obs.metadata_result:
        lines.extend(["", "METADATA CHECK:", obs.metadata_result[:300]])
    if obs.action_history:
        lines.extend(["", "ACTION HISTORY (last 5):"])
        for history in obs.action_history[-5:]:
            lines.append(
                f"  Step {history.step} [{history.action_type}]: {history.result_summary}"
            )
    if obs.message:
        lines.extend(["", f"FEEDBACK: {obs.message}"])
    lines.extend(["", "Next action (JSON only):"])
    return "\n".join(lines)


def parse_action(text: str) -> tuple[dict[str, Any], bool]:
    """Extract a single JSON action from model output."""
    text = (text or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line), True
            except json.JSONDecodeError:
                continue

    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        return {"action_type": "search", "query": "claim evidence"}, False


def run_episode(
    model: Any,
    tokenizer: Any,
    task_level: str,
    env: Any,
    scenario_id: str | None = None,
) -> dict[str, Any]:
    """
    Run one full episode directly against the local environment.

    Used for sanity checks, post-training evaluation, and demo generation.
    """
    InvestigateAction, _ = _import_env_types()

    obs = env.reset(task_level=task_level, scenario_id=scenario_id)
    messages = [
        {"role": "system", "content": _load_inference_prompt()},
        {
            "role": "user",
            "content": (
                f"Investigate this claim: {obs.claim}\n\nAvailable sources:\n"
                + "\n".join(
                    f"- [{source.source_id}] {source.title} ({source.domain}): {source.snippet}"
                    for source in obs.available_sources
                )
            ),
        },
    ]

    prompts: list[str] = []
    completions: list[str] = []
    rewards: list[float] = []
    done = False
    total_reward = 0.0
    invalid_output_penalty = 0.0
    valid_json_steps = 0
    used_advanced_tool = False
    final_verdict = None

    for _ in range(MAX_TURNS):
        if done:
            break

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        import torch

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

        try:
            action_dict, valid = parse_action(completion)
            if not valid:
                reward = -0.1
                rewards.append(reward)
                total_reward += reward
                invalid_output_penalty += reward
                prompts.append(prompt)
                completions.append(completion)
                messages.append({"role": "assistant", "content": completion})
                messages.append(
                    {
                        "role": "user",
                        "content": "Invalid action format. Provide one valid JSON action.",
                    }
                )
                continue
            action = InvestigateAction(**action_dict)
        except Exception:
            reward = -0.1
            rewards.append(reward)
            total_reward += reward
            invalid_output_penalty += reward
            prompts.append(prompt)
            completions.append(completion)
            messages.append({"role": "assistant", "content": completion})
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid action schema. Provide one valid JSON action.",
                }
            )
            continue

        valid_json_steps += 1
        if action.action_type in {"cross_reference", "trace_origin", "check_metadata"}:
            used_advanced_tool = True
        if action.action_type == "submit_verdict":
            final_verdict = action.verdict

        obs = env.step(action)
        reward = float(obs.reward or 0.0)
        done = bool(obs.done)
        total_reward += reward

        prompts.append(prompt)
        completions.append(completion)
        rewards.append(reward)

        messages.append({"role": "assistant", "content": completion})
        messages.append({"role": "user", "content": obs.message or "Action completed."})

        if hasattr(run_episode, "_episode_count"):
            run_episode._episode_count += 1
        else:
            run_episode._episode_count = 1

        if run_episode._episode_count % 50 == 0:
            print(f"\n=== Episode {run_episode._episode_count} Sample ===")
            print(f"  Claim: {obs.claim[:80]}...")
            print(f"  Valid JSON: {valid}")
            print(f"  Step reward: {reward:.3f}")
            print(f"  Total so far: {total_reward:.3f}")

    return {
        "prompts": prompts,
        "completions": completions,
        "rewards": rewards,
        "total_reward": total_reward,
        "step_count": len(rewards),
        "valid_json_steps": valid_json_steps,
        "total_steps": len(completions),
        "json_valid_rate": valid_json_steps / max(len(completions), 1),
        "used_advanced_tool": used_advanced_tool,
        "final_verdict": final_verdict,
        "timed_out": final_verdict is None,
        "scenario_id": getattr(getattr(env, "_current_scenario", None), "scenario_id", None),
        "truth_label": getattr(getattr(env, "_current_scenario", None), "truth_label", None),
        "invalid_output_penalty": invalid_output_penalty,
        "reward_components": obs.reward_components if getattr(obs, "done", False) else None,
        "grade_result": getattr(env.state, "grade_result", None),
    }


def rollout_once(trainer: Any, env: Any, tokenizer: Any, task_level: str) -> dict[str, Any]:
    """Play one full SniffTest episode for GRPO rollout collection."""
    from trl.experimental.openenv import generate_rollout_completions

    InvestigateAction, _ = _import_env_types()

    obs = env.reset(task_level=task_level)

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    rewards: list[float] = []
    valid_actions = 0
    total_actions = 0
    total_reward = 0.0
    invalid_output_penalty = 0.0

    for turn in range(MAX_TURNS):
        if getattr(obs, "done", False):
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_to_text(obs, turn + 1)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        out = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(out["prompt_ids"])
        completion_ids.extend(out["completion_ids"])
        logprobs.extend(out["logprobs"])

        completion_text = out.get("text") or tokenizer.decode(
            out["completion_ids"], skip_special_tokens=True
        )
        action_dict, valid = parse_action(completion_text)
        total_actions += 1
        valid_actions += int(valid)

        if not valid:
            step_reward = -0.1
            rewards.append(step_reward)
            total_reward += step_reward
            invalid_output_penalty += step_reward
            continue

        try:
            action = InvestigateAction(**action_dict)
        except Exception:
            valid_actions = max(0, valid_actions - 1)
            step_reward = -0.1
            rewards.append(step_reward)
            total_reward += step_reward
            invalid_output_penalty += step_reward
            continue

        obs = env.step(action)
        step_reward = float(obs.reward or 0.0)
        rewards.append(step_reward)
        total_reward += step_reward

    grade = getattr(env.state, "grade_result", {}) or {}
    components = getattr(obs, "reward_components", None) or {}
    base_format_reward = float(components.get("format", valid_actions / max(total_actions, 1)))

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "accuracy_reward": float(components.get("accuracy", grade.get("accuracy", 0.0))),
        "evidence_reward": float(
            components.get("evidence", grade.get("evidence_alignment", 0.0))
        ),
        "reasoning_reward": float(
            components.get("reasoning", grade.get("reasoning_depth", 0.0))
        ),
        "efficiency_reward": float(
            components.get("efficiency", grade.get("efficiency", 0.0))
        ),
        "format_reward": base_format_reward + invalid_output_penalty,
        "anti_hack_reward": float(components.get("anti_hack", 0.0)),
        "total_reward": float(
            (sum(components.values()) if components else obs.reward or 0.0)
            + invalid_output_penalty
        ),
    }


def reward_accuracy(completions, **kwargs):
    rewards = kwargs.get("accuracy_reward")
    return [float(value) for value in rewards] if rewards else [0.0] * len(completions)


def reward_evidence(completions, **kwargs):
    rewards = kwargs.get("evidence_reward")
    return [float(value) for value in rewards] if rewards else [0.0] * len(completions)


def reward_reasoning(completions, **kwargs):
    rewards = kwargs.get("reasoning_reward")
    return [float(value) for value in rewards] if rewards else [0.0] * len(completions)


def reward_efficiency(completions, **kwargs):
    rewards = kwargs.get("efficiency_reward")
    return [float(value) for value in rewards] if rewards else [0.0] * len(completions)


def reward_format(completions, **kwargs):
    rewards = kwargs.get("format_reward")
    return [float(value) for value in rewards] if rewards else [0.0] * len(completions)


def reward_anti_hack(completions, **kwargs):
    rewards = kwargs.get("anti_hack_reward")
    return [float(value) for value in rewards] if rewards else [0.0] * len(completions)


def _build_rollout_func(tokenizer: Any, env: Any):
    def rollout_func(prompts, trainer=None):
        batch = {
            key: []
            for key in [
                "prompt_ids",
                "completion_ids",
                "logprobs",
                "accuracy_reward",
                "evidence_reward",
                "reasoning_reward",
                "efficiency_reward",
                "format_reward",
                "anti_hack_reward",
                "total_reward",
            ]
        }

        for task_level in prompts:
            level = (task_level or "easy").strip()
            if level not in {"easy", "medium", "hard"}:
                level = "easy"
            episode = rollout_once(trainer=trainer, env=env, tokenizer=tokenizer, task_level=level)
            for key in batch:
                batch[key].append(episode[key])

        return batch

    return rollout_func


def sanity_check() -> None:
    """Run a few manual episodes to confirm rewards and grading flow."""
    _, SniffTestEnvironment = _import_env_types()
    env = SniffTestEnvironment()

    print("=" * 60)
    print("ENVIRONMENT SANITY CHECK")
    print("=" * 60)
    for level in ["easy", "medium", "hard"]:
        obs = env.reset(task_level=level)
        print(f"\n[{level.upper()}] {obs.claim[:90]}...")
        print(f"Visible sources: {len(obs.available_sources)}")
        if obs.available_sources:
            source_id = obs.available_sources[0].source_id
            InvestigateAction, _ = _import_env_types()
            obs = env.step(InvestigateAction(action_type="search", query="evidence fact-check"))
            obs = env.step(InvestigateAction(action_type="open_source", source_id=source_id))
            obs = env.step(
                InvestigateAction(
                    action_type="submit_verdict",
                    verdict="false",
                    confidence=0.4,
                    justification=(
                        "Sanity check only. This is a placeholder verdict submitted "
                        "after a minimal three-step investigation."
                    ),
                )
            )
            print(f"Final reward: {float(obs.reward or 0.0):.4f}")
            print(f"Reward components: {obs.reward_components}")


def _make_stage_dataset(level: str, episodes: int):
    from datasets import Dataset

    return Dataset.from_dict({"prompt": [level] * episodes})


def _stage_grpo_config(output_dir: Path, episodes: int):
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=max(10, episodes // 2),
        bf16=True,
        report_to="none",
    )


def train_curriculum(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = MODEL_NAME,
):
    """
    Curriculum training: easy -> medium -> hard.
    Advances stage-by-stage and saves a checkpoint after each stage.
    """
    from trl import GRPOTrainer

    _, SniffTestEnvironment = _import_env_types()
    model, tokenizer = load_unsloth_model(model_name=model_name)
    env = SniffTestEnvironment()
    rollout_func = _build_rollout_func(tokenizer=tokenizer, env=env)

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []

    for stage in CURRICULUM:
        level = stage["task_level"]
        episodes = int(stage["episodes"])
        threshold = stage["reward_threshold"]
        stage_output_dir = output_dir / f"{level}_stage"

        print(f"\n{'=' * 60}")
        print(f"CURRICULUM STAGE: {level.upper()} ({episodes} episodes)")
        print(f"{'=' * 60}")

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                reward_accuracy,
                reward_evidence,
                reward_reasoning,
                reward_efficiency,
                reward_format,
                reward_anti_hack,
            ],
            train_dataset=_make_stage_dataset(level, episodes),
            args=_stage_grpo_config(stage_output_dir, episodes),
            rollout_func=rollout_func,
        )
        trainer.train()

        reward_entries = [
            item
            for item in trainer.state.log_history
            if "reward" in item and item.get("reward") is not None
        ]
        for idx, entry in enumerate(reward_entries, start=1):
            all_results.append(
                {
                    "level": level,
                    "episode": idx,
                    "reward": float(entry.get("reward", 0.0)),
                }
            )

        recent_rewards = [float(item.get("reward", 0.0)) for item in reward_entries[-10:]]
        recent_avg = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
        print(f"  Recent average reward: {recent_avg:.4f}")

        if threshold is not None and recent_avg < threshold:
            print(
                f"  Threshold {threshold:.2f} not reached. Keeping stage checkpoint anyway."
            )
        elif threshold is not None:
            print(f"  Threshold {threshold:.2f} reached.")

        model.save_pretrained(str(stage_output_dir))
        tokenizer.save_pretrained(str(stage_output_dir))
        print(f"  Checkpoint saved: {stage_output_dir}")

    return model, tokenizer, all_results


def save_final_model(model: Any, tokenizer: Any, output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """
    Correct save path for a 4-bit Unsloth model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final_merged"
    adapter_path = output_dir / "lora_adapters"

    print(f"\nSaving final merged model to {final_path}...")
    model.save_pretrained_merged(
        str(final_path),
        tokenizer,
        save_method="merged_16bit",
    )
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Merged model saved: {final_path}")
    print(f"LoRA adapters saved: {adapter_path}")


def write_reward_history(results: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "reward_history.json"
    history_path.write_text(json.dumps(results, indent=2))
    print(f"Reward history saved -> {history_path}")
    return history_path


def cli() -> None:
    parser = argparse.ArgumentParser(description="SniffTest training entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export-sft")
    export_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPORT_PATH,
        help="Path to write chat-format SFT examples",
    )

    sft_parser = subparsers.add_parser("train-sft")
    sft_parser.add_argument("--epochs", type=int, default=2)
    sft_parser.add_argument("--learning-rate", type=float, default=2e-4)
    sft_parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Base model name or local checkpoint path to fine-tune",
    )
    sft_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "sft_warm_start",
    )

    subparsers.add_parser("sanity-check")

    grpo_parser = subparsers.add_parser("train-grpo")
    grpo_parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Base model name or local checkpoint path to continue RL from",
    )
    grpo_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    args = parser.parse_args()

    if args.command == "export-sft":
        export_sft_examples(args.output)
    elif args.command == "train-sft":
        train_sft(
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            model_name=args.model_name,
        )
    elif args.command == "sanity-check":
        sanity_check()
    elif args.command == "train-grpo":
        model, tokenizer, results = train_curriculum(
            output_dir=args.output_dir,
            model_name=args.model_name,
        )
        save_final_model(model, tokenizer, output_dir=args.output_dir)
        write_reward_history(results, args.output_dir)


if __name__ == "__main__":
    cli()
