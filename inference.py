"""
SniffTest Baseline Inference Script
====================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the project root.
- Uses OpenAI Client for all LLM calls.

STDOUT FORMAT
- Emits exactly three line types per episode:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Usage:
    # Against a running local server:
    python inference.py --base-url http://localhost:8000

    # Against a deployed HF Space:
    python inference.py --base-url https://your-space.hf.space
"""

import argparse
import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

try:
    from snifftest_env import InvestigateAction, SniffTestEnv, SniffTestObservation
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from snifftest_env import InvestigateAction, SniffTestEnv, SniffTestObservation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK: str = "snifftest_env"

MAX_STEPS: int = 10
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 400
SUCCESS_SCORE_THRESHOLD: float = 0.5

TASK_LEVELS = ["easy", "medium", "hard"]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert fact-checker investigating a claim. Use available tools methodically.

    Available actions (respond with a single JSON object on one line):
    {"action_type": "search", "query": "<search terms>"}
    {"action_type": "open_source", "source_id": "<src_id>"}
    {"action_type": "cross_reference", "source_ids": ["<src_id_1>", "<src_id_2>"]}
    {"action_type": "trace_origin", "source_id": "<src_id>"}
    {"action_type": "check_metadata", "source_id": "<src_id>"}
    {"action_type": "submit_verdict", "verdict": "true|false|misleading|unverifiable",
     "confidence": 0.0-1.0, "justification": "reasoning citing source IDs and domains"}

    Investigation strategy:
    1. Search for relevant sources on the claim topic
    2. Open the most relevant-looking sources to read full content
    3. Cross-reference suspicious or contradictory sources
    4. Check metadata on any low-credibility-looking sources
    5. Use trace_origin if the claim's propagation seems suspicious
    6. Submit a well-justified verdict citing specific source IDs

    Respond with ONLY the JSON action — no prose, no markdown fences.
    """
).strip()


# ---------------------------------------------------------------------------
# Logging helpers (strict format required by spec)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def _parse_action(text: str) -> dict:
    """Extract JSON from the model response, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    # Take only first JSON object if multiple lines
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return json.loads(text)


def _obs_to_user_message(obs: SniffTestObservation, step: int) -> str:
    """Render the observation as a user-turn message for the model."""
    sources_block = "\n".join(
        f"  [{s.source_id}] {s.title} ({s.domain})"
        + (" [READ]" if s.retrieved else "")
        + f"\n    {s.snippet}"
        for s in obs.available_sources
    )
    history_block = "\n".join(
        f"  Step {h.step} [{h.action_type}]: {h.result_summary}"
        for h in obs.action_history[-5:]
    )

    parts = [
        f"Step {step}/{MAX_STEPS}  |  Steps remaining: {obs.steps_remaining}",
        f"\nCLAIM: {obs.claim}",
        f"\nAVAILABLE SOURCES ({len(obs.available_sources)} discovered):\n{sources_block}",
    ]
    if obs.opened_content:
        parts.append(f"\nLAST OPENED:\n{obs.opened_content[:600]}")
    if obs.cross_reference_result:
        parts.append(f"\nCROSS-REFERENCE RESULT:\n{obs.cross_reference_result[:400]}")
    if obs.trace_result:
        parts.append(f"\nTRACE ORIGIN RESULT:\n{obs.trace_result[:400]}")
    if obs.metadata_result:
        parts.append(f"\nMETADATA CHECK:\n{obs.metadata_result[:400]}")
    if obs.action_history:
        parts.append(f"\nACTION HISTORY (last 5):\n{history_block}")
    if obs.message:
        parts.append(f"\nFEEDBACK: {obs.message}")
    parts.append("\nNext action (JSON only):")
    return "\n".join(parts)


def get_model_action(
    client: OpenAI,
    conversation: list,
    obs: SniffTestObservation,
    step: int,
) -> tuple:
    """Call the LLM and return (action_dict, error_str)."""
    user_msg = _obs_to_user_message(obs, step)
    conversation.append({"role": "user", "content": user_msg})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=conversation,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        conversation.append({"role": "assistant", "content": raw})
        action_dict = _parse_action(raw)
        return action_dict, None
    except Exception as exc:
        fallback = {"action_type": "search", "query": "evidence"}
        conversation.append({"role": "assistant", "content": json.dumps(fallback)})
        return fallback, str(exc)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


async def run_episode(
    env: SniffTestEnv,
    client: OpenAI,
    task_level: str,
) -> dict:
    """Run one full episode and return results dict."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error: Optional[str] = None

    log_start(task=task_level, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = await env.reset(task_level=task_level)
        obs: SniffTestObservation = reset_result.observation

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            action_dict, action_error = get_model_action(
                client, conversation, obs, step
            )
            if action_error:
                error = action_error

            # Validate action
            try:
                action = InvestigateAction(**action_dict)
            except Exception as ve:
                error = str(ve)
                action = InvestigateAction(action_type="search", query="claim evidence")
                action_dict = action.model_dump(exclude_none=True, exclude={"metadata"})

            step_result = await env.step(action)
            obs = step_result.observation
            reward = float(step_result.reward or 0.0)
            done = step_result.done

            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(action_dict, separators=(",", ":"))
            log_step(
                step=step, action=action_str, reward=reward, done=done, error=error
            )
            error = None  # reset per-step error

            if done:
                break

        # Compute normalised score from grader final_score embedded in last reward
        # (submit_verdict reward = grader.final_score in [0,1])
        # For timeout episodes, last reward is -0.5; use 0.0 for score
        final_reward = rewards[-1] if rewards else 0.0
        score = max(0.0, min(1.0, final_reward)) if steps_taken > 0 else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        error = str(exc)
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {
        "task_level": task_level,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main(base_url: str, runs_per_task: int) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy")

    summary: dict = {level: [] for level in TASK_LEVELS}

    async with SniffTestEnv(base_url=base_url) as env:
        for level in TASK_LEVELS:
            for run in range(runs_per_task):
                print(
                    f"[DEBUG] Running task={level} run={run + 1}/{runs_per_task}",
                    flush=True,
                )
                result = await run_episode(env, client, level)
                summary[level].append(result["score"])

    print("\n=== BASELINE RESULTS ===", flush=True)
    for level, scores in summary.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(
                f"{level.upper():8s}: avg_score={avg:.4f} over {len(scores)} run(s)",
                flush=True,
            )


def cli() -> None:
    parser = argparse.ArgumentParser(description="SniffTest baseline inference script")
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", "http://localhost:8000"),
        help="Base URL of the running SniffTest environment server",
    )
    parser.add_argument(
        "--runs-per-task",
        type=int,
        default=1,
        help="Number of episodes to run per difficulty level (default: 1)",
    )
    args = parser.parse_args()
    asyncio.run(main(base_url=args.base_url, runs_per_task=args.runs_per_task))


if __name__ == "__main__":
    cli()
