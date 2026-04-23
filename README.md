---
title: SniffTest — Misinformation Investigation Gym
emoji: 👃
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
  - fact-checking
  - information-integrity
  - misinformation
  - agentic
pinned: false
---

# 👃 SniffTest

> A misinformation investigation gym — because some claims just don't pass the sniff test.

SniffTest is an **OpenEnv-compliant RL environment** where an LLM agent investigates claims,
gathers evidence through tool calls, and submits a verdict against a deterministic hidden world
state. Three difficulty tiers: **🟢 Fresh** (easy), **🟡 Stale** (medium), **🔴 Rotten** (hard).

The environment uniquely features an **adversarial scenario generator** that tracks agent
weakness patterns across episodes and uses OpenAI to generate targeted scenarios that exploit
confirmed failure modes — a self-improving curriculum observable in real time via the `/state`
endpoint.

---

## Environment Overview & Motivation

**Real-world task:** Fact-checking and information integrity assessment is a genuine,
high-stakes human task. LLMs are confidently wrong about contested facts. SniffTest measures and
trains *epistemic humility under adversarial information conditions*.

**Partial observability:** The agent starts with only 3 sources visible (the highest-reliability
ones). Additional sources are unlocked via search — forcing strategic information gathering.

**Hidden ground truth:** Source reliability scores are never exposed in the observation.
The agent must infer credibility from content quality, metadata, cross-referencing, and
propagation signals.

**Dense rewards:** Each investigative action earns a shaped reward signal throughout the
trajectory (not just at the terminal step), teaching good investigative practice.

---

## Action Space

| `action_type` | Required Fields | Description |
|---|---|---|
| `search` | `query: str` | Search for relevant sources. Poor queries return fewer results. |
| `open_source` | `source_id: str` | Read the full content of a source (hidden until opened). |
| `cross_reference` | `source_ids: [str, str]` | Compare two sources for contradictions. |
| `trace_origin` | `source_id: str` | View the propagation chain — how the claim spread. |
| `check_metadata` | `source_id: str` | Inspect credibility signals: domain, author, publish date, flags. |
| `submit_verdict` | `verdict`, `confidence`, `justification` | Final verdict — ends the episode. |

**Verdict labels:** `"true"` · `"false"` · `"misleading"` · `"unverifiable"`

**Example action (JSON):**
```json
{"action_type": "submit_verdict", "verdict": "false", "confidence": 0.92,
 "justification": "src_001_b reveals the Valdris Institute is not an accredited research body."}
```

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `claim` | `str` | The claim under investigation |
| `available_sources` | `List[SourceSummary]` | Sources discovered so far (grows via search) |
| `opened_content` | `Optional[str]` | Full text of the most recently opened source |
| `cross_reference_result` | `Optional[str]` | Result of the most recent cross_reference |
| `trace_result` | `Optional[str]` | Result of the most recent trace_origin |
| `metadata_result` | `Optional[str]` | Result of the most recent check_metadata |
| `action_history` | `List[ActionLog]` | Ordered log of all actions this episode |
| `step_count` | `int` | Steps taken so far |
| `steps_remaining` | `int` | Steps left before timeout penalty |
| `message` | `str` | Thematic feedback (e.g. "💀 Foul odor detected") |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward signal for the last action |

**SourceSummary fields:** `source_id`, `title`, `domain`, `snippet`, `retrieved`

**Note:** `reliability_score` is intentionally hidden — the agent must infer credibility.

---

## Task Descriptions

| Task | Code | Display | Scenario Characteristics | Expected Score |
|---|---|---|---|---|
| Task 1 | `easy` | 🟢 Fresh | Clear signal, high-quality sources, obvious verdict after 1-2 source reads | 0.70–0.90 |
| Task 2 | `medium` | 🟡 Stale | Conflicting sources, partial truths, requires cross-referencing 2+ sources | 0.45–0.70 |
| Task 3 | `hard` | 🔴 Rotten | Coordinated misinformation, noisy sources, fabricated institutions, requires trace + metadata | 0.20–0.50 |

Pass `task_level` in the reset request body to select a difficulty:
```json
{"task_level": "hard"}
```

---

## Reward Function

### Step Rewards (dense signal throughout trajectory)

| Action | Reward | Notes |
|---|---|---|
| `search` | +0.05 | Flat reward for gathering information |
| `open_source` (key evidence) | +0.30 | Strong reward for finding decisive sources |
| `open_source` (non-key) | +0.05 | Small reward for exploration |
| `open_source` (duplicate) | -0.10 | Penalty for redundant action |
| `cross_reference` (key overlap) | +0.20 | Reward for comparing key sources |
| `cross_reference` (other) | +0.05 | Small reward |
| `trace_origin` | +0.15 | Good investigative practice |
| `check_metadata` (key source) | +0.15 | Reward for checking key source metadata |
| `check_metadata` (other) | +0.05 | Small reward |
| Repeated action type (3x) | -0.05 | Penalty for lazy loops |

### Terminal Reward (on `submit_verdict`)

Replaces the step reward entirely. Weighted score from TaskGrader:

| Dimension | Weight | Metric |
|---|---|---|
| **Accuracy** | 50% | Correct verdict label (0 or 1) |
| **Evidence alignment** | 25% | Fraction of key evidence sources opened before verdict |
| **Reasoning depth** | 15% | Source IDs / domains cited in justification |
| **Efficiency** | 10% | Continuous score: 1.0 at ≤ 3 steps, 0.0 at 10 steps (linear decay) |

### Timeout Penalty
If `max_steps` (10) is reached without `submit_verdict`: **-0.5** penalty, `done=True`.

---

## Dataset

The dataset (`data/claims_dataset.json`) contains claim investigation scenarios with:
- 4–6 sources per scenario (mix of reliable and unreliable)
- Hidden reliability scores (0.0–1.0) per source
- Propagation chains showing how claims spread
- Key evidence source IDs for grader evaluation
- Grader notes ending with smell-test verdict suffix

**Structure:**
```json
[{
  "scenario_id": "scenario_001",
  "difficulty": "easy|medium|hard",
  "claim": "...",
  "truth_label": "true|false|misleading|unverifiable",
  "manipulation_type": null | "fabricated" | "cherry_picked" | "outdated_context" | ...,
  "sources": [{
    "source_id": "src_001_a",
    "title": "...", "domain": "...", "snippet": "...",
    "full_content": "...",   // only revealed via open_source
    "reliability_score": 0.94,  // hidden from agent
    ...
  }],
  "propagation_chain": [...],
  "key_evidence_source_ids": ["src_001_a", "src_001_b"],
  "grader_notes": "... Smell test verdict: ROTTEN."
}]
```

---

## Setup & Usage

### Local Development

```bash
# 1. Clone and install
cd snifftest_env
uv sync   # or: pip install -e .

# 2. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# 3. Validate with OpenEnv CLI
openenv validate

# 4. Run baseline inference (requires a running server)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
python inference.py --base-url http://localhost:8000 --runs-per-task 3
```

### Docker

```bash
# Build (from snifftest_env/ directory)
docker build -t snifftest-env:latest .

# Run
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e HF_TOKEN=hf_... \
  snifftest-env:latest
```

### Training

For Hugging Face GPU compute or any other CUDA-capable training machine, install
the training stack with the optional extra:

```bash
uv sync --extra train
```

The training entrypoint is [training/train.py](/Users/kevin/Desktop/snifftest_env/training/train.py).
Training code and default training outputs are intentionally excluded from the
runtime Docker image.

---

## Adversarial Mode

After 5+ episodes, SniffTest's `WeaknessTracker` analyses agent performance and, when
weaknesses are confirmed, calls OpenAI `gpt-5-mini` to generate a batch of 5 targeted scenarios
that exploit those specific failure modes.

Observable via the `/state` endpoint:
```json
{
  "is_adversarial_episode": true,
  "confirmed_weaknesses": ["manipulation_types", "underused_tools"],
  "episodes_completed": 8,
  "adversarial_cache_size": 3
}
```

Graceful fallback: if `OPENAI_API_KEY` is not set, the environment silently uses
the static dataset. No crashes, no degraded functionality.

---

## Baseline Performance Scores

| Task | Difficulty | Model | Avg Score | Runs |
|---|---|---|---|---|
| Task 1 | 🟢 Fresh (easy) | Qwen2.5-72B-Instruct | 0.5786 | 3 |
| Task 2 | 🟡 Stale (medium) | Qwen2.5-72B-Instruct | 0.3000 | 3 |
| Task 3 | 🔴 Rotten (hard) | Qwen2.5-72B-Instruct | 0.2679 | 3 |

---

## Project Structure

```
snifftest_env/
├── README.md                    # This file (HF Space frontmatter at top)
├── openenv.yaml                 # OpenEnv manifest
├── pyproject.toml               # Project metadata and dependencies
├── uv.lock                      # Locked dependencies
├── Dockerfile                   # Root-level Dockerfile (full project context)
├── inference.py                 # Baseline inference script (required by spec)
├── __init__.py                  # Package exports
├── models.py                    # InvestigateAction + SniffTestObservation
├── client.py                    # SniffTestEnv(EnvClient)
├── data/
│   └── claims_dataset.json      # Static scenario dataset
├── scripts/
│   └── generate_dataset.py
├── outputs/
│   ├── logs/
│   └── evals/
└── server/
    ├── __init__.py
    ├── app.py                   # FastAPI application
    ├── snifftest_environment.py # Core environment logic
    ├── world_state.py           # ClaimScenario + SourceRecord models
    ├── tools.py                 # ToolEngine (deterministic lookups)
    ├── reward.py                # RewardEngine (dense step rewards)
    ├── grader.py                # TaskGrader (deterministic, LLM-free)
    ├── adversarial.py           # WeaknessTracker + AdversarialGenerator
    ├── requirements.txt
    └── Dockerfile               # Server Dockerfile (used by openenv build/push)
```
