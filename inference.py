#!/usr/bin/env python3
"""
inference.py
------------
Optimised LLM agent for the cloud-alert-triage OpenEnv environment.

Strategy: plan-then-execute.
  Phase 1 — Single LLM call: send ALL pending alerts, get a complete ordered
             action plan as a JSON array (link_alerts first, then triage/skip).
  Phase 2 — Execute the plan step-by-step.  Any missed alerts are handled by
             the heuristic fallback before the episode closes.

This approach lets the model see the full picture before deciding, which
dramatically improves incident correlation (link_alerts F1) and root-cause
accuracy on cascading failures.

Environment variables:
    ENV_URL          URL of the running environment server
                     (default: http://localhost:7860)
    API_BASE_URL     OpenAI-compatible API base URL
                     (default: https://api.groq.com/openai/v1)
    MODEL_NAME       Model to use
                     (default: llama-3.3-70b-versatile)
    OPENAI_API_KEY   API key — for Groq use your Groq key here
    HF_TOKEN         Hugging Face token (fallback key)

Usage:
    # 1. Start the environment server
    python -m uvicorn server.app:app --port 7860

    # 2. Run the agent
    export OPENAI_API_KEY=gsk_...   # Groq key
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from itertools import combinations
from typing import Any

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
# Priority: GROQ_API_KEY for Groq, OPENAI_API_KEY for OpenAI, HF_TOKEN as fallback
_api_base = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
if "groq" in _api_base:
    API_KEY: str = os.environ.get("GROQ_API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
elif "openai.com" in _api_base:
    API_KEY: str = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")
else:
    API_KEY: str = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
del _api_base

TASKS: list[str] = ["easy", "medium", "hard"]
DEFAULT_SEED: int = 42
TOTAL_BUDGET_SECONDS: float = 20 * 60
PER_TASK_BUDGET_SECONDS: float = 6 * 60
LLM_TIMEOUT_SECONDS: float = 60.0   # longer for full-plan calls


# ---------------------------------------------------------------------------
# Structured logging (exact format required by evaluator)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] {json.dumps({'task': task, 'env': 'cloud-alert-triage', 'model': model})}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error: str | None) -> None:
    print(f"[STEP] {json.dumps({'step': step, 'action': action, 'reward': reward, 'done': done, 'error': error})}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    print(f"[END] {json.dumps({'success': success, 'steps': steps, 'score': round(max(0.0, min(1.0, score)), 4), 'rewards': [round(r, 4) for r in rewards]})}", flush=True)


# ---------------------------------------------------------------------------
# Heuristic fallback (Minimal)
# ---------------------------------------------------------------------------

def minimal_fallback_action(alert: dict) -> dict:
    """Minimal, intentionally weak fallback to prevent cheating without LLM."""
    return {"action_type": "skip", "alert_id": alert["alert_id"]}


# ---------------------------------------------------------------------------
# LLM planning
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert SRE triaging cloud infrastructure alerts.
Review the alert symptoms and service dependency graph to deduce root causes.
Respond with ONLY a valid JSON array of actions — no prose, no markdown fences.

== STRATEGY ==
1. Examine ALL alerts and the service dependency map before deciding anything.
2. Look for cascading failures. If multiple dependents alert, the root cause is likely an upstream service.
3. Group related alerts using link_alerts BEFORE triaging.
4. Separate true infrastructure issues from false alarms (e.g. maintenance, batch jobs).
5. Produce exactly one triage or skip action per alert.

== OUTPUT FORMAT ==
Return a JSON array. Order: link_alerts first, then triage/skip.
[
  {"action_type":"link_alerts","alert_ids":["alert-003","alert-007"],"incident_label":"redis-cascade"},
  {"action_type":"triage","alert_id":"alert-001","root_cause":"resource_exhaustion","severity":"high","remediation":"scale_up"},
  {"action_type":"skip","alert_id":"alert-005"}
]

Valid root_cause values: resource_exhaustion, network_failure, deployment_bug, config_error, dependency_outage.
Valid severity values: critical, high, medium, low.
Valid remediation values: scale_up, escalate_to_team, rollback_deploy, fix_config, acknowledge_and_monitor.
"""


def _fmt_alert(a: dict) -> str:
    ctx = f' | ctx: {a["context"][:120]}' if a.get("context") else ""
    return (
        f'{a["alert_id"]} [{a["service"]}] {a["metric"]}={a["metric_value"]}'
        f'(thr={a["threshold"]}) | {a["message"][:120]}{ctx}'
    )


def _fmt_map(svc_map: dict) -> str:
    return "\n".join(
        f"  {s} -> [{', '.join(d) or 'none'}]"
        for s, d in sorted(svc_map.items())
    )


def build_plan_prompt(obs: dict) -> str:
    alerts = obs.get("alerts", [])
    pending = [a for a in alerts if not a.get("triaged", False)]
    lines = [
        f"Task has {len(pending)} alerts to triage. Step budget: {obs.get('max_steps')}.",
        "",
        "=== ALERTS ===",
        *[_fmt_alert(a) for a in pending],
        "",
        "=== SERVICE DEPENDENCY MAP ===",
        _fmt_map(obs.get("service_map", {})),
        "",
        "Produce a complete JSON action array covering EVERY alert above.",
        "Put link_alerts actions first. Do NOT omit any alert.",
    ]
    return "\n".join(lines)


def get_full_plan(client: OpenAI, obs: dict) -> tuple[list[dict], str | None]:
    """
    Ask the LLM for a complete ordered action plan for all pending alerts.
    Returns (plan_list, error_or_None).
    """
    prompt = build_plan_prompt(obs)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            max_tokens=4096,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        raw = (resp.choices[0].message.content or "").strip()
        return parse_plan(raw), None
    except Exception as exc:
        return [], str(exc)


def parse_plan(text: str) -> list[dict]:
    """Strip markdown fences; parse JSON array. Returns [] on any failure."""
    cleaned = text.strip()
    # Strip ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            ln for ln in lines
            if not ln.strip().startswith("```")
        ).strip()
    # Find outermost [ ... ]
    start = cleaned.find("[")
    end   = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(cleaned[start:end + 1])
        if isinstance(data, list):
            return [a for a in data if isinstance(a, dict) and "action_type" in a]
    except (json.JSONDecodeError, ValueError):
        pass
    return []


# ---------------------------------------------------------------------------
# Coverage enforcement
# ---------------------------------------------------------------------------

def fill_missing(plan: list[dict], all_alerts: list[dict]) -> list[dict]:
    """
    Ensure every pending alert has exactly one triage/skip action in the plan.
    Append heuristic actions for any alert not covered by the plan.
    Do NOT add a second action for already-covered alerts.
    """
    covered: set[str] = set()
    for action in plan:
        if action.get("action_type") in ("triage", "skip"):
            covered.add(action.get("alert_id", ""))

    extras = []
    for alert in all_alerts:
        if not alert.get("triaged", False) and alert["alert_id"] not in covered:
            extras.append(minimal_fallback_action(alert))

    return plan + extras


def build_full_plan(client: OpenAI, obs: dict) -> list[dict]:
    """
    Build the complete action plan for this episode:
    1. Try LLM plan.
    2. If LLM fails or returns an empty plan, fall back to minimal skip actions.
    3. Fill any coverage gaps with fallbacks regardless.
    """
    pending = [a for a in obs.get("alerts", []) if not a.get("triaged", False)]

    llm_plan, llm_err = get_full_plan(client, obs)

    if llm_err or not llm_plan:
        # Full heuristic fallback
        return [minimal_fallback_action(a) for a in pending]

    # Ensure every alert is covered
    return fill_missing(llm_plan, pending)


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(http: httpx.Client, task_id: str, seed: int) -> dict:
    r = http.post("/reset", json={"task_id": task_id, "seed": seed})
    r.raise_for_status()
    data = r.json()
    return data.get("observation", data)


def env_step(http: httpx.Client, action: dict) -> dict:
    r = http.post("/step", json=action)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, llm: OpenAI, http: httpx.Client, deadline: float) -> None:
    obs = env_reset(http, task_id, DEFAULT_SEED)
    log_start(task_id, MODEL_NAME)

    # Build complete action plan in one shot
    plan = build_full_plan(llm, obs)

    rewards: list[float] = []
    done = False
    step_num = 0
    grader_score = 0.0

    for action in plan:
        if done:
            break
        if time.time() > deadline:
            log_end(False, step_num, grader_score, rewards)
            return

        error: str | None = None
        try:
            result = env_step(http, action)
        except Exception as exc:
            error = str(exc)
            log_step(step_num + 1, action, 0.0, False, error)
            break

        reward      = float(result.get("reward", 0.0))
        done        = bool(result.get("done", False))
        info        = result.get("info", {})
        obs         = result.get("observation", obs)
        step_num   += 1
        rewards.append(reward)

        if done:
            grader_score = float(info.get("grader_score", 0.0))

        log_step(step_num, action, reward, done, error)

    # If episode not yet done (plan covered all alerts but done wasn't triggered)
    # handle any remaining alerts that appeared after mid-episode resets
    if not done:
        pending = [a for a in obs.get("alerts", []) if not a.get("triaged", False)]
        for alert in pending:
            if done or time.time() > deadline:
                break
            action = minimal_fallback_action(alert)
            try:
                result = env_step(http, action)
            except Exception as exc:
                log_step(step_num + 1, action, 0.0, False, str(exc))
                break
            reward     = float(result.get("reward", 0.0))
            done       = bool(result.get("done", False))
            obs        = result.get("observation", obs)
            step_num  += 1
            rewards.append(reward)
            if done:
                grader_score = float(result.get("info", {}).get("grader_score", 0.0))
            log_step(step_num, action, reward, done, None)

    log_end(done, step_num, grader_score, rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("[WARN] No HF_TOKEN or OPENAI_API_KEY found — LLM calls will fail.", file=sys.stderr)

    llm  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "placeholder")
    http = httpx.Client(base_url=ENV_URL, timeout=30.0)

    global_deadline = time.time() + TOTAL_BUDGET_SECONDS

    for task_id in TASKS:
        if time.time() > global_deadline:
            print("[WARN] Global budget exceeded — skipping remaining tasks.", file=sys.stderr)
            break
        task_deadline = min(time.time() + PER_TASK_BUDGET_SECONDS, global_deadline)
        try:
            run_task(task_id, llm, http, task_deadline)
        except Exception as exc:
            print(f"[ERROR] Task '{task_id}' crashed: {exc}", file=sys.stderr)
            log_end(False, 0, 0.0, [])


if __name__ == "__main__":
    main()
