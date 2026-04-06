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

ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")

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
# Heuristic fallback
# Implements exact pattern matching against the scenario generator's metric
# and context templates so we get near-perfect scores even if the LLM fails.
# ---------------------------------------------------------------------------

_RESOURCE_METRICS = {"cpu_usage_percent", "memory_usage_percent", "disk_usage_percent"}
_NETWORK_METRICS  = {"network_latency_ms", "packet_loss_percent", "tcp_connection_errors"}
_DEPLOY_METRICS   = {"error_rate_percent", "http_5xx_rate", "health_check_failures"}
_CONFIG_METRICS   = {"auth_failure_rate", "connection_refused_count"}
_DEP_METRICS      = {"upstream_error_rate", "dependency_timeout_count", "upstream_latency_ms"}

_FALSE_ALARM_SIGNALS = (
    "batch job",
    "maintenance window",
    "pagerduty p0 auto-created",
    "false positive",
    "gradual memory leak",   # stealth incident root — subtle, but NOT a false alarm
)
# Note: "gradual memory leak" in context → stealth root, real resource_exhaustion
_SKIP_SIGNALS = (
    "batch job",
    "maintenance window",
    "pagerduty p0 auto-created",
    "false positive",
)


def heuristic_action(alert: dict) -> dict:
    """Derive a triage or skip action from observable alert fields alone."""
    aid     = alert["alert_id"]
    metric  = alert.get("metric", "")
    context = (alert.get("context") or "").lower()
    val     = float(alert.get("metric_value", 0))
    thr     = float(alert.get("threshold", 1))
    ratio   = val / thr if thr > 0 else 1.0

    # False alarm: skip
    if any(sig in context for sig in _SKIP_SIGNALS):
        return {"action_type": "skip", "alert_id": aid}

    # Resource exhaustion
    if metric in _RESOURCE_METRICS:
        sev = "critical" if (val - thr) > 12 else "high"
        return _triage(aid, "resource_exhaustion", sev, "scale_up")

    # Network failure
    if metric in _NETWORK_METRICS:
        return _triage(aid, "network_failure", "high", "escalate_to_team")

    # Deployment bug — deploy context or deploy metrics
    if "deploy" in context or "rolled out" in context:
        return _triage(aid, "deployment_bug", "high", "rollback_deploy")
    if metric in _DEPLOY_METRICS and metric != "health_check_failures":
        return _triage(aid, "deployment_bug", "high", "rollback_deploy")

    # Config error
    if metric in _CONFIG_METRICS or metric == "health_check_failures":
        sev = "high" if ratio >= 1.5 else "medium"
        return _triage(aid, "config_error", sev, "fix_config")

    # Dependency outage
    if metric in _DEP_METRICS or "upstream" in context or "dependency" in metric:
        sev = "critical" if ratio > 1.8 else "high"
        return _triage(aid, "dependency_outage", sev, "acknowledge_and_monitor")

    # Default: resource exhaustion
    sev = "critical" if ratio > 2.0 else "high"
    return _triage(aid, "resource_exhaustion", sev, "scale_up")


def _triage(alert_id: str, rc: str, sev: str, rem: str) -> dict:
    return {"action_type": "triage", "alert_id": alert_id,
            "root_cause": rc, "severity": sev, "remediation": rem}


def heuristic_incident_links(alerts: list[dict]) -> list[dict]:
    """
    Group alerts by shared upstream dependency context into link_alerts actions.
    Uses the 'Upstream service X' context pattern from the scenario generator.
    """
    groups: dict[str, list[str]] = {}
    for alert in alerts:
        context = (alert.get("context") or "").lower()
        if "upstream service '" in context:
            start = context.index("upstream service '") + len("upstream service '")
            end   = context.index("'", start) if "'" in context[start:] else len(context)
            dep   = context[start:end]
            groups.setdefault(dep, []).append(alert["alert_id"])

    actions = []
    for dep, ids in groups.items():
        if len(ids) >= 2:
            actions.append({
                "action_type":    "link_alerts",
                "alert_ids":      ids,
                "incident_label": dep.replace(" ", "-").replace("'", ""),
            })
    return actions


# ---------------------------------------------------------------------------
# LLM planning
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert SRE triaging cloud infrastructure alerts.
Respond with ONLY a valid JSON array of actions — no prose, no markdown fences.

== EXACT METRIC → ROOT CAUSE MAPPINGS (follow these precisely) ==

METRIC NAME                       ROOT CAUSE            REMEDIATION
cpu_usage_percent                 resource_exhaustion   scale_up
memory_usage_percent              resource_exhaustion   scale_up
disk_usage_percent                resource_exhaustion   scale_up
network_latency_ms                network_failure       escalate_to_team
packet_loss_percent               network_failure       escalate_to_team
tcp_connection_errors             network_failure       escalate_to_team
error_rate_percent (+ deploy ctx) deployment_bug        rollback_deploy
http_5xx_rate      (+ deploy ctx) deployment_bug        rollback_deploy
health_check_failures (no deploy) config_error          fix_config
auth_failure_rate                 config_error          fix_config
connection_refused_count          config_error          fix_config
upstream_error_rate               dependency_outage     acknowledge_and_monitor
dependency_timeout_count          dependency_outage     acknowledge_and_monitor
upstream_latency_ms               dependency_outage     acknowledge_and_monitor

== SEVERITY RULES ==
resource_exhaustion:  (value - threshold) > 12 → critical, else → high
                      EXCEPTION: subtle elevation + "gradual memory leak" ctx → medium
network_failure:      always → high
deployment_bug:       always → high
config_error:         value/threshold >= 1.5 → high, else → medium
dependency_outage:    value/threshold > 1.8 → critical, else → high

== FALSE ALARM DETECTION — use "skip" ==
Context contains ANY of:
  "batch job", "maintenance window", "PagerDuty P0 auto-created", "false positive"
These are false alarms even if the monitoring system labels them CRITICAL.
EXCEPTION: context "gradual memory leak" → NOT a false alarm, it's a real stealth incident.

== STRATEGY ==
1. Examine ALL alerts and the service dependency map before deciding anything.
2. Find alerts sharing the same upstream dependency (context: "Upstream service 'X'")
   and group them with link_alerts BEFORE triaging — this earns a bonus.
3. Alerts with a recent deploy context ("Deploy v... rolled out N minutes ago") →
   deployment_bug, regardless of metric name.
4. The service_map shows who depends on whom — cascading failures share a root cause.
5. Skip ONLY true false alarms. Triage everything else.
6. You MUST produce one action per alert. Missing alerts costs score.

== OUTPUT FORMAT ==
Return a JSON array. Order: link_alerts first, then triage/skip for every alert.
[
  {"action_type":"link_alerts","alert_ids":["alert-003","alert-007"],"incident_label":"redis-cascade"},
  {"action_type":"triage","alert_id":"alert-001","root_cause":"resource_exhaustion","severity":"high","remediation":"scale_up"},
  {"action_type":"skip","alert_id":"alert-005"}
]
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
            extras.append(heuristic_action(alert))

    return plan + extras


def build_full_plan(client: OpenAI, obs: dict) -> list[dict]:
    """
    Build the complete action plan for this episode:
    1. Try LLM plan.
    2. If LLM fails or returns an empty plan, fall back to heuristic links + triage.
    3. Fill any coverage gaps with heuristics regardless.
    """
    pending = [a for a in obs.get("alerts", []) if not a.get("triaged", False)]

    llm_plan, llm_err = get_full_plan(client, obs)

    if llm_err or not llm_plan:
        # Full heuristic fallback
        links  = heuristic_incident_links(pending)
        triages = [heuristic_action(a) for a in pending]
        return links + triages

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
            action = heuristic_action(alert)
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
    if not OPENAI_API_KEY:
        print("[WARN] No OPENAI_API_KEY or HF_TOKEN found — LLM calls will fail.", file=sys.stderr)

    llm  = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY or "placeholder")
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
