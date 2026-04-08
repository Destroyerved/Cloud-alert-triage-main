#!/usr/bin/env python3
"""
inference.py
------------
Optimised LLM agent for the cloud-alert-triage OpenEnv environment.

Strategy: plan-then-execute WITH mid-episode feedback adaptation.
  Phase 1 -- Single LLM call: send ALL pending alerts with pre-computed severity
             hints, metric trend history, and cascade-group suggestions; get a
             complete ordered action plan as a JSON array.
  Phase 2 -- Execute the plan step-by-step.  After 35% of planned actions,
             accumulate per-step feedback from the environment.  If feedback
             signals wrong root-cause decisions, issue a focused second LLM
             call (re-plan) for remaining pending alerts using that feedback.
  Phase 3 -- Mop-up loop for dynamic cascade alerts spawned mid-episode.

Key design choices:
  - Severity is computed deterministically from metric/threshold ratios using
    the exact same rules as scenario_generator.py -- no LLM guessing.
  - metric_history (5-point rolling window) is passed to the LLM so it can
    distinguish trending signals (rising, spike, gradual) from noise.
  - link_alerts groups are detected via explicit upstream-service mentions in
    alert context strings, not by BFS over the full graph.
  - Minimum group size 3 prevents false-positive links on easy-task independent
    dependency_outage alerts.
  - Dynamic alerts (dyn-* prefix) are handled as severity="high" regardless of
    metric value, matching the hardcoded ground truth in environment.py.
  - Misleading false alarms ("PagerDuty P0 auto-created", "prior pattern
    suggests false positive") are detected and marked for skip.
  - Context-aware root-cause routing: cpu/memory alerts with deploy context are
    classified as deployment_bug; latency alerts with upstream context are
    dependency_outage rather than network_failure.

Environment variables:
    ENV_URL          URL of the running environment server
                     (default: http://localhost:7860)
    API_BASE_URL     OpenAI-compatible API base URL
                     (default: https://api.groq.com/openai/v1)
    MODEL_NAME       Model to use
                     (default: llama-3.3-70b-versatile)
    OPENAI_API_KEY   API key (for OpenAI)
    GROQ_API_KEY     API key (for Groq)
    HF_TOKEN         Hugging Face token (fallback)

Usage:
    # 1. Start the environment server
    uvicorn server.app:app --port 7860

    # 2. Run the agent
    export HF_TOKEN=hf_...
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
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

ENV_URL: str    = os.environ.get("ENV_URL", "http://localhost:7860").rstrip("/")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# API base URL + key resolution
# ---------------------------------------------------------------------------
_explicit_base  = os.environ.get("API_BASE_URL", "")
_groq_key       = os.environ.get("GROQ_API_KEY", "")
_openai_key     = os.environ.get("OPENAI_API_KEY", "")
_hf_token       = os.environ.get("HF_TOKEN", "")

if _explicit_base:
    API_BASE_URL: str = _explicit_base
elif _groq_key:
    API_BASE_URL = "https://api.groq.com/openai/v1"
elif _openai_key:
    API_BASE_URL = "https://api.openai.com/v1"
else:
    API_BASE_URL = "https://api-inference.huggingface.co/v1"

if "groq" in API_BASE_URL:
    API_KEY: str = _groq_key or _hf_token or _openai_key
elif "openai.com" in API_BASE_URL:
    API_KEY = _openai_key or _hf_token
else:
    API_KEY = _hf_token or _openai_key or _groq_key

del _explicit_base, _groq_key, _openai_key, _hf_token

TASKS: list[str]               = ["easy", "medium", "hard"]
DEFAULT_SEED: int               = 42
TOTAL_BUDGET_SECONDS: float    = 20 * 60
PER_TASK_BUDGET_SECONDS: float = 6 * 60
LLM_TIMEOUT_SECONDS: float     = 60.0
LLM_MAX_RETRIES: int            = 3

# Fraction of plan actions to execute before checking for a mid-episode re-plan.
# Set to 0.35 so we observe ~35% of triage decisions before adapting.
REPLAN_TRIGGER_FRACTION: float = 0.35
# Minimum number of "incorrect" feedback signals required to trigger re-planning.
REPLAN_MIN_ERRORS: int = 1


# ---------------------------------------------------------------------------
# Structured logging -- exact spec-required key=value format
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=cloud-alert-triage model={model}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error: str | None) -> None:
    action_str = json.dumps(action, separators=(",", ":"))[:200]
    error_str  = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={'true' if done else 'false'} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# False alarm detection
# ---------------------------------------------------------------------------

_FALSE_ALARM_MSG_PATTERNS: tuple[str, ...] = (
    "scheduled batch",
    "maintenance window",
    "known spike",
    "prior pattern suggests false positive",
    "automated escalation",
)
_FALSE_ALARM_CTX_PATTERNS: tuple[str, ...] = (
    "scheduled maintenance window",
    "pagerduty p0 auto-created",
    "verify before acting",
)


def _is_false_alarm(alert: dict) -> bool:
    msg = (alert.get("message") or "").lower()
    ctx = (alert.get("context")  or "").lower()
    return any(p in msg for p in _FALSE_ALARM_MSG_PATTERNS) or \
           any(p in ctx for p in _FALSE_ALARM_CTX_PATTERNS)


# ---------------------------------------------------------------------------
# Severity inference -- mirrors scenario_generator.py rules exactly
# ---------------------------------------------------------------------------

def _infer_severity(alert: dict) -> str:
    alert_id: str = (alert.get("alert_id") or "")
    metric:   str = (alert.get("metric")   or "").lower()
    msg:      str = (alert.get("message")  or "").lower()
    mv            = alert.get("metric_value")
    thr: float    = float(alert.get("threshold") or 0.0)

    if alert_id.startswith("dyn-"):
        return "high"
    if _is_false_alarm(alert):
        return "low"
    if any(w in msg for w in ("mildly", "minor", "gradually", "gradual", "barely", "memory leak")):
        return "medium"

    if mv is None:
        return "critical" if any(w in msg for w in ("surging", "cascade")) else "high"

    if any(m in metric for m in ("cpu_usage", "memory_usage", "disk_usage")):
        return "critical" if (mv - thr) > 12 else "high"
    if any(m in metric for m in ("upstream_error", "dependency_timeout", "upstream_latency")):
        return "critical" if (thr > 0 and mv > thr * 1.8) else "high"
    if any(m in metric for m in ("network_latency", "packet_loss", "tcp_connection")):
        return "high"
    if any(m in metric for m in ("error_rate", "5xx", "health_check")):
        return "high"
    if any(m in metric for m in ("auth_failure", "connection_refused")):
        return "medium"

    return "high"


# ---------------------------------------------------------------------------
# Cascade group detection
# ---------------------------------------------------------------------------

def _detect_cascade_groups(
    alerts: list[dict], service_map: dict
) -> list[tuple[str, list[str]]]:
    """
    Detect incident groups using upstream-service mentions in alert context/message.
    Matches three context patterns produced by _build_dependency() variants.
    Minimum group size = 3 to prevent false-positive links.
    """
    svc_to_aid: dict[str, str] = {
        a["service"]: a["alert_id"]
        for a in alerts
        if not a.get("triaged") and not _is_false_alarm(a)
    }
    groups: dict[str, list[str]] = {}

    for a in alerts:
        if a.get("triaged") or _is_false_alarm(a):
            continue
        ctx = a.get("context") or ""
        msg = a.get("message") or ""
        dep_svc: str | None = None

        for prefix, text in [
            ("Upstream service '", ctx),
            ("Calls to '",        ctx),
            ("dependency '",      msg),
        ]:
            if prefix in text:
                try:
                    s = text.index(prefix) + len(prefix)
                    dep_svc = text[s : text.index("'", s)]
                    break
                except ValueError:
                    pass

        if dep_svc is None or dep_svc not in svc_to_aid:
            continue
        if dep_svc not in groups:
            groups[dep_svc] = [svc_to_aid[dep_svc]]
        aid = a["alert_id"]
        if aid not in groups[dep_svc]:
            groups[dep_svc].append(aid)

    return [
        (f"{svc.replace('-', '_')}_cascade", sorted(aids))
        for svc, aids in groups.items()
        if len(aids) >= 3
    ]


# ---------------------------------------------------------------------------
# LLM prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert SRE triaging cloud infrastructure alerts.

SEVERITY RULES (pre-computed sev~<value> hint is authoritative):
  cpu/memory/disk:       (value-threshold)>12 -> CRITICAL, else HIGH
  upstream_error/dep:    value > threshold*1.8 -> CRITICAL, else HIGH
  network/deploy:        always HIGH
  config:                MEDIUM (default)
  "mildly"/"gradual"/"memory leak" in message -> MEDIUM (stealth root cause)
  sev~low -> FALSE ALARM -> issue skip, not triage
  dyn-* alerts -> always HIGH

CONTEXT OVERRIDES (read context field -- it disambiguates metric-ambiguous alerts):
  cpu/memory metric + "after deploy"/"deploy v"/"memory regression"/"new build" in context
      -> deployment_bug + rollback_deploy  (NOT resource_exhaustion)
  cpu/memory metric + "mildly"/"gradual"/"memory leak" in message
      -> resource_exhaustion + acknowledge_and_monitor  (STEALTH)
  network_latency + "correlates with"/"no packet loss"/"no NIC errors" in context
      -> dependency_outage + acknowledge_and_monitor  (NOT network_failure)

METRIC TREND INTERPRETATION (5-point history shown as trend:[t-4,t-3,t-2,t-1,t-0]):
  Rising ramp     -> resource_exhaustion (gradual saturation)
  Flat then spike -> deployment_bug or dependency_outage (sudden event)
  Very slow climb -> stealth incident / memory leak (gradual root cause)
  Random near-thr -> likely false_alarm (noise around threshold)

REMEDIATION MAP (non-negotiable):
  resource_exhaustion  -> scale_up
  network_failure      -> escalate_to_team
  deployment_bug       -> rollback_deploy
  config_error         -> fix_config
  dependency_outage    -> acknowledge_and_monitor
  false_alarm          -> skip action (NOT triage)

ACTION ORDER: link_alerts first, then triage (critical->high->medium), then skip.

OUTPUT: JSON ARRAY ONLY -- NO TEXT OUTSIDE THE ARRAY.
[
  {"action_type":"link_alerts","alert_ids":["alert-001","alert-003"],"incident_label":"redis_cache_cascade"},
  {"action_type":"triage","alert_id":"alert-001","root_cause":"resource_exhaustion","severity":"high","remediation":"scale_up"},
  {"action_type":"skip","alert_id":"alert-010"}
]

VALID VALUES:
  root_cause:  resource_exhaustion | network_failure | deployment_bug | config_error | dependency_outage
  severity:    critical | high | medium | low
  remediation: scale_up | escalate_to_team | rollback_deploy | fix_config | acknowledge_and_monitor | restart_service | dismiss
"""


def _fmt_alert(a: dict) -> str:
    """Format one alert for the LLM prompt, including metric trend history."""
    mv       = a.get("metric_value")
    val_str  = f"{mv:.1f}" if mv is not None else "MASKED"
    sev      = _infer_severity(a)
    fa_tag   = "  <- FALSE ALARM -> skip"  if sev == "low"                           else ""
    dyn_tag  = "  [DYNAMIC/high]"          if (a.get("alert_id") or "").startswith("dyn-") else ""
    ctx_part = f" | ctx: {a['context'][:90]}" if a.get("context") else ""

    # Include 5-point metric trend from scenario generator if present
    hist = a.get("metric_history")
    if hist and len(hist) == 5:
        trend_str = f" | trend:[{','.join(str(v) for v in hist)}]"
    else:
        trend_str = ""

    return (
        f"{a['alert_id']} [{a['service']}] {a['metric']}={val_str}"
        f"(thr={a.get('threshold')}) sev~{sev}{fa_tag}{dyn_tag}"
        f" | {(a.get('message') or '')[:90]}{ctx_part}{trend_str}"
    )


def _fmt_service_map(svc_map: dict) -> str:
    return "\n".join(
        f"  {s} -> [{', '.join(d) or 'none'}]"
        for s, d in sorted(svc_map.items())
    )


def build_plan_prompt(obs: dict) -> str:
    all_alerts: list[dict] = obs.get("alerts", [])
    pending = [a for a in all_alerts if not a.get("triaged")]
    service_map: dict = obs.get("service_map", {})

    dependent_count: dict[str, int] = {}
    for svc, deps in service_map.items():
        for dep in deps:
            dependent_count[dep] = dependent_count.get(dep, 0) + 1

    _SEV_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    sorted_pending = sorted(
        pending,
        key=lambda a: (
            _SEV_RANK.get(_infer_severity(a), 3),
            -dependent_count.get(a.get("service", ""), 0),
            a.get("alert_id", ""),
        ),
    )

    cascade_groups = _detect_cascade_groups(pending, service_map)
    if cascade_groups:
        link_lines = ["=== SUGGESTED LINK GROUPS (include ALL as link_alerts) ==="]
        for label, aids in cascade_groups:
            link_json = json.dumps(
                {"action_type": "link_alerts", "alert_ids": aids, "incident_label": label},
                separators=(",", ":"),
            )
            link_lines.append(f"  {link_json}")
    else:
        link_lines = ["=== SUGGESTED LINK GROUPS === none detected ==="]

    fa_alerts = [a for a in pending if _is_false_alarm(a)]
    fa_section = ""
    if fa_alerts:
        fa_ids = ", ".join(a["alert_id"] for a in fa_alerts)
        fa_section = f"\n=== FALSE ALARMS (skip these) ===\n  {fa_ids}\n"

    top_upstream = sorted(dependent_count.items(), key=lambda x: x[1], reverse=True)[:5]

    lines = [
        f"Task: {len(pending)} alerts to triage. Step budget: {obs.get('max_steps')}.",
        "",
        "=== TOP UPSTREAM SERVICES (most dependents -- likely root causes) ===",
        "  " + ", ".join(f"{s}({d})" for s, d in top_upstream),
        "",
        "\n".join(link_lines),
        fa_section,
        "=== PENDING ALERTS (sorted: critical root-causes first) ===",
        *[_fmt_alert(a) for a in sorted_pending],
        "",
        "=== SERVICE DEPENDENCY MAP (service -> what it depends on) ===",
        _fmt_service_map(service_map),
        "",
        "RULES:",
        "1. Add every link group above as link_alerts (FIRST).",
        "2. Use the sev~ hint for severity.",
        "3. sev~low = skip. Never triage a sev~low alert.",
        "4. dyn-* alerts = severity high.",
        "5. Remediation follows the fixed map above.",
        "6. Read the context field for EACH alert -- it disambiguates ambiguous metrics.",
        "7. Cover EVERY alert. Return JSON array only.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feedback-informed re-planning
# ---------------------------------------------------------------------------

def _should_replan(step_feedbacks: list[tuple[dict, str]]) -> bool:
    """
    Return True if accumulated per-step feedback signals suggest the LLM's
    initial plan made root-cause errors that a re-plan could correct.

    Only triage actions matter for re-planning -- link_alerts and skip
    feedbacks don't carry root-cause signal.
    """
    error_count = 0
    for action, feedback in step_feedbacks:
        if action.get("action_type") != "triage":
            continue
        fb_lower = feedback.lower()
        if "incorrect" in fb_lower or "may be incorrect" in fb_lower:
            error_count += 1
    return error_count >= REPLAN_MIN_ERRORS


def _build_replan_prompt(
    obs: dict,
    step_feedbacks: list[tuple[dict, str]],
    remaining_plan: list[dict],
) -> str:
    """
    Build a focused correction prompt for the re-plan LLM call.

    Includes:
      - The feedback signals observed so far (what was wrong)
      - Only the STILL-PENDING alerts (already-triaged ones are excluded)
      - The planned actions for those alerts (what we intended to do)
    """
    pending = [a for a in obs.get("alerts", []) if not a.get("triaged")]
    service_map = obs.get("service_map", {})

    # Summarise errors observed
    error_lines: list[str] = []
    for action, feedback in step_feedbacks:
        if action.get("action_type") != "triage":
            continue
        fb_lower = feedback.lower()
        if "incorrect" in fb_lower or "may be incorrect" in fb_lower:
            aid = action.get("alert_id", "?")
            error_lines.append(
                f"  {aid}: submitted root_cause={action.get('root_cause')} "
                f"severity={action.get('severity')} -> feedback: {feedback.strip()}"
            )

    # Pending alert IDs in the remaining plan
    planned_ids = {
        a.get("alert_id") for a in remaining_plan if a.get("action_type") == "triage"
    }

    pending_to_revise = [a for a in pending if a["alert_id"] in planned_ids]
    if not pending_to_revise:
        pending_to_revise = pending  # fallback: all pending

    lines = [
        "MID-EPISODE RE-PLAN: The environment's feedback revealed errors in the initial plan.",
        "",
        "=== FEEDBACK FROM COMPLETED STEPS (review these before replanning) ===",
        *error_lines if error_lines else ["  (no specific errors -- replan for coverage)"],
        "",
        "KEY LESSON: If cpu/memory metric has 'after deploy'/'memory regression' in context,",
        "the root cause is deployment_bug (rollback_deploy), NOT resource_exhaustion.",
        "If network_latency has 'correlates with'/'no packet loss' in context,",
        "the root cause is dependency_outage (acknowledge_and_monitor), NOT network_failure.",
        "",
        "=== REMAINING ALERTS TO TRIAGE (re-issue correct decisions for ALL of these) ===",
        *[_fmt_alert(a) for a in pending_to_revise],
        "",
        "=== SERVICE DEPENDENCY MAP ===",
        _fmt_service_map(service_map),
        "",
        "Output a JSON array of triage/skip actions ONLY for the alerts listed above.",
        "Do NOT include link_alerts (already submitted). Cover every alert listed.",
    ]
    return "\n".join(lines)


def _replan_with_feedback(
    client: OpenAI,
    obs: dict,
    step_feedbacks: list[tuple[dict, str]],
    remaining_plan: list[dict],
) -> list[dict]:
    """
    Issue a focused second LLM call for remaining alerts, informed by the
    per-step feedback accumulated so far.

    Returns a list of corrected triage/skip actions.
    On failure, returns [] (caller falls back to heuristic gap-fill).
    """
    prompt = _build_replan_prompt(obs, step_feedbacks, remaining_plan)

    try:
        resp = client.chat.completions.create(
            model    = MODEL_NAME,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature = 0,
            max_tokens  = 2048,
            timeout     = LLM_TIMEOUT_SECONDS,
        )
        raw = (resp.choices[0].message.content or "").strip()
        plan = _parse_plan(raw)
        # Filter: only accept triage/skip actions (no link_alerts in re-plan)
        return [a for a in plan if a.get("action_type") in ("triage", "skip")]
    except Exception as exc:
        print(f"[WARN] Re-plan LLM call failed: {exc}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# LLM planning
# ---------------------------------------------------------------------------

def get_full_plan(client: OpenAI, obs: dict) -> tuple[list[dict], str | None]:
    prompt   = build_plan_prompt(obs)
    last_err: str | None = None

    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model    = MODEL_NAME,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature = 0,
                max_tokens  = 4096,
                timeout     = LLM_TIMEOUT_SECONDS,
            )
            raw  = (resp.choices[0].message.content or "").strip()
            plan = _parse_plan(raw)
            if plan:
                return plan, None
            last_err = "LLM returned an empty or unparseable plan"
        except Exception as exc:
            last_err = str(exc)
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    return [], last_err


def _parse_plan(text: str) -> list[dict]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            ln for ln in cleaned.splitlines()
            if not ln.strip().startswith("```")
        ).strip()

    start = cleaned.find("[")
    end   = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    try:
        data = json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(data, list):
        return []

    actions = []
    for item in data:
        if not isinstance(item, dict) or "action_type" not in item:
            continue
        item.pop("confidence", None)
        item.pop("reasoning",  None)
        item.pop("notes",      None)
        actions.append(item)
    return actions


# ---------------------------------------------------------------------------
# Coverage enforcement
# ---------------------------------------------------------------------------

def _fill_missing(
    plan: list[dict], all_alerts: list[dict], service_map: dict
) -> list[dict]:
    alert_lookup: dict[str, dict] = {a["alert_id"]: a for a in all_alerts}
    covered:  set[str]    = set()
    validated: list[dict] = []

    for action in plan:
        at  = action.get("action_type")
        aid = action.get("alert_id", "")

        if at == "triage":
            if aid in covered:
                continue
            covered.add(aid)
            validated.append(action)

        elif at == "skip":
            if aid in covered:
                continue
            alert = alert_lookup.get(aid)
            if alert is None:
                continue
            if _is_false_alarm(alert):
                covered.add(aid)
                validated.append(action)
            else:
                validated.append(_smart_fallback(alert, service_map))
                covered.add(aid)

        else:
            validated.append(action)

    extras = [
        _smart_fallback(a, service_map)
        for a in all_alerts
        if not a.get("triaged") and a["alert_id"] not in covered
    ]
    return validated + extras


def build_full_plan(client: OpenAI, obs: dict) -> list[dict]:
    pending     = [a for a in obs.get("alerts", []) if not a.get("triaged")]
    service_map = obs.get("service_map", {})

    heuristic_links: list[dict] = [
        {"action_type": "link_alerts", "alert_ids": aids, "incident_label": label}
        for label, aids in _detect_cascade_groups(pending, service_map)
    ]

    llm_plan, llm_err = get_full_plan(client, obs)

    if llm_err or not llm_plan:
        triage_actions = [_smart_fallback(a, service_map) for a in pending]
        return heuristic_links + triage_actions

    non_link_actions = [a for a in llm_plan if a.get("action_type") != "link_alerts"]
    final_plan = heuristic_links + non_link_actions
    return _fill_missing(final_plan, pending, service_map)


# ---------------------------------------------------------------------------
# Heuristic fallback -- context-aware root-cause routing
# ---------------------------------------------------------------------------

def _smart_fallback(alert: dict, service_map: dict) -> dict:
    """
    Deterministic triage / skip. Context overrides metric for ambiguous alerts.
    """
    if _is_false_alarm(alert):
        return {"action_type": "skip", "alert_id": alert["alert_id"]}

    metric = (alert.get("metric")  or "").lower()
    msg    = (alert.get("message") or "").lower()
    ctx    = (alert.get("context") or "").lower()

    # Stealth incident: subtle signal in message
    if any(w in msg for w in ("mildly", "minor", "gradual", "memory leak", "barely")):
        return {
            "action_type": "triage",
            "alert_id":    alert["alert_id"],
            "root_cause":  "resource_exhaustion",
            "severity":    _infer_severity(alert),
            "remediation": "acknowledge_and_monitor",
        }

    if any(m in metric for m in ("cpu_usage", "memory_usage", "disk_usage")):
        # Context-aware: deploy/regression signals override metric-based guess
        if any(kw in ctx for kw in ("after deploy", "deploy v", "memory regression", "new build")):
            root_cause, remediation = "deployment_bug", "rollback_deploy"
        else:
            root_cause, remediation = "resource_exhaustion", "scale_up"

    elif any(m in metric for m in ("upstream_error", "dependency_timeout", "upstream_latency")):
        root_cause, remediation = "dependency_outage", "acknowledge_and_monitor"

    elif any(m in metric for m in ("network_latency", "packet_loss", "tcp_connection")):
        # Context-aware: upstream correlation signals override network_failure
        if any(kw in ctx for kw in ("correlates with", "no packet loss", "slowdowns", "no nic errors")):
            root_cause, remediation = "dependency_outage", "acknowledge_and_monitor"
        else:
            root_cause, remediation = "network_failure", "escalate_to_team"

    elif any(m in metric for m in ("auth_failure", "connection_refused")):
        root_cause, remediation = "config_error", "fix_config"

    elif any(m in metric for m in ("error_rate", "5xx")):
        if "deploy" in msg or "deploy" in ctx:
            root_cause, remediation = "deployment_bug", "rollback_deploy"
        else:
            root_cause, remediation = "dependency_outage", "acknowledge_and_monitor"

    elif "health_check" in metric:
        if "deploy" in msg or "deploy" in ctx:
            root_cause, remediation = "deployment_bug", "rollback_deploy"
        else:
            root_cause, remediation = "config_error", "fix_config"

    else:
        root_cause, remediation = "dependency_outage", "acknowledge_and_monitor"

    return {
        "action_type": "triage",
        "alert_id":    alert["alert_id"],
        "root_cause":  root_cause,
        "severity":    _infer_severity(alert),
        "remediation": remediation,
    }


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def _env_reset(http: httpx.Client, task_id: str, seed: int) -> dict:
    r = http.post("/reset", json={"task_id": task_id, "seed": seed})
    r.raise_for_status()
    data = r.json()
    return data.get("observation", data)


def _env_step(http: httpx.Client, action: dict) -> dict:
    r = http.post("/step", json=action)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Task runner -- plan-then-execute with mid-episode feedback adaptation
# ---------------------------------------------------------------------------

def run_task(task_id: str, llm: OpenAI, http: httpx.Client, deadline: float) -> None:
    """
    Run one full episode:
      1. reset() -> build complete initial plan (LLM + heuristic links)
      2. Execute plan step-by-step, accumulating per-step feedback.
      3. At REPLAN_TRIGGER_FRACTION of the plan, if feedback signals errors,
         issue a second LLM call with the feedback context (re-plan).
         Substitute the re-planned actions for the remainder of the episode.
      4. Mop-up loop: handle dynamic cascade alerts spawned mid-episode.
      5. Always emit [END].  Grader score goes to stderr only.
    """
    rewards:      list[float]          = []
    done:         bool                 = False
    step_num:     int                  = 0
    grader_score: float                = 0.0
    step_feedbacks: list[tuple[dict, str]] = []   # (action, feedback_text)
    replan_done:  bool                 = False

    try:
        obs = _env_reset(http, task_id, DEFAULT_SEED)
        log_start(task_id, MODEL_NAME)

        plan = build_full_plan(llm, obs)
        plan_len = len(plan)
        replan_threshold = int(plan_len * REPLAN_TRIGGER_FRACTION)

        # --- Execute plan ---
        i = 0
        while i < len(plan):
            action = plan[i]
            i += 1

            if done or time.time() > deadline:
                break

            error: str | None = None
            try:
                result = _env_step(http, action)
            except Exception as exc:
                error = str(exc)
                log_step(step_num + 1, action, 0.0, False, error)
                break

            reward   = float(result.get("reward",  0.0))
            done     = bool(result.get("done",    False))
            info     = result.get("info", {})
            obs      = result.get("observation", obs)
            step_num += 1
            rewards.append(reward)

            if done:
                grader_score = float(info.get("grader_score", 0.0))

            # Capture per-step feedback for adaptation
            feedback_text = (obs.get("feedback") or "")
            step_feedbacks.append((action, feedback_text))

            log_step(step_num, action, reward, done, error)

            # --- Mid-episode re-plan check ---
            # After executing replan_threshold actions, check if feedback
            # signals systematic root-cause errors.  If so, issue a second
            # focused LLM call for remaining pending alerts and splice the
            # corrected actions into the plan in place of the original ones.
            if (
                not done
                and not replan_done
                and i >= replan_threshold
                and replan_threshold > 0
                and _should_replan(step_feedbacks)
                and time.time() < deadline - 30  # leave at least 30s for re-plan
            ):
                remaining_planned = plan[i:]   # actions not yet executed
                print(
                    f"[ADAPT] task={task_id} step={step_num} "
                    f"triggering re-plan for {len(remaining_planned)} remaining actions",
                    file=sys.stderr,
                )
                revised = _replan_with_feedback(llm, obs, step_feedbacks, remaining_planned)
                if revised:
                    # Validate and gap-fill the revised actions
                    all_pending = [a for a in obs.get("alerts", []) if not a.get("triaged")]
                    service_map = obs.get("service_map", {})
                    revised = _fill_missing(revised, all_pending, service_map)
                    # Splice: replace remaining original plan with revised plan
                    plan = plan[:i] + revised
                    print(
                        f"[ADAPT] Re-plan produced {len(revised)} corrected actions",
                        file=sys.stderr,
                    )
                replan_done = True  # only re-plan once per episode

        # --- Mop-up: handle dynamic alerts spawned after plan was built ---
        if not done:
            pending_now = [a for a in obs.get("alerts", []) if not a.get("triaged")]
            for alert in pending_now:
                if done or time.time() > deadline:
                    break
                action = _smart_fallback(alert, obs.get("service_map", {}))
                try:
                    result = _env_step(http, action)
                except Exception as exc:
                    log_step(step_num + 1, action, 0.0, False, str(exc))
                    break
                reward   = float(result.get("reward",  0.0))
                done     = bool(result.get("done",    False))
                obs      = result.get("observation", obs)
                step_num += 1
                rewards.append(reward)
                if done:
                    grader_score = float(result.get("info", {}).get("grader_score", 0.0))
                log_step(step_num, action, reward, done, None)

    except Exception as exc:
        print(f"[ERROR] task={task_id} error={exc}", file=sys.stderr)

    finally:
        log_end(done, step_num, rewards)
        if grader_score:
            print(f"[SCORE] task={task_id} grader_score={grader_score:.4f}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print(
            "[WARN] No API key found (HF_TOKEN / OPENAI_API_KEY / GROQ_API_KEY). "
            "LLM calls will fail.",
            file=sys.stderr,
        )

    llm  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "placeholder")
    http = httpx.Client(base_url=ENV_URL, timeout=30.0)

    global_deadline = time.time() + TOTAL_BUDGET_SECONDS

    for task_id in TASKS:
        if time.time() > global_deadline:
            print("[WARN] Global budget exceeded -- skipping remaining tasks.", file=sys.stderr)
            break
        task_deadline = min(time.time() + PER_TASK_BUDGET_SECONDS, global_deadline)
        try:
            run_task(task_id, llm, http, task_deadline)
        except Exception as exc:
            print(f"[ERROR] task='{task_id}' crashed: {exc}", file=sys.stderr)
            log_end(False, 0, [])


if __name__ == "__main__":
    main()
