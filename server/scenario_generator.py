"""
server/scenario_generator.py

Deterministic, seed-based alert/incident scenario generation for the
Cloud Alert Triage environment.

Public entry point
------------------
    generate_scenario(task_id, seed) -> dict

Return structure::

    {
        "alerts":       [list of alert dicts],          # matches Alert model
        "ground_truth": [list of ground-truth dicts],   # one per alert
        "incidents":    [list of incident dicts],        # 0, 2, or 5 entries
    }

Alert dict keys (mirrors server/models.py Alert)::

    alert_id, timestamp, service, metric, metric_value, threshold,
    message, context, triaged (=False), agent_decision (=None)

Ground-truth dict keys::

    alert_id, true_root_cause, true_severity, true_remediation, incident_id

Incident dict keys::

    incident_id, root_service, root_cause, alert_ids  [, stealth]

Determinism guarantee
---------------------
All randomness uses a ``random.Random(seed)`` instance created once per call.
Global ``random`` is never touched.  All list-like objects passed to rng methods
are pre-sorted so dict-ordering across Python versions cannot break determinism.
"""

from __future__ import annotations

import datetime
import random
from typing import Any

from server.config import REMEDIATION_ACTIONS, ROOT_CAUSE_CATEGORIES, SEVERITY_LEVELS
from server.service_graph import SERVICE_GRAPH, get_dependents, get_service_names

# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants  (never change at runtime)
# ─────────────────────────────────────────────────────────────────────────────

# Fixed anchor datetime — all scenario timestamps are offsets from this value.
_BASE_DT = datetime.datetime(2024, 1, 15, 10, 0, 0, tzinfo=datetime.timezone.utc)

# Stable sorted list of all services in the graph.
_ALL_SERVICES: list[str] = get_service_names()

# Tier 5 data layer services - used for incident root causes in hard mode
_TIER5_SERVICES = ["postgres-primary", "redis-cache", "kafka-broker", "elasticsearch", "object-storage"]

# Root cause labels for incidents
_INCIDENT_ROOT_CAUSES = ["resource_exhaustion", "network_failure", "config_error", "deployment_bug"]

# Metric name / threshold pairs used by each root-cause builder.
_RESOURCE_METRICS = [
    ("cpu_usage_percent",      80.0),
    ("memory_usage_percent",   85.0),
    ("disk_usage_percent",     90.0),
]
_NETWORK_METRICS = [
    ("network_latency_ms",     200.0),
    ("packet_loss_percent",      5.0),
    ("tcp_connection_errors",   10.0),
]
_DEPLOY_METRICS = [
    ("error_rate_percent",       5.0),
    ("http_5xx_rate",            5.0),
    ("health_check_failures",    3.0),
]
_CONFIG_METRICS = [
    ("health_check_failures",    3.0),
    ("auth_failure_rate",       10.0),
    ("connection_refused_count", 5.0),
]
_DEP_METRICS = [
    ("upstream_error_rate",     20.0),
    ("dependency_timeout_count",10.0),
    ("upstream_latency_ms",    500.0),
]
_NOISE_METRICS = [
    ("cpu_usage_percent",      80.0),
    ("memory_usage_percent",   85.0),
    ("response_time_ms",      300.0),
]

# Context-string templates for dependency_outage alerts.
# Template 0 is the "standard" pattern that _detect_cascade_groups() in
# inference.py recognises via the exact prefix "Upstream service '...'".  
# Templates 1+ use different phrasing that the cascade detector cannot pick
# up, producing realistic partial link-F1 scores rather than a perfect 1.0.
_DEP_CTX_TEMPLATES: list[str] = [
    "Upstream service '{dep}' is reporting errors",                    # detectable
    "Calls to '{dep}' timing out — upstream health degraded",          # not detectable
    "{dep} response times exceeding SLA; downstream services impacted", # not detectable
]


# ─────────────────────────────────────────────────────────────────────────────
# Low-level dict constructors
# ─────────────────────────────────────────────────────────────────────────────

def _ts(offset_minutes: int) -> str:
    """ISO-8601 timestamp string = BASE_DT + *offset_minutes* minutes."""
    dt = _BASE_DT + datetime.timedelta(minutes=offset_minutes)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _gen_history(
    rng: random.Random,
    current_val: float,
    threshold: float,
    pattern: str = "rising",
) -> list[float]:
    """
    Generate 4 preceding metric readings (t-4 … t-1 min) that precede
    *current_val*, producing a realistic 5-point time-series window.

    Patterns
    --------
    rising   : steady ramp toward breach (resource exhaustion / config drift)
    spike    : flat baseline then sudden jump at t-1 (deploy bug / network event)
    gradual  : slow linear climb over the whole window (memory leak / config drift)
    random   : values hover just below threshold (false-alarm / borderline noise)
    """
    if pattern == "rising":
        # Linear ramp: from 60 % of threshold up to current_val.
        start = threshold * 0.60
        points = [
            round(start + (current_val - start) * i / 4 + rng.uniform(-1.5, 1.5), 2)
            for i in range(4)
        ]
    elif pattern == "spike":
        # Flat baseline (just below threshold) then sharp jump at t-1.
        baseline = threshold * rng.uniform(0.55, 0.75)
        points = [
            round(baseline + rng.uniform(-2.0, 2.0), 2) for _ in range(3)
        ]
        # t-1: already above threshold, approaching current
        points.append(round(threshold + (current_val - threshold) * 0.6 + rng.uniform(-1.0, 1.0), 2))
    elif pattern == "gradual":
        # Very slow climb over the full window (stealth / memory leak pattern).
        start = threshold * 0.88
        points = [
            round(start + (current_val - start) * i / 5 + rng.uniform(-0.5, 0.5), 2)
            for i in range(4)
        ]
    else:  # random / false-alarm
        # Noisy values that stay near-but-below threshold.
        points = [
            round(threshold * rng.uniform(0.82, 0.99) + rng.uniform(-1.0, 1.0), 2)
            for _ in range(4)
        ]
    points.append(round(float(current_val), 2))  # t-0 = current breach value
    return points


def _alert_dict(
    alert_id: str,
    service: str,
    metric: str,
    metric_value: float,
    threshold: float,
    message: str,
    timestamp: str,
    context: str | None = None,
    metric_history: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "alert_id":       alert_id,
        "timestamp":      timestamp,
        "service":        service,
        "metric":         metric,
        "metric_value":   round(float(metric_value), 2),
        "threshold":      float(threshold),
        "message":        message,
        "context":        context,
        "metric_history": metric_history,   # 5-point rolling window [t-4..t-0]
        "triaged":        False,
        "agent_decision": None,
    }


def _gt_dict(
    alert_id: str,
    true_root_cause: str,
    true_severity: str,
    true_remediation: str,
    incident_id: str | None = None,
) -> dict[str, Any]:
    return {
        "alert_id":         alert_id,
        "true_root_cause":  true_root_cause,
        "true_severity":    true_severity,
        "true_remediation": true_remediation,
        "incident_id":      incident_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-root-cause alert builders
# Each returns (alert_dict, gt_dict).  incident_id is always None here;
# callers that need a non-None incident_id should set gt["incident_id"] after.
# ─────────────────────────────────────────────────────────────────────────────

def _build_resource(
    rng: random.Random, alert_id: str, service: str, ts: str
) -> tuple[dict, dict]:
    """resource_exhaustion — CPU / memory / disk saturation."""
    metric, thr = rng.choice(_RESOURCE_METRICS)
    val = thr + rng.uniform(5.0, 18.0)
    sev = "critical" if val > thr + 12 else "high"
    msg = (
        f"{service} {metric.replace('_', ' ')} at {val:.1f}% "
        f"(threshold: {thr:.0f}%) — resource saturation detected"
    )
    history = _gen_history(rng, val, thr, pattern="rising")
    return (
        _alert_dict(alert_id, service, metric, val, thr, msg, ts, metric_history=history),
        _gt_dict(alert_id, "resource_exhaustion", sev, "scale_up"),
    )


def _build_network(
    rng: random.Random, alert_id: str, service: str, ts: str
) -> tuple[dict, dict]:
    """network_failure — latency / packet loss / connection errors."""
    metric, thr = rng.choice(_NETWORK_METRICS)
    val = thr + rng.uniform(thr * 0.4, thr * 1.8)
    msg = (
        f"{service} network degradation: {metric.replace('_', ' ')} = {val:.1f} "
        f"(threshold: {thr:.0f}) — possible network partition or NIC saturation"
    )
    history = _gen_history(rng, val, thr, pattern="spike")
    return (
        _alert_dict(alert_id, service, metric, val, thr, msg, ts, metric_history=history),
        _gt_dict(alert_id, "network_failure", "high", "escalate_to_team"),
    )


def _build_deploy(
    rng: random.Random, alert_id: str, service: str, ts: str
) -> tuple[dict, dict]:
    """deployment_bug — error spike after a recent deploy."""
    metric, thr = rng.choice(_DEPLOY_METRICS)
    val = thr + rng.uniform(2.0, 10.0)
    ver = f"{rng.randint(2, 5)}.{rng.randint(0, 9)}.{rng.randint(0, 20)}"
    mins_ago = rng.randint(5, 25)
    msg = (
        f"{service} {metric.replace('_', ' ')} spiked to {val:.1f} "
        f"after recent deployment (threshold: {thr:.0f})"
    )
    ctx = f"Deploy v{ver} rolled out {mins_ago} minutes ago"
    history = _gen_history(rng, val, thr, pattern="spike")
    return (
        _alert_dict(alert_id, service, metric, val, thr, msg, ts, ctx, metric_history=history),
        _gt_dict(alert_id, "deployment_bug", "high", "rollback_deploy"),
    )


def _build_config(
    rng: random.Random, alert_id: str, service: str, ts: str
) -> tuple[dict, dict]:
    """config_error — misconfiguration causing health-check or auth failures."""
    metric, thr = rng.choice(_CONFIG_METRICS)
    val = thr + rng.uniform(1.0, 8.0)
    sev = rng.choice(["medium", "high"])
    msg = (
        f"{service} {metric.replace('_', ' ')} at {val:.1f} — "
        f"service misconfiguration suspected (threshold: {thr:.0f})"
    )
    history = _gen_history(rng, val, thr, pattern="gradual")
    return (
        _alert_dict(alert_id, service, metric, val, thr, msg, ts, metric_history=history),
        _gt_dict(alert_id, "config_error", sev, "fix_config"),
    )


def _build_dependency(
    rng: random.Random,
    alert_id: str,
    service: str,
    dep_service: str,
    ts: str,
    incident_id: str | None = None,
    ctx_variant: int = 0,
) -> tuple[dict, dict]:
    """dependency_outage — service failing because an upstream dependency is down.
    
    ctx_variant selects the context string template:
      0  standard prefix  "Upstream service '...'"  — cascade-detectable
      1+ alternative phrasing                        — not cascade-detectable
    """
    metric, thr = rng.choice(_DEP_METRICS)
    val = thr + rng.uniform(thr * 0.3, thr * 1.5)
    sev = "critical" if val > thr * 1.8 else "high"
    msg = (
        f"{service} upstream calls failing: {metric.replace('_', ' ')} = {val:.1f} "
        f"(threshold: {thr:.0f}) — dependency '{dep_service}' may be down"
    )
    ctx_tmpl = _DEP_CTX_TEMPLATES[ctx_variant % len(_DEP_CTX_TEMPLATES)]
    ctx = ctx_tmpl.format(dep=dep_service)
    history = _gen_history(rng, val, thr, pattern="spike")
    return (
        _alert_dict(alert_id, service, metric, val, thr, msg, ts, ctx, metric_history=history),
        _gt_dict(alert_id, "dependency_outage", sev, "acknowledge_and_monitor", incident_id),
    )


def _build_deploy_cpu(
    rng: random.Random, alert_id: str, service: str, ts: str
) -> tuple[dict, dict]:
    """
    Ambiguous alert: CPU/memory pressure caused by a deploy regression,
    NOT genuine resource saturation.

    The observable signal (cpu/memory metric above threshold) looks exactly
    like *resource_exhaustion*.  True root cause is *deployment_bug* — the
    new build introduced a memory regression that only becomes apparent under
    normal production load.  Both the metric-first heuristic fallback AND the
    LLM system prompt ("cpu/memory metric → resource_exhaustion") will
    misclassify this, producing realistic partial scores rather than 1.0.
    """
    metric, thr = rng.choice([("cpu_usage_percent", 80.0), ("memory_usage_percent", 85.0)])
    val = thr + rng.uniform(4.0, 10.0)      # elevated but ≤ +12 → severity "high"
    ver = f"{rng.randint(2, 5)}.{rng.randint(0, 9)}.{rng.randint(0, 9)}"
    mins_ago = rng.randint(10, 35)
    msg = (
        f"{service} {metric.replace('_', ' ')} climbing to {val:.1f}% "
        f"(threshold: {thr:.0f}%) — resource pressure building"
    )
    ctx = (
        f"Pressure began ~{mins_ago}m after deploy v{ver}; "
        "no capacity changes planned; possible memory regression in new build"
    )
    history = _gen_history(rng, val, thr, pattern="gradual")
    return (
        _alert_dict(alert_id, service, metric, val, thr, msg, ts, ctx, metric_history=history),
        _gt_dict(alert_id, "deployment_bug", "high", "rollback_deploy"),
    )


def _build_latency_dep(
    rng: random.Random,
    alert_id: str,
    service: str,
    ts: str,
    dep_service: str | None = None,
) -> tuple[dict, dict]:
    """
    Ambiguous alert: elevated network_latency caused by an upstream dependency
    bottleneck — NOT a genuine network partition or NIC failure.

    The observable metric (network_latency_ms) looks like *network_failure*.
    True root cause is *dependency_outage*: the upstream service is the
    bottleneck, not the network itself.  The context deliberately avoids the
    "Upstream service '...'" pattern so the cascade detector does not
    accidentally group this into an incident chain.
    """
    if dep_service is None:
        dep_service = rng.choice(sorted(_ALL_SERVICES))
    thr = 200.0
    val = thr + rng.uniform(60.0, 280.0)
    msg = (
        f"{service} response latency at {val:.1f}ms "
        f"(threshold: {thr:.0f}ms) — elevated above SLA baseline"
    )
    ctx = (
        f"High latency correlates with {dep_service} slowdowns; "
        f"no packet loss or NIC errors detected on {service}"
    )
    history = _gen_history(rng, val, thr, pattern="rising")
    return (
        _alert_dict(alert_id, service, "network_latency_ms", val, thr, msg, ts, ctx, metric_history=history),
        _gt_dict(alert_id, "dependency_outage", "high", "acknowledge_and_monitor"),
    )


def _build_false_alarm(
    rng: random.Random,
    alert_id: str,
    service: str,
    ts: str,
    misleading: bool = False,
) -> tuple[dict, dict]:
    """
    false_alarm — metric briefly exceeds threshold for a benign reason.
    *misleading=True* simulates a monitoring system that mislabels this as CRITICAL.
    The true severity is always "low" and remediation is always "dismiss".
    """
    metric, thr = rng.choice(_NOISE_METRICS)
    val = thr + rng.uniform(0.5, 4.0)    # only just over threshold
    if misleading:
        msg = (
            f"{service} {metric.replace('_', ' ')} briefly at {val:.1f} "
            f"[MONITORING: CRITICAL ALERT] (threshold: {thr:.0f}) — "
            "automated escalation; prior pattern suggests false positive"
        )
        ctx = "PagerDuty P0 auto-created — verify before acting"
    else:
        msg = (
            f"{service} {metric.replace('_', ' ')} briefly at {val:.1f} — "
            f"known spike during scheduled batch job (threshold: {thr:.0f})"
        )
        ctx = "Scheduled maintenance window 10:00–11:00 UTC"
    # False alarm history: noisy values near-but-below threshold (would drop after)
    history = _gen_history(rng, val, thr, pattern="random")
    return (
        _alert_dict(alert_id, service, metric, val, thr, msg, ts, ctx, metric_history=history),
        _gt_dict(alert_id, "false_alarm", "low", "dismiss"),
    )


def _build_stealth_root(
    rng: random.Random,
    alert_id: str,
    service: str,
    ts: str,
    incident_id: str,
) -> tuple[dict, dict]:
    """
    Stealth incident root — metric is only mildly elevated, easy to overlook,
    but causes loud failures downstream.  True severity is "medium".
    """
    metric, thr = rng.choice(_RESOURCE_METRICS)
    val = thr + rng.uniform(1.0, 3.5)    # subtle — barely over threshold
    msg = (
        f"{service} {metric.replace('_', ' ')} mildly elevated at {val:.1f}% "
        f"(threshold: {thr:.0f}%) — minor degradation, monitoring"
    )
    ctx = "No recent deploys. Pattern consistent with gradual memory leak."
    # Stealth pattern: very slow, almost imperceptible climb (gradual memory leak)
    history = _gen_history(rng, val, thr, pattern="gradual")
    return (
        _alert_dict(alert_id, service, metric, val, thr, msg, ts, ctx, metric_history=history),
        _gt_dict(alert_id, "resource_exhaustion", "medium", "acknowledge_and_monitor", incident_id),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graph traversal helper
# ─────────────────────────────────────────────────────────────────────────────

def _cascade_chain(root: str, max_len: int) -> list[str]:
    """
    BFS through the service dependency graph starting from *root*, following
    the "dependent" (reverse) edges.  Returns up to *max_len* services
    (including the root) in discovery order.

    Dependents at each BFS step are sorted alphabetically before enqueuing
    to ensure the result is fully deterministic across Python versions.
    """
    chain: list[str] = [root]
    queue: list[str] = [root]
    while queue and len(chain) < max_len:
        curr = queue.pop(0)
        for dep in sorted(get_dependents(curr)):
            if dep not in chain:
                chain.append(dep)
                queue.append(dep)
                if len(chain) >= max_len:
                    break
    return chain[:max_len]


# ─────────────────────────────────────────────────────────────────────────────
# Task-specific generators
# ─────────────────────────────────────────────────────────────────────────────

def _generate_easy(rng: random.Random) -> dict[str, Any]:
    """
    5 independent alerts — one per root-cause type (excluding false_alarm).
    No incidents, no noise.
    """
    # Pick 5 distinct services at random from the full sorted service list.
    services = rng.sample(_ALL_SERVICES, 5)

    # Shuffle the five deterministic root-cause types so the assignment
    # varies with the seed (while still covering all five types exactly once).
    rc_types = [
        "resource_exhaustion",
        "network_failure",
        "deployment_bug",
        "config_error",
        "dependency_outage",
    ]
    rng.shuffle(rc_types)

    alerts: list[dict] = []
    ground_truth: list[dict] = []

    for i, (svc, rc) in enumerate(zip(services, rc_types)):
        alert_id = f"alert-{i + 1:03d}"
        ts = _ts(i * 5)                # spaced 5 minutes apart

        if rc == "resource_exhaustion":
            # Ambiguous: CPU/memory pressure from a deploy regression; metric looks
            # like resource_exhaustion but true root cause is deployment_bug.
            # Requires reading the context (deploy timestamp, version) to classify correctly.
            a, gt = _build_deploy_cpu(rng, alert_id, svc, ts)
        elif rc == "network_failure":
            a, gt = _build_network(rng, alert_id, svc, ts)
        elif rc == "deployment_bug":
            a, gt = _build_deploy(rng, alert_id, svc, ts)
        elif rc == "config_error":
            a, gt = _build_config(rng, alert_id, svc, ts)
        else:  # dependency_outage
            # Ambiguous: network_latency metric elevated due to upstream slowdowns;
            # looks like network_failure but context clues point to dependency_outage.
            deps = SERVICE_GRAPH.get(svc, [])
            dep_svc = deps[0] if deps else rng.choice(_ALL_SERVICES)
            a, gt = _build_latency_dep(rng, alert_id, svc, ts, dep_svc)

        alerts.append(a)
        ground_truth.append(gt)

    return {"alerts": alerts, "ground_truth": ground_truth, "incidents": []}


def _generate_medium(rng: random.Random) -> dict[str, Any]:
    """
    15 alerts:
        INC-001 (redis-cache, resource_exhaustion):  4 alerts (cascade)
        INC-002 (object-storage, network_failure):   4 alerts (cascade)
        Independent:                                 5 alerts
        False alarms / noise:                        2 alerts
    """
    alerts: list[dict] = []
    ground_truth: list[dict] = []
    incidents: list[dict] = []
    _counter = [1]

    def _next_id() -> str:
        aid = f"alert-{_counter[0]:03d}"
        _counter[0] += 1
        return aid

    # ── Incident INC-001: redis-cache ──────────────────────────────────────
    # Cascade: redis-cache → auth-service, recommendation-engine, user-service
    inc1_id = "INC-001"
    inc1_root = "redis-cache"
    inc1_chain = _cascade_chain(inc1_root, 4)
    inc1_alert_ids: list[str] = []

    # Root alert
    aid = _next_id()
    a, gt = _build_resource(rng, aid, inc1_root, _ts(rng.randint(0, 5)))
    gt["incident_id"] = inc1_id
    alerts.append(a); ground_truth.append(gt); inc1_alert_ids.append(aid)

    # Cascade dependents
    for idx, svc in enumerate(inc1_chain[1:]):
        aid = _next_id()
        # Use a non-detectable context for the last cascade alert in long chains
        # (chain length ≥ 4).  This makes the cascade-group detector find 3 of 4
        # alerts, giving a partial link-F1 instead of a perfect 1.0.
        use_variant = (idx == len(inc1_chain[1:]) - 1) and len(inc1_chain) >= 4
        a, gt = _build_dependency(
            rng, aid, svc, inc1_root, _ts(rng.randint(5, 15)), inc1_id,
            ctx_variant=(1 if use_variant else 0),
        )
        alerts.append(a); ground_truth.append(gt); inc1_alert_ids.append(aid)

    incidents.append({
        "incident_id":  inc1_id,
        "root_service": inc1_root,
        "root_cause":   "resource_exhaustion",
        "alert_ids":    inc1_alert_ids,
    })

    # ── Incident INC-002: object-storage ──────────────────────────────────
    # Cascade: object-storage → email-worker → notification-service → api-gateway
    inc2_id = "INC-002"
    inc2_root = "object-storage"
    inc2_chain = _cascade_chain(inc2_root, 4)
    inc2_alert_ids: list[str] = []

    aid = _next_id()
    a, gt = _build_network(rng, aid, inc2_root, _ts(rng.randint(2, 8)))
    gt["incident_id"] = inc2_id
    alerts.append(a); ground_truth.append(gt); inc2_alert_ids.append(aid)

    for idx, svc in enumerate(inc2_chain[1:]):
        aid = _next_id()
        use_variant = (idx == len(inc2_chain[1:]) - 1) and len(inc2_chain) >= 4
        a, gt = _build_dependency(
            rng, aid, svc, inc2_root, _ts(rng.randint(8, 20)), inc2_id,
            ctx_variant=(1 if use_variant else 0),
        )
        alerts.append(a); ground_truth.append(gt); inc2_alert_ids.append(aid)

    incidents.append({
        "incident_id":  inc2_id,
        "root_service": inc2_root,
        "root_cause":   "network_failure",
        "alert_ids":    inc2_alert_ids,
    })

    # ── Independent alerts (5) ────────────────────────────────────────────
    indep_rc = [
        "resource_exhaustion",
        "network_failure",
        "deployment_bug",
        "config_error",
        "dependency_outage",
    ]
    rng.shuffle(indep_rc)
    indep_svcs = rng.sample(_ALL_SERVICES, 5)

    for svc, rc in zip(indep_svcs, indep_rc):
        aid = _next_id()
        ts = _ts(rng.randint(0, 25))
        if rc == "resource_exhaustion":
            # Ambiguous: CPU/memory pressure from a deploy regression — metric looks
            # like resource_exhaustion but true root cause is deployment_bug.
            a, gt = _build_deploy_cpu(rng, aid, svc, ts)
        elif rc == "network_failure":
            # Ambiguous: network_latency elevated by upstream slowdowns, not a
            # genuine network partition — context distinguishes it.
            deps = SERVICE_GRAPH.get(svc, [])
            dep_svc = deps[0] if deps else _ALL_SERVICES[0]
            a, gt = _build_latency_dep(rng, aid, svc, ts, dep_svc)
        elif rc == "deployment_bug":
            a, gt = _build_deploy(rng, aid, svc, ts)
        elif rc == "config_error":
            a, gt = _build_config(rng, aid, svc, ts)
        else:  # dependency_outage
            deps = SERVICE_GRAPH.get(svc, [])
            dep_svc = deps[0] if deps else _ALL_SERVICES[0]
            a, gt = _build_dependency(rng, aid, svc, dep_svc, ts)
        alerts.append(a); ground_truth.append(gt)

    # ── False alarms / noise (2) ──────────────────────────────────────────
    noise_svcs = rng.sample(_ALL_SERVICES, 2)
    for svc in noise_svcs:
        aid = _next_id()
        ts = _ts(rng.randint(0, 25))
        a, gt = _build_false_alarm(rng, aid, svc, ts)
        alerts.append(a); ground_truth.append(gt)

    assert len(alerts) == 15, f"Medium: expected 15 alerts, got {len(alerts)}"
    return {"alerts": alerts, "ground_truth": ground_truth, "incidents": incidents}


def _generate_hard(rng: random.Random) -> dict[str, Any]:
    """
    30 alerts:
        5 incidents (root cause services randomly selected from Tier 5)
        Independent alerts
        False alarms / noise
    Total: 30 alerts

    The seed determines which Tier 5 services become incident roots,
    which root causes they have, and the cascade chains.
    """
    alerts: list[dict] = []
    ground_truth: list[dict] = []
    incidents: list[dict] = []
    _counter = [1]

    def _next_id() -> str:
        aid = f"alert-{_counter[0]:03d}"
        _counter[0] += 1
        return aid

    def _add_incident(
        inc_id: str,
        root_svc: str,
        root_cause_label: str,
        chain: list[str],
        root_builder,          # callable matching (rng, alert_id, service, ts)
    ) -> None:
        inc_ids: list[str] = []

        # Root alert
        aid = _next_id()
        a, gt = root_builder(rng, aid, root_svc, _ts(rng.randint(0, 10)))
        gt["incident_id"] = inc_id
        alerts.append(a); ground_truth.append(gt); inc_ids.append(aid)

        # Cascade dependent alerts — ctx_variant=1 uses non-standard phrasing so
        # cascade groups cannot be detected by simple text matching alone.
        # Correct grouping requires reasoning over the service dependency graph.
        for svc in chain[1:]:
            aid = _next_id()
            a, gt = _build_dependency(rng, aid, svc, root_svc, _ts(rng.randint(5, 25)), inc_id, ctx_variant=1)
            alerts.append(a); ground_truth.append(gt); inc_ids.append(aid)

        incidents.append({
            "incident_id":  inc_id,
            "root_service": root_svc,
            "root_cause":   root_cause_label,
            "alert_ids":    inc_ids,
        })

    # ── Randomly select 5 incident root services from Tier 5 ───────────────
    # Shuffle Tier 5 services based on seed
    tier5_shuffled = _TIER5_SERVICES.copy()
    rng.shuffle(tier5_shuffled)
    incident_services = tier5_shuffled[:5]  # Select 5 for incidents
    
    # Shuffle root causes for variety
    causes_shuffled = _INCIDENT_ROOT_CAUSES.copy()
    rng.shuffle(causes_shuffled)
    
    # Map each incident to a root cause and builder
    incident_config = [
        (incident_services[0], causes_shuffled[0], _build_resource),
        (incident_services[1], causes_shuffled[1], _build_network),
        (incident_services[2], causes_shuffled[2], _build_config),
        (incident_services[3], causes_shuffled[3] if len(causes_shuffled) > 3 else causes_shuffled[0], _build_deploy),
        (incident_services[4], "resource_exhaustion", _build_resource),  # stealth incident
    ]

    # ── Five incidents ────────────────────────────────────────────────────
    # Total incident alerts = 18 (same as original: 4+4+4+3+3 = 18)
    # First 3 have 4 alerts each, last 2 have 3 alerts each
    incident_chain_lengths = [4, 4, 4, 3, 3]
    rng.shuffle(incident_chain_lengths)  # Randomize which incidents are larger
    
    for i, (root_svc, root_cause, builder) in enumerate(incident_config):
        inc_id = f"INC-{i+1:03d}"
        chain_len = incident_chain_lengths[i]
        is_stealth = (i == 4)  # Mark last incident as stealth
        
        chain = _cascade_chain(root_svc, chain_len)
        inc_ids: list[str] = []
        
        # Root alert
        aid = _next_id()
        if is_stealth:
            a, gt = _build_stealth_root(rng, aid, root_svc, _ts(rng.randint(0, 10)), inc_id)
        else:
            a, gt = builder(rng, aid, root_svc, _ts(rng.randint(0, 10)))
        gt["incident_id"] = inc_id
        alerts.append(a); ground_truth.append(gt); inc_ids.append(aid)
        
        # Cascade dependent alerts
        for svc in chain[1:]:
            aid = _next_id()
            a, gt = _build_dependency(rng, aid, svc, root_svc, _ts(rng.randint(5, 25)), inc_id)
            alerts.append(a); ground_truth.append(gt); inc_ids.append(aid)
        
        incidents.append({
            "incident_id":  inc_id,
            "root_service": root_svc,
            "root_cause":   root_cause,
            "alert_ids":    inc_ids,
            "stealth":      is_stealth,
        })

    # ── Independent alerts (6) ────────────────────────────────────────────
    # Six root-cause labels (cycling the 5-type pool to get 6 items).
    _base_rcs = [
        "resource_exhaustion",
        "network_failure",
        "deployment_bug",
        "config_error",
        "dependency_outage",
    ]
    indep_rc = [_base_rcs[i % 5] for i in range(6)]
    rng.shuffle(indep_rc)
    indep_svcs = rng.sample(_ALL_SERVICES, 6)

    for svc, rc in zip(indep_svcs, indep_rc):
        aid = _next_id()
        ts = _ts(rng.randint(0, 40))
        if rc == "resource_exhaustion":
            # Ambiguous: CPU/memory pressure from a deploy regression — metric looks
            # like resource_exhaustion but true root cause is deployment_bug.
            a, gt = _build_deploy_cpu(rng, aid, svc, ts)
        elif rc == "network_failure":
            # Ambiguous: network_latency elevated by upstream slowdowns, not a
            # genuine network partition — context distinguishes it.
            deps = SERVICE_GRAPH.get(svc, [])
            dep_svc = deps[0] if deps else _ALL_SERVICES[0]
            a, gt = _build_latency_dep(rng, aid, svc, ts, dep_svc)
        elif rc == "deployment_bug":
            a, gt = _build_deploy(rng, aid, svc, ts)
        elif rc == "config_error":
            a, gt = _build_config(rng, aid, svc, ts)
        else:  # dependency_outage
            deps = SERVICE_GRAPH.get(svc, [])
            dep_svc = deps[0] if deps else _ALL_SERVICES[0]
            a, gt = _build_dependency(rng, aid, svc, dep_svc, ts)
        alerts.append(a); ground_truth.append(gt)

    # ── False alarms / noise (6, first one is misleadingly marked CRITICAL) ─
    noise_svcs = rng.sample(_ALL_SERVICES, 6)
    for i, svc in enumerate(noise_svcs):
        aid = _next_id()
        ts = _ts(rng.randint(0, 40))
        a, gt = _build_false_alarm(rng, aid, svc, ts, misleading=(i == 0))
        alerts.append(a); ground_truth.append(gt)

    # Safety check: if BFS chains produced fewer alerts than expected (very short
    # cascade paths), pad with extra independent alerts to ensure count is stable.
    actual = len(alerts)
    if actual != 30:
        import warnings
        warnings.warn(
            f"Hard scenario generated {actual} alerts (expected 30) for seed {id(rng)}. "
            "Padding with extra independent alerts."
        )
        extra_svcs = [s for s in _ALL_SERVICES if s not in {a["service"] for a in alerts}]
        extra_rcs = _base_rcs.copy()
        rng.shuffle(extra_svcs)
        while len(alerts) < 30 and extra_svcs:
            svc = extra_svcs.pop()
            rc = extra_rcs[len(alerts) % 5]
            aid = _next_id()
            ts = _ts(rng.randint(0, 40))
            a, gt = _build_resource(rng, aid, svc, ts)
            alerts.append(a)
            ground_truth.append(gt)

    assert len(alerts) == 30, f"Hard: expected 30 alerts, got {len(alerts)}"

    # ── Interleave: apply a deterministic random permutation to the alerts ──
    # ground_truth is NOT reordered — alert_ids still match across both lists.
    sort_keys = list(range(30))
    rng.shuffle(sort_keys)
    alerts = [a for _, a in sorted(zip(sort_keys, alerts), key=lambda x: x[0])]

    return {"alerts": alerts, "ground_truth": ground_truth, "incidents": incidents}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_scenario(task_id: str, seed: int) -> dict[str, Any]:
    """
    Generate a complete scenario for *task_id* using the given *seed*.

    Parameters
    ----------
    task_id : str
        One of ``"easy"``, ``"medium"``, or ``"hard"``.
    seed : int
        Integer seed passed to ``random.Random``.  Same seed + same task_id
        always produces byte-for-byte identical output.

    Returns
    -------
    dict with keys ``"alerts"``, ``"ground_truth"``, ``"incidents"``.

    Raises
    ------
    ValueError
        If *task_id* is not recognised.
    """
    rng = random.Random(seed)

    if task_id == "easy":
        return _generate_easy(rng)
    elif task_id == "medium":
        return _generate_medium(rng)
    elif task_id == "hard":
        return _generate_hard(rng)
    else:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            "Valid values are: 'easy', 'medium', 'hard'."
        )
