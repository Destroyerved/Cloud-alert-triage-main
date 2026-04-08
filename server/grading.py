from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

from server.config import SEVERITY_ORDER

# ─────────────────────────────────────────────────────────────────────────────
# Per-task weights
# ─────────────────────────────────────────────────────────────────────────────

_WEIGHTS: dict[str, dict[str, float]] = {
    "easy":   {"rc": 0.38, "sev": 0.28, "rem": 0.28, "link": 0.00, "fa": 0.00, "eff": 0.03, "ord": 0.03},
    "medium": {"rc": 0.28, "sev": 0.20, "rem": 0.20, "link": 0.20, "fa": 0.07, "eff": 0.02, "ord": 0.03},
    "hard":   {"rc": 0.28, "sev": 0.20, "rem": 0.17, "link": 0.18, "fa": 0.09, "eff": 0.04, "ord": 0.04},
}

_STEALTH_BONUS: dict[str, float] = {
    "easy":   0.00,
    "medium": 0.00,
    "hard":   0.10,
}

_EFFICIENCY_FLOOR: float = 0.20

# Root-cause and remediation partial credit
_RC_RELATED_PAIRS: set[frozenset[str]] = {
    frozenset(("resource_exhaustion", "deployment_bug")),
    frozenset(("network_failure", "dependency_outage")),
}
_RC_PARTIAL_CREDIT: float = 0.60

_REM_RELATED_PAIRS: set[frozenset[str]] = {
    frozenset(("scale_up", "rollback_deploy")),
    frozenset(("escalate_to_team", "acknowledge_and_monitor")),
}
_REM_PARTIAL_CREDIT: float = 0.40

# Hard bounds for the final score — strictly open interval (0, 1)
_SCORE_MIN: float = 0.0001
_SCORE_MAX: float = 0.9999


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def grade_episode(
    task_id: str,
    final_state_dict: dict[str, Any],
) -> float:
    """
    Compute a grader score guaranteed to be in the open interval (0, 1).

    The return value is ALWAYS in [0.0001, 0.9999] — never exactly 0 or 1.
    This invariant is enforced by the final clamp regardless of what component
    scorers return.
    """
    try:
        return _grade_episode_inner(task_id, final_state_dict)
    except Exception:
        # Any unexpected exception returns the safe midpoint rather than
        # propagating and leaving the environment without a grader_score.
        return 0.5


def _grade_episode_inner(
    task_id: str,
    final_state_dict: dict[str, Any],
) -> float:
    if task_id not in _WEIGHTS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Valid values: {sorted(_WEIGHTS.keys())}"
        )

    all_ground_truth: list[dict[str, Any]] = final_state_dict.get("ground_truth", [])
    dynamic_ids: set[str] = set(final_state_dict.get("dynamic_alert_ids", set()))
    incidents: list[dict[str, Any]] = final_state_dict.get("incidents", [])
    agent_links: list[dict[str, Any]] = final_state_dict.get("agent_links", [])
    agent_decisions: list[dict[str, Any]] = final_state_dict.get("agent_decisions", [])
    triage_order: list[str] = final_state_dict.get("triage_order", [])
    steps_used: int = int(final_state_dict.get("step_number", 0))
    max_steps: int = max(1, int(final_state_dict.get("max_steps", 1)))

    # Filter out dynamic alerts from grading
    ground_truth: list[dict[str, Any]] = [
        gt for gt in all_ground_truth
        if gt["alert_id"] not in dynamic_ids
    ]

    # Separate triageable alerts from false alarms
    triageable_gt: list[dict[str, Any]] = [
        gt for gt in ground_truth if gt.get("true_root_cause") != "false_alarm"
    ]

    decisions_by_id: dict[str, dict[str, Any]] = {
        d["alert_id"]: d
        for d in agent_decisions
        if d.get("action_type") == "triage"
    }

    skips_by_id: set[str] = {
        d["alert_id"]
        for d in agent_decisions
        if d.get("action_type") == "skip"
    }

    original_ids: set[str] = {gt["alert_id"] for gt in ground_truth}

    w = _WEIGHTS[task_id]

    # ── Component scores — each clamped to [0, 1] defensively ───────────────
    rc_score   = _clamp01(_root_cause_accuracy(decisions_by_id, triageable_gt))
    sev_score  = _clamp01(_severity_accuracy(decisions_by_id, triageable_gt))
    rem_score  = _clamp01(_remediation_accuracy(decisions_by_id, triageable_gt))
    link_score = _clamp01(_incident_link_f1(agent_links, ground_truth))
    fa_score   = _clamp01(_false_alarm_accuracy(decisions_by_id, skips_by_id, ground_truth))
    eff_score  = _clamp01(_efficiency_score(steps_used, max_steps))
    ord_score  = _clamp01(_triage_ordering_score(triage_order, ground_truth, dynamic_ids))

    base_score = (
        w["rc"]  * rc_score +
        w["sev"] * sev_score +
        w["rem"] * rem_score +
        w["link"] * link_score +
        w["fa"]  * fa_score +
        w["eff"] * eff_score +
        w["ord"] * ord_score
    )

    # Coverage penalty
    handled_original = sum(
        1 for aid in original_ids
        if aid in decisions_by_id or aid in skips_by_id
    )
    n_total = max(1, len(ground_truth))
    coverage = handled_original / n_total
    coverage_penalty = coverage ** 1.5

    score = base_score * coverage_penalty

    # Stealth bonus (hard only)
    bonus = _STEALTH_BONUS.get(task_id, 0.0)
    if bonus > 0:
        score += bonus * _stealth_bonus(decisions_by_id, ground_truth, incidents)

    # ── Strict (0, 1) enforcement ────────────────────────────────────────────
    # 1. Replace NaN / Inf with safe midpoint
    if not math.isfinite(score):
        score = 0.5
    # 2. Clamp to open interval — NEVER returns 0.0 or 1.0
    return round(max(_SCORE_MIN, min(_SCORE_MAX, score)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Component scorers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp01(value: float) -> float:
    """Clamp a component score to [0.0, 1.0] and replace NaN with 0."""
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _root_cause_accuracy(decisions_by_id, ground_truth) -> float:
    if not ground_truth:
        return 1.0
    total = 0.0
    for gt in ground_truth:
        agent_rc = decisions_by_id.get(gt["alert_id"], {}).get("root_cause")
        true_rc = gt.get("true_root_cause", "")
        if agent_rc == true_rc:
            total += 1.0
        elif agent_rc and frozenset((agent_rc, true_rc)) in _RC_RELATED_PAIRS:
            total += _RC_PARTIAL_CREDIT
    return total / len(ground_truth)


def _severity_accuracy(decisions_by_id, ground_truth) -> float:
    if not ground_truth:
        return 1.0
    total = 0.0
    for gt in ground_truth:
        decision = decisions_by_id.get(gt["alert_id"])
        if decision is None:
            continue
        agent_sev = decision.get("severity", "")
        true_sev = gt.get("true_severity", "")
        if agent_sev == true_sev:
            total += 1.0
        else:
            agent_rank = SEVERITY_ORDER.get(agent_sev, 2)
            true_rank = SEVERITY_ORDER.get(true_sev, 2)
            distance = abs(agent_rank - true_rank)
            if distance == 1:
                total += 0.50
            elif distance == 2:
                total += 0.15
    return total / len(ground_truth)


def _remediation_accuracy(decisions_by_id, ground_truth) -> float:
    if not ground_truth:
        return 1.0
    total = 0.0
    for gt in ground_truth:
        agent_rem = decisions_by_id.get(gt["alert_id"], {}).get("remediation")
        true_rem = gt.get("true_remediation", "")
        if agent_rem == true_rem:
            total += 1.0
        elif agent_rem and frozenset((agent_rem, true_rem)) in _REM_RELATED_PAIRS:
            total += _REM_PARTIAL_CREDIT
    return total / len(ground_truth)


def _incident_link_f1(agent_links, ground_truth) -> float:
    true_groups = defaultdict(list)
    for gt in ground_truth:
        inc_id = gt.get("incident_id")
        if inc_id is not None:
            true_groups[inc_id].append(gt["alert_id"])
    true_pairs = _pairs_from_groups(
        [ids for ids in true_groups.values() if len(ids) >= 2]
    )
    if not true_pairs:
        return 1.0
    agent_pairs = _pairs_from_groups(
        [link["alert_ids"] for link in agent_links if link.get("alert_ids")]
    )
    if not agent_pairs:
        return 0.0
    tp = len(true_pairs & agent_pairs)
    precision = tp / len(agent_pairs) if agent_pairs else 0.0
    recall = tp / len(true_pairs) if true_pairs else 0.0
    denom = precision + recall
    return (2 * precision * recall / denom) if denom > 0 else 0.0


def _false_alarm_accuracy(decisions_by_id, skips_by_id, ground_truth) -> float:
    fa_alerts = [gt for gt in ground_truth if gt.get("true_root_cause") == "false_alarm"]
    if not fa_alerts:
        return 1.0
    real_alerts = [gt for gt in ground_truth if gt.get("true_root_cause") != "false_alarm"]
    correctly_skipped_fa = sum(1 for gt in fa_alerts if gt["alert_id"] in skips_by_id)
    correctly_triaged_real = sum(1 for gt in real_alerts if gt["alert_id"] in decisions_by_id)
    total = max(1, len(ground_truth))
    base = (correctly_skipped_fa + correctly_triaged_real) / total
    skip_ratio = len(skips_by_id) / total
    penalty = max(0.0, 1.0 - skip_ratio * 0.5)
    return base * penalty


def _efficiency_score(steps_used: int, max_steps: int) -> float:
    if max_steps <= 0:
        return _EFFICIENCY_FLOOR
    raw = 1.0 - (steps_used / max_steps)
    return max(_EFFICIENCY_FLOOR, raw)


def _triage_ordering_score(
    triage_order: list[str],
    ground_truth: list[dict[str, Any]],
    dynamic_ids: set[str],
) -> float:
    sev_rank: dict[str, int] = {}
    for gt in ground_truth:
        if gt["alert_id"] not in dynamic_ids:
            sev_rank[gt["alert_id"]] = SEVERITY_ORDER.get(gt.get("true_severity", "low"), 3)
    ordered = [aid for aid in triage_order if aid in sev_rank]
    if len(ordered) < 2:
        return 1.0
    concordant = 0
    total = 0
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            total += 1
            if sev_rank[ordered[i]] <= sev_rank[ordered[j]]:
                concordant += 1
    return concordant / total if total > 0 else 1.0


def _stealth_bonus(decisions_by_id, ground_truth, incidents) -> float:
    stealth_inc = next(
        (inc for inc in incidents if inc.get("stealth")), None
    )
    if stealth_inc is None:
        return 0.0
    stealth_id = stealth_inc.get("incident_id") or stealth_inc.get("id")
    if stealth_id is None:
        return 0.0
    stealth_alerts = [
        gt for gt in ground_truth if gt.get("incident_id") == stealth_id
    ]
    for gt in stealth_alerts:
        decision = decisions_by_id.get(gt["alert_id"])
        if decision and decision.get("root_cause") == gt.get("true_root_cause"):
            return 1.0
    return 0.0


def _pairs_from_groups(groups):
    pairs = set()
    for group in groups:
        for a, b in itertools.combinations(group, 2):
            pairs.add(frozenset((a, b)))
    return pairs
