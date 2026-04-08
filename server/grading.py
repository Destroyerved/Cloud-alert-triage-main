from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any

from server.config import SEVERITY_ORDER

# ─────────────────────────────────────────────────────────────────────────────
# Per-task weights
#
# Two operational-quality dimensions ("eff", "ord") ensure grader scores land
# in (0, 1) rather than exactly 0 or 1.  Their small weights act as
# tie-breakers: classification accuracy dominates, but efficiency and ordering
# differentiate agents of similar correctness.
# ─────────────────────────────────────────────────────────────────────────────

_WEIGHTS: dict[str, dict[str, float]] = {
    "easy":   {"rc": 0.38, "sev": 0.28, "rem": 0.28, "link": 0.00, "fa": 0.00, "eff": 0.03, "ord": 0.03},
    "medium": {"rc": 0.28, "sev": 0.20, "rem": 0.20, "link": 0.20, "fa": 0.07, "eff": 0.02, "ord": 0.03},
    "hard":   {"rc": 0.28, "sev": 0.20, "rem": 0.17, "link": 0.18, "fa": 0.09, "eff": 0.04, "ord": 0.04},
}

_STEALTH_BONUS: dict[str, float] = {
    "easy": 0.00,
    "medium": 0.00,
    "hard": 0.10,
}

# Efficiency floor: completing all alerts has inherent value even if the full
# step budget is consumed.  The floor prevents a binary 0/1 cliff and rewards
# agents that finish the job at all.
_EFFICIENCY_FLOOR: float = 0.20

# ─────────────────────────────────────────────────────────────────────────────
# Root-cause and remediation partial credit
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def grade_episode(
    task_id: str,
    final_state_dict: dict[str, Any],
) -> float:
    """
    Compute a (0.0, 1.0) episode score based on weighted accuracy components.
    """

    if task_id not in _WEIGHTS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Valid values: {sorted(_WEIGHTS.keys())}"
        )

    all_ground_truth: list[dict[str, Any]] = final_state_dict.get("ground_truth", [])
    dynamic_ids: set[str] = final_state_dict.get("dynamic_alert_ids", set())
    incidents: list[dict[str, Any]] = final_state_dict.get("incidents", [])
    agent_links: list[dict[str, Any]] = final_state_dict.get("agent_links", [])
    agent_decisions: list[dict[str, Any]] = final_state_dict.get("agent_decisions", [])
    triage_order: list[str] = final_state_dict.get("triage_order", [])
    steps_used: int = final_state_dict.get("step_number", 0)
    max_steps: int = final_state_dict.get("max_steps", 1)

    # Filter out dynamic alerts so the grader scores only original alerts.
    ground_truth: list[dict[str, Any]] = [
        gt for gt in all_ground_truth
        if gt["alert_id"] not in dynamic_ids
    ]

    # Separate triageable alerts from false alarms.
    # Classification accuracy (rc, sev, rem) is computed ONLY over triageable
    # alerts — false alarms are scored separately via false_alarm_accuracy.
    triageable_gt: list[dict[str, Any]] = [
        gt for gt in ground_truth if gt["true_root_cause"] != "false_alarm"
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

    # Core classification score — rc/sev/rem use triageable_gt only
    base_score = (
        w["rc"]   * _root_cause_accuracy(decisions_by_id, triageable_gt) +
        w["sev"]  * _severity_accuracy(decisions_by_id, triageable_gt) +
        w["rem"]  * _remediation_accuracy(decisions_by_id, triageable_gt) +
        w["link"] * _incident_link_f1(agent_links, ground_truth) +
        w["fa"]   * _false_alarm_accuracy(decisions_by_id, skips_by_id, ground_truth)
    )

    # Operational quality components
    base_score += (
        w["eff"]  * _efficiency_score(steps_used, max_steps) +
        w["ord"]  * _triage_ordering_score(triage_order, ground_truth, dynamic_ids)
    )

    # Coverage penalty — uses ALL ground truth (including FAs)
    handled_original = sum(
        1 for aid in original_ids
        if aid in decisions_by_id or aid in skips_by_id
    )
    coverage = handled_original / len(ground_truth) if ground_truth else 1.0
    coverage_penalty = coverage ** 1.5

    score = base_score * coverage_penalty

    # Stealth bonus (hard only)
    score += _STEALTH_BONUS[task_id] * _stealth_bonus(
        decisions_by_id, ground_truth, incidents
    )

    # Clamp to open interval (0, 1) — strictly between 0 and 1.
    # - Minimum: 0.0001 (cannot be 0 or lower)
    # - Maximum: 0.9999 (cannot be 1 or higher)
    return round(max(0.0001, min(0.9999, score)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Component scorers
# ─────────────────────────────────────────────────────────────────────────────

def _root_cause_accuracy(decisions_by_id, ground_truth) -> float:
    """Root cause accuracy over triageable (non-FA) alerts only."""
    if not ground_truth:
        return 1.0
    total = 0.0
    for gt in ground_truth:
        agent_rc = decisions_by_id.get(gt["alert_id"], {}).get("root_cause")
        true_rc = gt["true_root_cause"]
        if agent_rc == true_rc:
            total += 1.0
        elif agent_rc and frozenset((agent_rc, true_rc)) in _RC_RELATED_PAIRS:
            total += _RC_PARTIAL_CREDIT
    return total / len(ground_truth)


def _severity_accuracy(decisions_by_id, ground_truth) -> float:
    """Severity accuracy over triageable (non-FA) alerts only."""
    if not ground_truth:
        return 1.0
    total = 0.0
    for gt in ground_truth:
        decision = decisions_by_id.get(gt["alert_id"])
        if decision is None:
            continue
        agent_sev = decision.get("severity", "")
        true_sev = gt["true_severity"]
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
    """Remediation accuracy over triageable (non-FA) alerts only."""
    if not ground_truth:
        return 1.0
    total = 0.0
    for gt in ground_truth:
        agent_rem = decisions_by_id.get(gt["alert_id"], {}).get("remediation")
        true_rem = gt["true_remediation"]
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
    precision = tp / len(agent_pairs)
    recall = tp / len(true_pairs)
    denom = precision + recall
    return (2 * precision * recall / denom) if denom > 0 else 0.0


def _false_alarm_accuracy(decisions_by_id, skips_by_id, ground_truth) -> float:
    fa_alerts = [gt for gt in ground_truth if gt["true_root_cause"] == "false_alarm"]
    if not fa_alerts:
        return 1.0
    real_alerts = [gt for gt in ground_truth if gt["true_root_cause"] != "false_alarm"]
    correctly_skipped_fa = sum(
        1 for gt in fa_alerts if gt["alert_id"] in skips_by_id
    )
    correctly_triaged_real = sum(
        1 for gt in real_alerts if gt["alert_id"] in decisions_by_id
    )
    total = len(ground_truth)
    base = (correctly_skipped_fa + correctly_triaged_real) / total if total > 0 else 1.0
    skip_ratio = len(skips_by_id) / total if total > 0 else 0
    penalty = max(0.0, 1 - skip_ratio * 0.5)
    return base * penalty


def _efficiency_score(steps_used: int, max_steps: int) -> float:
    """max(0.20, 1.0 − steps/max_steps) — floor rewards task completion."""
    if max_steps <= 0:
        return 1.0
    raw = 1.0 - (steps_used / max_steps)
    return max(_EFFICIENCY_FLOOR, raw)


def _triage_ordering_score(
    triage_order: list[str],
    ground_truth: list[dict[str, Any]],
    dynamic_ids: set[str],
) -> float:
    """Pairwise concordance — did the agent triage critical alerts first?"""
    sev_rank: dict[str, int] = {}
    for gt in ground_truth:
        if gt["alert_id"] not in dynamic_ids:
            sev_rank[gt["alert_id"]] = SEVERITY_ORDER.get(gt["true_severity"], 3)
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
        if decision and decision.get("root_cause") == gt["true_root_cause"]:
            return 1.0
    return 0.0


def _pairs_from_groups(groups):
    pairs = set()
    for group in groups:
        for a, b in itertools.combinations(group, 2):
            pairs.add(frozenset((a, b)))
    return pairs
