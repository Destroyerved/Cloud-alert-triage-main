from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any

from server.config import SEVERITY_ORDER

# ─────────────────────────────────────────────────────────────────────────────
# Per-task weights
# ─────────────────────────────────────────────────────────────────────────────

_WEIGHTS: dict[str, dict[str, float]] = {
    "easy":   {"rc": 0.40, "sev": 0.30, "rem": 0.30, "link": 0.00, "fa": 0.00},
    "medium": {"rc": 0.30, "sev": 0.20, "rem": 0.20, "link": 0.20, "fa": 0.10},
    "hard":   {"rc": 0.25, "sev": 0.20, "rem": 0.15, "link": 0.25, "fa": 0.10},
}

_STEALTH_BONUS: dict[str, float] = {
    "easy": 0.00,
    "medium": 0.00,
    "hard": 0.05,
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def grade_episode(
    task_id: str,
    final_state_dict: dict[str, Any],
) -> float:
    """
    Compute a [0.0, 1.0] episode score based on weighted accuracy components.

    Dynamic alerts (spawned by the cascade mechanic at runtime) are
    **excluded** from the grader's ground-truth set so the final score
    reflects only the original scenario alerts.  This is the safe-for-
    deadline policy documented in task.txt.
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

    # Filter out dynamic alerts so the grader scores only original alerts.
    ground_truth: list[dict[str, Any]] = [
        gt for gt in all_ground_truth
        if gt["alert_id"] not in dynamic_ids
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

    w = _WEIGHTS[task_id]

    # Core score
    base_score = (
        w["rc"]   * _root_cause_accuracy(decisions_by_id, ground_truth) +
        w["sev"]  * _severity_accuracy(decisions_by_id, ground_truth) +
        w["rem"]  * _remediation_accuracy(decisions_by_id, ground_truth) +
        w["link"] * _incident_link_f1(agent_links, ground_truth) +
        w["fa"]   * _false_alarm_accuracy(decisions_by_id, skips_by_id, ground_truth)
    )

    # Coverage penalty (prevents skipping most alerts)
    coverage = len(decisions_by_id) / len(ground_truth) if ground_truth else 1.0
    coverage_penalty = coverage ** 1.5

    score = base_score * coverage_penalty

    # Stealth bonus (hard only)
    score += _STEALTH_BONUS[task_id] * _stealth_bonus(
        decisions_by_id, ground_truth, incidents
    )

    return round(max(0.0, min(1.0, score)), 6)


# ─────────────────────────────────────────────────────────────────────────────
# Component scorers
# ─────────────────────────────────────────────────────────────────────────────

def _root_cause_accuracy(decisions_by_id, ground_truth) -> float:
    if not ground_truth:
        return 1.0
    correct = sum(
        1
        for gt in ground_truth
        if decisions_by_id.get(gt["alert_id"], {}).get("root_cause") == gt["true_root_cause"]
    )
    return correct / len(ground_truth)


def _severity_accuracy(decisions_by_id, ground_truth) -> float:
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
            if abs(agent_rank - true_rank) == 1:
                total += 0.15  # within 1 level = partial credit

    return total / len(ground_truth)


def _remediation_accuracy(decisions_by_id, ground_truth) -> float:
    if not ground_truth:
        return 1.0
    correct = sum(
        1
        for gt in ground_truth
        if decisions_by_id.get(gt["alert_id"], {}).get("remediation") == gt["true_remediation"]
    )
    return correct / len(ground_truth)


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

    # Penalize excessive skipping
    skip_ratio = len(skips_by_id) / total if total > 0 else 0
    penalty = max(0.0, 1 - skip_ratio * 0.5)

    return base * penalty


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