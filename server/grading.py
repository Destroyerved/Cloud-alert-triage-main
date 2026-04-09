from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

from server.config import SEVERITY_ORDER

# ─────────────────────────────────────────────────────────────
# Strict open-interval constants — score is ALWAYS in (EPS, 1-EPS)
# ─────────────────────────────────────────────────────────────

EPS = 1e-4          # minimum score
ONE = 1 - 1e-4      # maximum score


def _safe(x: float) -> float:
    """Clamp any float into the strict open interval (EPS, ONE).
    Never returns 0.0 or 1.0 under any circumstances.
    """
    if x is None:
        return EPS
    try:
        x = float(x)
    except (TypeError, ValueError):
        return EPS
    if math.isnan(x) or math.isinf(x):
        return EPS
    if x <= 0:
        return EPS
    if x >= 1:
        return ONE
    return x


def _safe_div(a: float, b: float) -> float:
    """Division that returns EPS instead of 0/0 or n/0."""
    if b == 0:
        return EPS
    result = a / b
    return EPS if result == 0 else result


# ─────────────────────────────────────────────────────────────
# Weights — NO zeros, all strictly inside (0, 1)
# ─────────────────────────────────────────────────────────────

_WEIGHTS: dict[str, dict[str, float]] = {
    "easy":   {"rc": 0.39, "sev": 0.29, "rem": 0.29, "link": 0.01, "fa": 0.02},
    "medium": {"rc": 0.28, "sev": 0.19, "rem": 0.19, "link": 0.20, "fa": 0.14},
    "hard":   {"rc": 0.24, "sev": 0.19, "rem": 0.14, "link": 0.24, "fa": 0.14},
}

_STEALTH_BONUS: dict[str, float] = {
    "easy":   0.01,
    "medium": 0.01,
    "hard":   0.05,
}


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def grade_episode(
    task_id: str,
    final_state_dict: dict[str, Any],
) -> float:
    """Return a score guaranteed strictly inside (EPS, ONE) — never 0.0 or 1.0."""
    try:
        score = _grade_inner(task_id, final_state_dict)
    except Exception:
        score = 0.5   # safe midpoint on unexpected failure

    # Single final clamp — no round() after this
    score = max(EPS, min(ONE, score))
    return score


def _grade_inner(
    task_id: str,
    final_state_dict: dict[str, Any],
) -> float:
    if task_id not in _WEIGHTS:
        raise ValueError(f"Invalid task_id: {task_id}")

    all_ground_truth: list[dict[str, Any]] = final_state_dict.get("ground_truth", [])
    dynamic_ids: set[str] = set(final_state_dict.get("dynamic_alert_ids", set()))
    incidents: list[dict[str, Any]] = final_state_dict.get("incidents", [])
    agent_links: list[dict[str, Any]] = final_state_dict.get("agent_links", [])
    agent_decisions: list[dict[str, Any]] = final_state_dict.get("agent_decisions", [])

    # Exclude dynamic (cascade-spawned) alerts from grading
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

    # Component scores — each passed through _safe before use
    rc   = _safe(_root_cause_accuracy(decisions_by_id, ground_truth))
    sev  = _safe(_severity_accuracy(decisions_by_id, ground_truth))
    rem  = _safe(_remediation_accuracy(decisions_by_id, ground_truth))
    link = _safe(_incident_link_f1(agent_links, ground_truth))
    fa   = _safe(_false_alarm_accuracy(decisions_by_id, skips_by_id, ground_truth))

    base_score = (
        w["rc"]   * rc  +
        w["sev"]  * sev +
        w["rem"]  * rem +
        w["link"] * link +
        w["fa"]   * fa
    )

    # Coverage: fraction of original alerts the agent acted on
    n_gt = max(1, len(ground_truth))
    handled = len(decisions_by_id) + len(skips_by_id)
    coverage = _safe(_safe_div(handled, n_gt))

    score = base_score * coverage

    # Quality penalties (multiplicative, bounded so score can't reach 0)
    if rc < 0.7:
        score *= 0.3
    if sev < 0.7:
        score *= 0.3
    if rem < 0.7:
        score *= 0.3

    # Skip-abuse penalty
    skip_ratio = _safe_div(len(skips_by_id), n_gt)
    score *= max(EPS, 1 - skip_ratio * 2)

    # Stealth bonus — small additive bump
    stealth = _safe(_stealth_bonus(decisions_by_id, ground_truth, incidents))
    score += _STEALTH_BONUS[task_id] * stealth

    return score   # caller applies the final clamp


# ─────────────────────────────────────────────────────────────
# Component scorers
# ─────────────────────────────────────────────────────────────

def _root_cause_accuracy(
    decisions_by_id: dict[str, Any],
    ground_truth: list[dict[str, Any]],
) -> float:
    if not ground_truth:
        return ONE   # vacuously correct

    correct = sum(
        1
        for gt in ground_truth
        if decisions_by_id.get(gt["alert_id"], {}).get("root_cause") == gt["true_root_cause"]
    )
    return _safe_div(correct, len(ground_truth))


def _severity_accuracy(
    decisions_by_id: dict[str, Any],
    ground_truth: list[dict[str, Any]],
) -> float:
    if not ground_truth:
        return ONE

    total = 0.0
    for gt in ground_truth:
        decision = decisions_by_id.get(gt["alert_id"])
        if decision is None:
            continue
        agent = decision.get("severity", "")
        true  = gt.get("true_severity", "")
        if agent == true:
            total += 1.0
        else:
            ar = SEVERITY_ORDER.get(agent, 2)
            tr = SEVERITY_ORDER.get(true, 2)
            if abs(ar - tr) == 1:
                total += 0.15

    return _safe_div(total, len(ground_truth))


def _remediation_accuracy(
    decisions_by_id: dict[str, Any],
    ground_truth: list[dict[str, Any]],
) -> float:
    if not ground_truth:
        return ONE

    correct = sum(
        1
        for gt in ground_truth
        if decisions_by_id.get(gt["alert_id"], {}).get("remediation") == gt["true_remediation"]
    )
    return _safe_div(correct, len(ground_truth))


def _incident_link_f1(
    agent_links: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> float:
    true_groups: dict[str, list[str]] = defaultdict(list)
    for gt in ground_truth:
        inc_id = gt.get("incident_id")
        if inc_id is not None:
            true_groups[inc_id].append(gt["alert_id"])

    true_pairs = _pairs_from_groups(
        [ids for ids in true_groups.values() if len(ids) >= 2]
    )

    if not true_pairs:
        return ONE   # no incidents → vacuously correct; use ONE not 1.0

    agent_pairs = _pairs_from_groups(
        [link["alert_ids"] for link in agent_links if link.get("alert_ids")]
    )

    if not agent_pairs:
        return EPS   # agent made no links → near zero but not 0.0

    tp = len(true_pairs & agent_pairs)

    precision = _safe_div(tp, len(agent_pairs))
    recall    = _safe_div(tp, len(true_pairs))
    denom     = precision + recall

    if denom == 0:
        return EPS

    return 2 * precision * recall / denom


def _false_alarm_accuracy(
    decisions_by_id: dict[str, Any],
    skips_by_id: set[str],
    ground_truth: list[dict[str, Any]],
) -> float:
    if not ground_truth:
        return ONE

    fa_alerts   = [gt for gt in ground_truth if gt.get("true_root_cause") == "false_alarm"]
    real_alerts = [gt for gt in ground_truth if gt.get("true_root_cause") != "false_alarm"]

    correct_fa   = sum(1 for gt in fa_alerts   if gt["alert_id"] in skips_by_id)
    correct_real = sum(1 for gt in real_alerts  if gt["alert_id"] in decisions_by_id)

    base       = _safe_div(correct_fa + correct_real, len(ground_truth))
    skip_ratio = _safe_div(len(skips_by_id), len(ground_truth))
    penalty    = max(EPS, 1 - skip_ratio * 0.5)

    return base * penalty


def _stealth_bonus(
    decisions_by_id: dict[str, Any],
    ground_truth: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
) -> float:
    stealth_inc = next((i for i in incidents if i.get("stealth")), None)
    if not stealth_inc:
        return EPS   # no stealth incident → no bonus; not 0.0

    stealth_id = stealth_inc.get("incident_id") or stealth_inc.get("id")
    if stealth_id is None:
        return EPS

    stealth_alerts = [gt for gt in ground_truth if gt.get("incident_id") == stealth_id]

    for gt in stealth_alerts:
        d = decisions_by_id.get(gt["alert_id"])
        if d and d.get("root_cause") == gt.get("true_root_cause"):
            return ONE   # correctly identified — use ONE not 1.0

    return EPS


def _pairs_from_groups(groups: list[list[str]]) -> set[frozenset[str]]:
    pairs: set[frozenset[str]] = set()
    for group in groups:
        for a, b in itertools.combinations(group, 2):
            pairs.add(frozenset((a, b)))
    return pairs
