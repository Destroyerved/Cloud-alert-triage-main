"""
server/rewards.py

Per-step reward calculation for the Cloud Alert Triage environment.
Implemented in Phase 5.

Public API
----------
    compute_reward(action_dict, ground_truth_list, env_state_dict) -> float

Private helpers (importable for unit tests)
-------------------------------------------
    _reward_triage(action_dict, ground_truth_list, env_state_dict) -> float
    _reward_link(alert_ids, true_incidents)                         -> float
    _reward_skip(alert_id, ground_truth_list)                       -> float
    _penalty_budget(step, max_steps)                                -> float

Reward table (master plan Section 8)
-------------------------------------
  triage — root_cause exact match        -> +0.30
  triage — severity exact match          -> +0.30
  triage — severity within 1 level       -> +0.10  (not stacked with exact)
  triage — remediation exact match       -> +0.20
  triage — incident link bonus           -> +0.10  (alert in incident AND
                                                    agent previously linked it
                                                    correctly)
  link_alerts — correct pair             -> +0.15 per pair
  link_alerts — incorrect pair           -> -0.10 per pair
  skip — true false_alarm                -> +0.20
  skip — real alert                      -> -0.30
  budget pressure (step >= 80% budget)   -> -0.05

Design constraints
------------------
- Deterministic: same inputs -> same output, always.
- No side effects: does not mutate any argument.
- env_state_dict is the snapshot produced by AlertTriageEnv._make_state_snapshot();
  keys used: step_number, max_steps, incidents, agent_links.
- The environment handles invalid-action (-0.10) and double-triage (-0.15)
  penalties directly before calling compute_reward; those cases never reach
  this module.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from server.config import SEVERITY_ORDER


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_reward(
    action_dict: dict[str, Any],
    ground_truth_list: list[dict[str, Any]],
    env_state_dict: dict[str, Any],
) -> float:
    """
    Compute the scalar reward for a single action.

    Parameters
    ----------
    action_dict        : plain dict with at least ``action_type`` key.
    ground_truth_list  : list of GT dicts (one per alert); each has
                         ``alert_id``, ``true_root_cause``, ``true_severity``,
                         ``true_remediation``, ``incident_id``.
    env_state_dict     : snapshot of env state; relevant keys:
                         ``step_number``, ``max_steps``,
                         ``incidents`` (list of incident dicts),
                         ``agent_links`` (list of prior link_alerts decisions).

    Returns
    -------
    float  -- may be positive, zero, or negative.
    """
    action_type = action_dict.get("action_type", "")

    if action_type == "triage":
        base = _reward_triage(action_dict, ground_truth_list, env_state_dict)
    elif action_type == "link_alerts":
        base = _reward_link(
            action_dict.get("alert_ids") or [],
            env_state_dict.get("incidents") or [],
        )
    elif action_type == "skip":
        base = _reward_skip(action_dict.get("alert_id"), ground_truth_list)
    else:
        # Unrecognised action_type -- defensive fallback (env catches bad types
        # before calling here, but guard anyway).
        base = 0.0

    penalty = _penalty_budget(
        env_state_dict.get("step_number", 0),
        env_state_dict.get("max_steps", 1),
    )

    return round(base + penalty, 6)


# ---------------------------------------------------------------------------
# Per-action-type reward functions
# ---------------------------------------------------------------------------

def _reward_triage(
    action_dict: dict[str, Any],
    ground_truth_list: list[dict[str, Any]],
    env_state_dict: dict[str, Any],
) -> float:
    """
    Compute base reward for a ``triage`` action.

    Components
    ----------
    +0.30   root_cause exact match
    +0.30   severity exact match          (mutually exclusive with +0.10 below)
    +0.10   severity within 1 level of correct
    +0.20   remediation exact match
    +0.10   incident-link bonus (alert in incident AND previously linked correctly)
    ------
    max     +0.90
    """
    alert_id = action_dict.get("alert_id")
    gt = _find_gt(alert_id, ground_truth_list)
    if gt is None:
        # No ground truth entry found -- cannot score.
        return 0.0

    reward = 0.0

    # -- root cause -----------------------------------------------------------
    if action_dict.get("root_cause") == gt["true_root_cause"]:
        reward += 0.30

    # -- severity -------------------------------------------------------------
    agent_sev: str | None = action_dict.get("severity")
    true_sev: str = gt["true_severity"]
    if agent_sev == true_sev:
        reward += 0.30
    else:
        agent_rank = SEVERITY_ORDER.get(agent_sev or "", -99)
        true_rank = SEVERITY_ORDER.get(true_sev, -99)
        if agent_rank != -99 and true_rank != -99:
            if abs(agent_rank - true_rank) == 1:
                reward += 0.10

    # -- remediation ----------------------------------------------------------
    if action_dict.get("remediation") == gt["true_remediation"]:
        reward += 0.20

    # -- incident link bonus --------------------------------------------------
    incident_id: str | None = gt.get("incident_id")
    if incident_id is not None:
        agent_links: list[dict[str, Any]] = env_state_dict.get("agent_links") or []
        if _agent_correctly_linked(alert_id, incident_id, agent_links, ground_truth_list):
            reward += 0.10

    return reward


def _reward_link(
    alert_ids: list[str],
    true_incidents: list[dict[str, Any]],
) -> float:
    """
    Compute reward for a ``link_alerts`` action.

    For every pair (a, b) chosen from alert_ids:
        both in the same true incident  -> +0.15
        otherwise                       -> -0.10

    Parameters
    ----------
    alert_ids      : IDs the agent wants to group together.
    true_incidents : incident dicts from the scenario; each has
                     ``incident_id`` and ``alert_ids`` (list of member IDs).
    """
    if len(alert_ids) < 2:
        return 0.0

    # Build alert_id -> incident_id lookup from ground truth.
    alert_to_incident: dict[str, str] = {}
    for inc in true_incidents:
        inc_id: str = inc.get("incident_id", "")
        for aid in inc.get("alert_ids") or []:
            alert_to_incident[aid] = inc_id

    reward = 0.0
    for a, b in combinations(alert_ids, 2):
        inc_a = alert_to_incident.get(a)
        inc_b = alert_to_incident.get(b)
        # Both must belong to a real (non-None) incident, and it must be the same one.
        if inc_a is not None and inc_a == inc_b:
            reward += 0.15
        else:
            reward -= 0.10

    return reward


def _reward_skip(
    alert_id: str | None,
    ground_truth_list: list[dict[str, Any]],
) -> float:
    """
    Compute reward for a ``skip`` action.

        +0.20  alert is a true false_alarm
        -0.30  alert is a real (non-false-alarm) alert
    """
    if alert_id is None:
        return 0.0
    gt = _find_gt(alert_id, ground_truth_list)
    if gt is None:
        return 0.0
    if gt["true_root_cause"] == "false_alarm":
        return 0.20
    return -0.30


def _penalty_budget(step: int, max_steps: int) -> float:
    """
    Return -0.05 once ``step`` >= 80% of ``max_steps``; otherwise 0.0.

    This urgency signal is applied on every action taken after 80% of the
    step budget is consumed.

    Parameters
    ----------
    step      : current step_number from the env state snapshot (the count of
                steps already taken at the time this action is evaluated).
    max_steps : total step budget for the episode.
    """
    if max_steps > 0 and step >= 0.8 * max_steps:
        return -0.05
    return 0.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_gt(
    alert_id: str | None,
    ground_truth_list: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the ground-truth entry for *alert_id*, or None if not found."""
    if alert_id is None:
        return None
    for gt in ground_truth_list:
        if gt.get("alert_id") == alert_id:
            return gt
    return None


def _agent_correctly_linked(
    alert_id: str,
    incident_id: str,
    agent_links: list[dict[str, Any]],
    ground_truth_list: list[dict[str, Any]],
) -> bool:
    """
    Return True if the agent has previously issued a ``link_alerts`` action
    that grouped *alert_id* together with at least one other alert that truly
    belongs to the same incident (*incident_id*).

    This is the condition for the +0.10 incident-link bonus on a triage action.
    """
    # Build the set of all alert IDs that truly belong to this incident.
    true_incident_members: set[str] = {
        gt["alert_id"]
        for gt in ground_truth_list
        if gt.get("incident_id") == incident_id
    }

    for link in agent_links:
        linked_ids: set[str] = set(link.get("alert_ids") or [])
        if alert_id not in linked_ids:
            continue
        # There must be at least one other member of the same true incident
        # also present in this link group.
        other_true_members = (linked_ids - {alert_id}) & true_incident_members
        if other_true_members:
            return True
    return False
