"""
tests/test_rewards.py

Unit tests for server/rewards.py — per-step reward calculations.
Run with: pytest tests/test_rewards.py -v

All expected values are derived directly from the reward table in
master plan Section 8.  Each test carries an explicit comment showing
the arithmetic so failures are easy to diagnose.

Severity rank reference (SEVERITY_ORDER from config.py):
    critical -> 0
    high     -> 1
    medium   -> 2
    low      -> 3
"""

from __future__ import annotations

import pytest

from server.rewards import (
    _penalty_budget,
    _reward_link,
    _reward_skip,
    _reward_triage,
    compute_reward,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# Minimal ground-truth list covering three alerts:
#   alert-001: resource_exhaustion / high       / scale_up          (no incident)
#   alert-002: deployment_bug      / critical   / rollback_deploy    (no incident)
#   alert-003: false_alarm         / low        / dismiss            (no incident)
GT_BASIC: list[dict] = [
    {
        "alert_id":        "alert-001",
        "true_root_cause": "resource_exhaustion",
        "true_severity":   "high",
        "true_remediation":"scale_up",
        "incident_id":     None,
    },
    {
        "alert_id":        "alert-002",
        "true_root_cause": "deployment_bug",
        "true_severity":   "critical",
        "true_remediation":"rollback_deploy",
        "incident_id":     None,
    },
    {
        "alert_id":        "alert-003",
        "true_root_cause": "false_alarm",
        "true_severity":   "low",
        "true_remediation":"dismiss",
        "incident_id":     None,
    },
]

# Ground-truth list where alert-001 and alert-002 share an incident.
GT_WITH_INCIDENT: list[dict] = [
    {
        "alert_id":        "alert-001",
        "true_root_cause": "resource_exhaustion",
        "true_severity":   "high",
        "true_remediation":"scale_up",
        "incident_id":     "INC-001",
    },
    {
        "alert_id":        "alert-002",
        "true_root_cause": "dependency_outage",
        "true_severity":   "critical",
        "true_remediation":"escalate_to_team",
        "incident_id":     "INC-001",
    },
    {
        "alert_id":        "alert-003",
        "true_root_cause": "false_alarm",
        "true_severity":   "low",
        "true_remediation":"dismiss",
        "incident_id":     None,
    },
]

# Incident list matching GT_WITH_INCIDENT
INCIDENTS_INC001: list[dict] = [
    {
        "incident_id":  "INC-001",
        "root_service": "some-service",
        "root_cause":   "resource_exhaustion",
        "alert_ids":    ["alert-001", "alert-002"],
    }
]

# Env state snapshot at an early step (step 1 of 10 -- no budget pressure).
ENV_EARLY = {
    "step_number": 1,
    "max_steps":   10,
    "incidents":   [],
    "agent_links": [],
}

# Env state snapshot at a late step (step 9 of 10 -- budget pressure active).
ENV_LATE = {
    "step_number": 9,
    "max_steps":   10,
    "incidents":   [],
    "agent_links": [],
}

# Env state with incident list (for link_alerts and link-bonus tests).
ENV_WITH_INCIDENT = {
    "step_number": 1,
    "max_steps":   10,
    "incidents":   INCIDENTS_INC001,
    "agent_links": [],
}


# ---------------------------------------------------------------------------
# _reward_triage
# ---------------------------------------------------------------------------

class TestRewardTriage:

    def test_perfect_triage_all_correct(self):
        """
        root_cause correct (+0.30) + severity exact (+0.30) + remediation (+0.20)
        = 0.80
        """
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",   # correct
            "severity":    "high",                   # correct
            "remediation": "scale_up",               # correct
        }
        result = _reward_triage(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.80)

    def test_all_wrong_returns_zero(self):
        """All three fields wrong -> 0.00."""
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "network_failure",    # wrong
            "severity":    "low",                # wrong (high->low = 2 levels off)
            "remediation": "restart_service",    # wrong
        }
        result = _reward_triage(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.0)

    def test_severity_within_one_level_partial_credit(self):
        """
        true=high(1), agent=medium(2), diff=1 -> +0.10 (not +0.30).
        root_cause correct (+0.30) + severity partial (+0.10) + remediation (+0.20)
        = 0.60
        """
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",   # correct
            "severity":    "medium",                 # off by 1
            "remediation": "scale_up",               # correct
        }
        result = _reward_triage(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.60)

    def test_severity_off_by_two_no_partial_credit(self):
        """
        true=critical(0), agent=low(3), diff=3 -> +0.00 for severity.
        root_cause correct (+0.30) + severity wrong (+0.00) + remediation (+0.20)
        = 0.50
        """
        action = {
            "action_type": "triage",
            "alert_id":    "alert-002",
            "root_cause":  "deployment_bug",    # correct
            "severity":    "low",               # off by 3
            "remediation": "rollback_deploy",   # correct
        }
        result = _reward_triage(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.50)

    def test_only_root_cause_correct(self):
        """
        root_cause correct (+0.30), severity off by 2, remediation wrong
        = 0.30
        """
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",   # correct
            "severity":    "low",                    # off by 2
            "remediation": "restart_service",        # wrong
        }
        result = _reward_triage(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.30)

    def test_only_severity_correct(self):
        """
        root_cause wrong (+0.00) + severity exact (+0.30) + remediation wrong (+0.00)
        = 0.30
        """
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "network_failure",   # wrong
            "severity":    "high",               # correct
            "remediation": "restart_service",    # wrong
        }
        result = _reward_triage(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.30)

    def test_only_remediation_correct(self):
        """
        root_cause wrong (+0.00) + severity off by 2 (+0.00) + remediation (+0.20)
        = 0.20
        """
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "network_failure",   # wrong
            "severity":    "low",               # off by 2
            "remediation": "scale_up",          # correct
        }
        result = _reward_triage(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.20)

    def test_severity_within_one_level_critical_to_high(self):
        """
        true=critical(0), agent=high(1), diff=1 -> +0.10 partial.
        true remediation correct -> +0.20. root_cause correct -> +0.30.
        = 0.60
        """
        action = {
            "action_type": "triage",
            "alert_id":    "alert-002",
            "root_cause":  "deployment_bug",    # correct
            "severity":    "high",              # off by 1
            "remediation": "rollback_deploy",   # correct
        }
        result = _reward_triage(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.60)

    def test_unknown_alert_id_returns_zero(self):
        """No matching GT entry -> 0.0 (can't score)."""
        action = {
            "action_type": "triage",
            "alert_id":    "alert-999",
            "root_cause":  "resource_exhaustion",
            "severity":    "high",
            "remediation": "scale_up",
        }
        result = _reward_triage(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.0)

    def test_incident_link_bonus_applied_when_previously_linked(self):
        """
        alert-001 is in INC-001, agent previously linked [alert-001, alert-002].
        Perfect triage: +0.30 + 0.30 + 0.20 + 0.10 (bonus) = 0.90
        """
        env_with_link = {
            "step_number": 2,
            "max_steps":   10,
            "incidents":   INCIDENTS_INC001,
            "agent_links": [
                {
                    "action_type":    "link_alerts",
                    "alert_ids":      ["alert-001", "alert-002"],
                    "incident_label": "my-incident",
                }
            ],
        }
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",   # correct
            "severity":    "high",                   # correct
            "remediation": "scale_up",               # correct
        }
        result = _reward_triage(action, GT_WITH_INCIDENT, env_with_link)
        assert result == pytest.approx(0.90)

    def test_incident_link_bonus_not_applied_when_not_linked(self):
        """
        alert-001 is in INC-001 but no prior link_alerts.
        Perfect triage: +0.30 + 0.30 + 0.20 = 0.80 (no bonus).
        """
        env_no_link = {
            "step_number": 2,
            "max_steps":   10,
            "incidents":   INCIDENTS_INC001,
            "agent_links": [],
        }
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",
            "severity":    "high",
            "remediation": "scale_up",
        }
        result = _reward_triage(action, GT_WITH_INCIDENT, env_no_link)
        assert result == pytest.approx(0.80)

    def test_incident_link_bonus_not_applied_for_wrong_link(self):
        """
        Agent linked alert-001 with alert-003, but alert-003 is NOT in INC-001.
        No bonus should be awarded.
        Perfect triage: +0.30 + 0.30 + 0.20 = 0.80 (no bonus).
        """
        env_wrong_link = {
            "step_number": 2,
            "max_steps":   10,
            "incidents":   INCIDENTS_INC001,
            "agent_links": [
                {
                    "action_type":    "link_alerts",
                    "alert_ids":      ["alert-001", "alert-003"],  # alert-003 not in incident
                    "incident_label": "wrong-group",
                }
            ],
        }
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",
            "severity":    "high",
            "remediation": "scale_up",
        }
        result = _reward_triage(action, GT_WITH_INCIDENT, env_wrong_link)
        assert result == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# _reward_link
# ---------------------------------------------------------------------------

class TestRewardLink:

    def test_single_correct_pair(self):
        """
        [alert-001, alert-002] are both in INC-001 -> 1 correct pair -> +0.15
        """
        result = _reward_link(
            ["alert-001", "alert-002"],
            INCIDENTS_INC001,
        )
        assert result == pytest.approx(0.15)

    def test_single_incorrect_pair(self):
        """
        [alert-001, alert-002] with no shared incident -> 1 wrong pair -> -0.10
        """
        result = _reward_link(
            ["alert-001", "alert-002"],
            [],  # no incidents defined
        )
        assert result == pytest.approx(-0.10)

    def test_three_alerts_all_in_same_incident(self):
        """
        [A, B, C] all in INC-001.
        Pairs: (A,B), (A,C), (B,C) -> 3 correct pairs -> 3 * 0.15 = 0.45
        """
        incidents = [
            {
                "incident_id": "INC-001",
                "root_service": "svc",
                "root_cause":   "resource_exhaustion",
                "alert_ids":    ["alert-001", "alert-002", "alert-004"],
            }
        ]
        result = _reward_link(
            ["alert-001", "alert-002", "alert-004"],
            incidents,
        )
        assert result == pytest.approx(0.45)

    def test_mixed_correct_and_incorrect_pairs(self):
        """
        [alert-001, alert-002, alert-003].
        alert-001 and alert-002 are in INC-001; alert-003 is not.
        Pairs:
          (alert-001, alert-002) -> correct -> +0.15
          (alert-001, alert-003) -> wrong   -> -0.10
          (alert-002, alert-003) -> wrong   -> -0.10
        Total = 0.15 - 0.10 - 0.10 = -0.05
        """
        result = _reward_link(
            ["alert-001", "alert-002", "alert-003"],
            INCIDENTS_INC001,
        )
        assert result == pytest.approx(-0.05)

    def test_two_alerts_different_incidents(self):
        """
        alert-001 in INC-001, alert-004 in INC-002 -> wrong pair -> -0.10
        """
        incidents = [
            {"incident_id": "INC-001", "root_service": "a", "root_cause": "resource_exhaustion", "alert_ids": ["alert-001"]},
            {"incident_id": "INC-002", "root_service": "b", "root_cause": "network_failure",     "alert_ids": ["alert-004"]},
        ]
        result = _reward_link(
            ["alert-001", "alert-004"],
            incidents,
        )
        assert result == pytest.approx(-0.10)

    def test_fewer_than_two_alert_ids_returns_zero(self):
        """Degenerate case: only one alert_id -> 0.0 (no pairs)."""
        result = _reward_link(["alert-001"], INCIDENTS_INC001)
        assert result == pytest.approx(0.0)

    def test_empty_alert_ids_returns_zero(self):
        """Empty list -> 0.0."""
        result = _reward_link([], INCIDENTS_INC001)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _reward_skip
# ---------------------------------------------------------------------------

class TestRewardSkip:

    def test_skip_true_false_alarm(self):
        """
        alert-003 has true_root_cause='false_alarm' -> +0.20
        """
        result = _reward_skip("alert-003", GT_BASIC)
        assert result == pytest.approx(0.20)

    def test_skip_real_alert(self):
        """
        alert-001 is a real alert (resource_exhaustion) -> -0.30
        """
        result = _reward_skip("alert-001", GT_BASIC)
        assert result == pytest.approx(-0.30)

    def test_skip_critical_alert_negative_penalty(self):
        """
        alert-002 is critical / deployment_bug -> -0.30
        """
        result = _reward_skip("alert-002", GT_BASIC)
        assert result == pytest.approx(-0.30)

    def test_skip_unknown_alert_id_returns_zero(self):
        """No GT entry -> 0.0 (defensive; env would have caught this)."""
        result = _reward_skip("alert-999", GT_BASIC)
        assert result == pytest.approx(0.0)

    def test_skip_none_alert_id_returns_zero(self):
        """None alert_id -> 0.0."""
        result = _reward_skip(None, GT_BASIC)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _penalty_budget
# ---------------------------------------------------------------------------

class TestPenaltyBudget:

    def test_no_penalty_before_80_percent(self):
        """step=3, max=10 -> 30% used -> no penalty."""
        assert _penalty_budget(3, 10) == pytest.approx(0.0)

    def test_no_penalty_at_79_percent(self):
        """step=7, max=10 -> 70% -> no penalty (threshold is >= 80%)."""
        assert _penalty_budget(7, 10) == pytest.approx(0.0)

    def test_penalty_at_exactly_80_percent(self):
        """step=8, max=10 -> exactly 80% -> -0.05."""
        assert _penalty_budget(8, 10) == pytest.approx(-0.05)

    def test_penalty_beyond_80_percent(self):
        """step=9, max=10 -> 90% -> -0.05."""
        assert _penalty_budget(9, 10) == pytest.approx(-0.05)

    def test_penalty_at_100_percent(self):
        """step=10, max=10 -> 100% -> -0.05."""
        assert _penalty_budget(10, 10) == pytest.approx(-0.05)

    def test_no_penalty_with_zero_max_steps(self):
        """Guard against division by zero: max_steps=0 -> 0.0."""
        assert _penalty_budget(0, 0) == pytest.approx(0.0)

    def test_penalty_for_medium_task(self):
        """step=21, max=25 -> 84% -> -0.05."""
        assert _penalty_budget(21, 25) == pytest.approx(-0.05)

    def test_no_penalty_for_medium_task_early(self):
        """step=10, max=25 -> 40% -> 0.0."""
        assert _penalty_budget(10, 25) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_reward (integration — tests the combined output)
# ---------------------------------------------------------------------------

class TestComputeReward:

    def test_perfect_triage_no_budget_pressure(self):
        """
        Perfect triage + step=1 (no budget pressure).
        0.30 + 0.30 + 0.20 + 0.0 (budget) = 0.80
        """
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",
            "severity":    "high",
            "remediation": "scale_up",
        }
        result = compute_reward(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.80)

    def test_perfect_triage_with_budget_pressure(self):
        """
        Perfect triage + step=9/10 (budget pressure active).
        0.80 - 0.05 = 0.75
        """
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",
            "severity":    "high",
            "remediation": "scale_up",
        }
        result = compute_reward(action, GT_BASIC, ENV_LATE)
        assert result == pytest.approx(0.75)

    def test_all_wrong_triage_no_budget_pressure(self):
        """All wrong + no budget pressure -> 0.00."""
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "network_failure",
            "severity":    "low",
            "remediation": "restart_service",
        }
        result = compute_reward(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.0)

    def test_all_wrong_triage_with_budget_pressure(self):
        """All wrong + budget pressure -> -0.05."""
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "network_failure",
            "severity":    "low",
            "remediation": "restart_service",
        }
        result = compute_reward(action, GT_BASIC, ENV_LATE)
        assert result == pytest.approx(-0.05)

    def test_partial_triage_severity_off_by_one(self):
        """
        root_cause correct (+0.30) + severity partial (+0.10) + remediation (+0.20)
        = 0.60, step=1 -> no budget pressure.
        """
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",
            "severity":    "medium",               # off by 1 from "high"
            "remediation": "scale_up",
        }
        result = compute_reward(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.60)

    def test_skip_false_alarm_no_budget_pressure(self):
        """Skip true false_alarm + step=1 -> +0.20."""
        action = {"action_type": "skip", "alert_id": "alert-003"}
        result = compute_reward(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.20)

    def test_skip_real_alert_no_budget_pressure(self):
        """Skip real alert + step=1 -> -0.30."""
        action = {"action_type": "skip", "alert_id": "alert-001"}
        result = compute_reward(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(-0.30)

    def test_skip_false_alarm_with_budget_pressure(self):
        """Skip true false_alarm + step=9/10 -> 0.20 - 0.05 = 0.15."""
        action = {"action_type": "skip", "alert_id": "alert-003"}
        result = compute_reward(action, GT_BASIC, ENV_LATE)
        assert result == pytest.approx(0.15)

    def test_link_alerts_correct_pair(self):
        """
        [alert-001, alert-002] both in INC-001 -> 1 correct pair -> +0.15.
        step=1 -> no budget pressure.
        """
        env = {**ENV_EARLY, "incidents": INCIDENTS_INC001}
        action = {
            "action_type":    "link_alerts",
            "alert_ids":      ["alert-001", "alert-002"],
            "incident_label": "my-incident",
        }
        result = compute_reward(action, GT_WITH_INCIDENT, env)
        assert result == pytest.approx(0.15)

    def test_link_alerts_incorrect_pair(self):
        """
        [alert-001, alert-002] with no shared incident -> -0.10.
        step=1 -> no budget pressure.
        """
        action = {
            "action_type":    "link_alerts",
            "alert_ids":      ["alert-001", "alert-002"],
            "incident_label": "wrong-group",
        }
        result = compute_reward(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(-0.10)

    def test_link_alerts_with_budget_pressure(self):
        """
        Correct pair (+0.15) + budget pressure (-0.05) = 0.10.
        """
        env_late_incident = {**ENV_LATE, "incidents": INCIDENTS_INC001}
        action = {
            "action_type":    "link_alerts",
            "alert_ids":      ["alert-001", "alert-002"],
            "incident_label": "my-incident",
        }
        result = compute_reward(action, GT_WITH_INCIDENT, env_late_incident)
        assert result == pytest.approx(0.10)

    def test_unknown_action_type_returns_budget_penalty_only(self):
        """
        Unknown action_type -> base=0.0, only budget penalty may apply.
        step=1 -> 0.0 total.
        """
        action = {"action_type": "invalid_type", "alert_id": "alert-001"}
        result = compute_reward(action, GT_BASIC, ENV_EARLY)
        assert result == pytest.approx(0.0)

    def test_result_is_deterministic(self):
        """Same inputs -> same output on repeated calls."""
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",
            "severity":    "high",
            "remediation": "scale_up",
        }
        r1 = compute_reward(action, GT_BASIC, ENV_EARLY)
        r2 = compute_reward(action, GT_BASIC, ENV_EARLY)
        assert r1 == r2

    def test_no_mutation_of_inputs(self):
        """compute_reward must not mutate action_dict, ground_truth, or env_state."""
        import copy
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",
            "severity":    "high",
            "remediation": "scale_up",
        }
        action_copy = copy.deepcopy(action)
        gt_copy     = copy.deepcopy(GT_BASIC)
        env_copy    = copy.deepcopy(ENV_EARLY)

        compute_reward(action, GT_BASIC, ENV_EARLY)

        assert action   == action_copy
        assert GT_BASIC == gt_copy
        assert ENV_EARLY == env_copy


# ---------------------------------------------------------------------------
# Example trajectories from master plan Section 8
# ---------------------------------------------------------------------------

class TestMasterPlanExamples:
    """
    Validate the explicit reward examples given in the master plan.
    These use GT_BASIC (no incidents, 3 alerts).
    """

    def test_example1_perfect_triage(self):
        """
        Master plan Example 1: triage alert-001 correctly -> +0.80.
        (step_number=0 so no budget pressure)
        """
        env_step0 = {**ENV_EARLY, "step_number": 0}
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",
            "severity":    "high",
            "remediation": "scale_up",
        }
        result = compute_reward(action, GT_BASIC, env_step0)
        assert result == pytest.approx(0.80)

    def test_example2_partial_triage_severity_off_by_one(self):
        """
        Master plan Example 2 (Step 1):
        root_cause correct (+0.30) + severity within 1 (+0.10) + remediation (+0.20)
        = 0.60
        """
        env_step0 = {**ENV_EARLY, "step_number": 0}
        action = {
            "action_type": "triage",
            "alert_id":    "alert-001",
            "root_cause":  "resource_exhaustion",  # correct
            "severity":    "medium",                # off by 1 from "high"
            "remediation": "scale_up",              # correct
        }
        result = compute_reward(action, GT_BASIC, env_step0)
        assert result == pytest.approx(0.60)

    def test_example2_link_correct_pair(self):
        """
        Master plan Example 2 (Step 3): correct pair -> +0.15.
        (Using GT_WITH_INCIDENT so the pair is real.)
        """
        env_incidents = {**ENV_EARLY, "step_number": 2, "incidents": INCIDENTS_INC001}
        action = {
            "action_type":    "link_alerts",
            "alert_ids":      ["alert-001", "alert-002"],
            "incident_label": "my-incident",
        }
        result = compute_reward(action, GT_WITH_INCIDENT, env_incidents)
        assert result == pytest.approx(0.15)

    def test_example2_skip_false_alarm(self):
        """
        Master plan Example 2 (Step 4): skip true false_alarm -> +0.20.
        """
        env_step3 = {**ENV_EARLY, "step_number": 3}
        action = {"action_type": "skip", "alert_id": "alert-003"}
        result = compute_reward(action, GT_BASIC, env_step3)
        assert result == pytest.approx(0.20)

    def test_example3_skip_real_critical_alert(self):
        """
        Master plan Example 3 (Step 4): skip a real critical alert -> -0.30.
        """
        env_step3 = {**ENV_EARLY, "step_number": 3}
        action = {"action_type": "skip", "alert_id": "alert-002"}  # critical/deployment_bug
        result = compute_reward(action, GT_BASIC, env_step3)
        assert result == pytest.approx(-0.30)
