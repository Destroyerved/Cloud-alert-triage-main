"""
tests/test_graders.py
Tests for server/grading.py — end-of-episode scoring.
Run with: pytest tests/test_graders.py -v
"""

import pytest

from server.grading import grade_episode


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_state(task_id, decisions, ground_truth, incidents=None, step_number=None, max_steps=10):
    """Build a minimal EnvironmentState dict for grader input."""
    agent_links = [d for d in decisions if d.get("action_type") == "link_alerts"]
    return {
        "task_id": task_id,
        "seed": 42,
        "step_number": step_number or max_steps,
        "max_steps": max_steps,
        "done": True,
        "alerts": [],
        "ground_truth": ground_truth,
        "agent_decisions": decisions,
        "agent_links": agent_links,
        "incidents": incidents or [],
        "cumulative_reward": 0.0,
        "grader_score": None,
    }


GROUND_TRUTH_5 = [
    {"alert_id": f"alert-{i:03d}", "true_root_cause": "resource_exhaustion",
     "true_severity": "high", "true_remediation": "scale_up", "incident_id": None}
    for i in range(1, 6)
]

PERFECT_DECISIONS_5 = [
    {"alert_id": f"alert-{i:03d}", "action_type": "triage",
     "root_cause": "resource_exhaustion", "severity": "high", "remediation": "scale_up"}
    for i in range(1, 6)
]

ALL_WRONG_DECISIONS_5 = [
    {"alert_id": f"alert-{i:03d}", "action_type": "triage",
     "root_cause": "network_failure", "severity": "low", "remediation": "restart_service"}
    for i in range(1, 6)
]


# ─────────────────────────────────────────────────────────────────────────────
# Easy grader
# ─────────────────────────────────────────────────────────────────────────────

class TestEasyGrader:

    def test_perfect_run_scores_1_0(self):
        """Perfect decisions on easy task → 1.0."""
        state = _make_state("easy", PERFECT_DECISIONS_5, GROUND_TRUTH_5)
        score = grade_episode("easy", state)
        assert score == pytest.approx(1.0)

    def test_all_wrong_scores_near_zero(self):
        """All wrong decisions → score close to 0.0."""
        state = _make_state("easy", ALL_WRONG_DECISIONS_5, GROUND_TRUTH_5)
        score = grade_episode("easy", state)
        # Severity "low" vs "high" is off by 3 levels → 0.0. No partial credit here.
        assert score < 0.15

    def test_empty_decisions_scores_zero(self):
        """No decisions at all (agent made no moves) → 0.0."""
        state = _make_state("easy", [], GROUND_TRUTH_5)
        score = grade_episode("easy", state)
        assert score == pytest.approx(0.0)

    def test_partial_run_in_range(self):
        """Triaging 3/5 correctly, 2 untriaged → score between 0 and 1."""
        decisions = PERFECT_DECISIONS_5[:3]
        state = _make_state("easy", decisions, GROUND_TRUTH_5)
        score = grade_episode("easy", state)
        # Coverage penalty (0.6^1.5 ≈ 0.465) is applied on top of the
        # weighted accuracy sum, so final score is well below the raw 0.60.
        assert 0.0 < score < 1.0

    def test_score_always_in_range(self):
        """Grader output is always clamped to [0.0, 1.0]."""
        for decisions in [PERFECT_DECISIONS_5, ALL_WRONG_DECISIONS_5, []]:
            state = _make_state("easy", decisions, GROUND_TRUTH_5)
            score = grade_episode("easy", state)
            assert 0.0 <= score <= 1.0

    def test_severity_partial_credit(self):
        """Severity off by exactly 1 level → 0.3 partial credit per alert."""
        # "high" → "medium" is off by 1 (SEVERITY_ORDER: critical=0, high=1, medium=2, low=3)
        decisions = [
            {"alert_id": f"alert-{i:03d}", "action_type": "triage",
             "root_cause": "resource_exhaustion", "severity": "medium", "remediation": "scale_up"}
            for i in range(1, 6)
        ]
        state = _make_state("easy", decisions, GROUND_TRUTH_5)
        score = grade_episode("easy", state)
        # rc=1.0, sev=0.3, rem=1.0 (coverage=1.0, no penalty)
        # 0.40*1.0 + 0.30*0.3 + 0.30*1.0 = 0.40 + 0.09 + 0.30 = 0.79
        assert score == pytest.approx(0.79)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestGraderDeterminism:

    def test_same_input_same_output(self):
        """Identical state → identical score (determinism)."""
        state = _make_state("easy", PERFECT_DECISIONS_5, GROUND_TRUTH_5)
        score_a = grade_episode("easy", state)
        score_b = grade_episode("easy", state)
        assert score_a == score_b

    def test_different_decisions_different_scores(self):
        """Different decision quality → different scores."""
        perfect_state = _make_state("easy", PERFECT_DECISIONS_5, GROUND_TRUTH_5)
        wrong_state = _make_state("easy", ALL_WRONG_DECISIONS_5, GROUND_TRUTH_5)
        assert grade_episode("easy", perfect_state) > grade_episode("easy", wrong_state)


# ─────────────────────────────────────────────────────────────────────────────
# Medium grader
# ─────────────────────────────────────────────────────────────────────────────

# Ground truth: 6 alerts, 2 incidents of 3 alerts each
_GT_MEDIUM = [
    {"alert_id": "a1", "true_root_cause": "deployment_bug", "true_severity": "high",
     "true_remediation": "rollback_deploy", "incident_id": "inc-1"},
    {"alert_id": "a2", "true_root_cause": "deployment_bug", "true_severity": "high",
     "true_remediation": "rollback_deploy", "incident_id": "inc-1"},
    {"alert_id": "a3", "true_root_cause": "deployment_bug", "true_severity": "medium",
     "true_remediation": "rollback_deploy", "incident_id": "inc-1"},
    {"alert_id": "a4", "true_root_cause": "network_failure", "true_severity": "critical",
     "true_remediation": "escalate_to_team", "incident_id": "inc-2"},
    {"alert_id": "a5", "true_root_cause": "network_failure", "true_severity": "critical",
     "true_remediation": "escalate_to_team", "incident_id": "inc-2"},
    {"alert_id": "a6", "true_root_cause": "false_alarm", "true_severity": "low",
     "true_remediation": "dismiss", "incident_id": None},
]

_PERFECT_TRIAGE_MEDIUM = [
    {"alert_id": "a1", "action_type": "triage", "root_cause": "deployment_bug",
     "severity": "high", "remediation": "rollback_deploy"},
    {"alert_id": "a2", "action_type": "triage", "root_cause": "deployment_bug",
     "severity": "high", "remediation": "rollback_deploy"},
    {"alert_id": "a3", "action_type": "triage", "root_cause": "deployment_bug",
     "severity": "medium", "remediation": "rollback_deploy"},
    {"alert_id": "a4", "action_type": "triage", "root_cause": "network_failure",
     "severity": "critical", "remediation": "escalate_to_team"},
    {"alert_id": "a5", "action_type": "triage", "root_cause": "network_failure",
     "severity": "critical", "remediation": "escalate_to_team"},
]

_CORRECT_LINKS_MEDIUM = [
    {"action_type": "link_alerts", "alert_ids": ["a1", "a2", "a3"], "incident_label": "inc-1"},
    {"action_type": "link_alerts", "alert_ids": ["a4", "a5"], "incident_label": "inc-2"},
]

_CORRECT_SKIP_FA = [{"alert_id": "a6", "action_type": "skip"}]


class TestMediumGrader:

    def test_incident_linking_weighted(self):
        """Medium task includes incident link F1 in score (weight 0.20)."""
        # Perfect triage but no linking → score < 1.0 (missing link contribution)
        decisions = _PERFECT_TRIAGE_MEDIUM[:]
        state = _make_state("medium", decisions, _GT_MEDIUM)
        score_no_links = grade_episode("medium", state)

        # Perfect triage + correct linking → higher score
        decisions_with_links = _PERFECT_TRIAGE_MEDIUM + _CORRECT_LINKS_MEDIUM
        state_with_links = _make_state("medium", decisions_with_links, _GT_MEDIUM)
        score_with_links = grade_episode("medium", state_with_links)

        assert score_with_links > score_no_links

    def test_perfect_triage_no_links_below_1(self):
        """Perfect triage but no incident links → score < 1.0 for medium."""
        decisions = _PERFECT_TRIAGE_MEDIUM[:]
        state = _make_state("medium", decisions, _GT_MEDIUM)
        score = grade_episode("medium", state)
        assert score < 1.0

    def test_false_alarm_identification(self):
        """Correctly skipping a false alarm contributes to false_alarm_accuracy."""
        # Without skipping the FA, fa_accuracy is lower
        decisions_no_skip = _PERFECT_TRIAGE_MEDIUM + _CORRECT_LINKS_MEDIUM
        state_no_skip = _make_state("medium", decisions_no_skip, _GT_MEDIUM)

        # With correct FA skip
        decisions_with_skip = _PERFECT_TRIAGE_MEDIUM + _CORRECT_LINKS_MEDIUM + _CORRECT_SKIP_FA
        state_with_skip = _make_state("medium", decisions_with_skip, _GT_MEDIUM)

        assert grade_episode("medium", state_with_skip) > grade_episode("medium", state_no_skip)

    def test_score_in_range(self):
        """Medium grader always returns [0.0, 1.0]."""
        for decisions in [[], _PERFECT_TRIAGE_MEDIUM, _PERFECT_TRIAGE_MEDIUM + _CORRECT_LINKS_MEDIUM]:
            score = grade_episode("medium", _make_state("medium", decisions, _GT_MEDIUM))
            assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Hard grader
# ─────────────────────────────────────────────────────────────────────────────

_GT_HARD = [
    {"alert_id": "h1", "true_root_cause": "config_error", "true_severity": "critical",
     "true_remediation": "fix_config", "incident_id": "stealth-inc"},
    {"alert_id": "h2", "true_root_cause": "config_error", "true_severity": "high",
     "true_remediation": "fix_config", "incident_id": "stealth-inc"},
]

_STEALTH_INCIDENT = [{"incident_id": "stealth-inc", "stealth": True}]

_STEALTH_CORRECT_TRIAGE = [
    {"alert_id": "h1", "action_type": "triage", "root_cause": "config_error",
     "severity": "critical", "remediation": "fix_config"},
    {"alert_id": "h2", "action_type": "triage", "root_cause": "config_error",
     "severity": "high", "remediation": "fix_config"},
]

_STEALTH_WRONG_TRIAGE = [
    {"alert_id": "h1", "action_type": "triage", "root_cause": "network_failure",
     "severity": "critical", "remediation": "fix_config"},
    {"alert_id": "h2", "action_type": "triage", "root_cause": "network_failure",
     "severity": "high", "remediation": "fix_config"},
]


class TestHardGrader:

    def test_stealth_bonus_applies(self):
        """Hard grader awards exactly +0.05 bonus for identifying the stealth incident root cause."""
        # Same correct triage; only difference is whether the stealth marker exists.
        state_with_stealth = _make_state(
            "hard", _STEALTH_CORRECT_TRIAGE, _GT_HARD, incidents=_STEALTH_INCIDENT
        )
        state_no_stealth = _make_state(
            "hard", _STEALTH_CORRECT_TRIAGE, _GT_HARD, incidents=[]
        )
        score_with = grade_episode("hard", state_with_stealth)
        score_without = grade_episode("hard", state_no_stealth)
        assert score_with > score_without
        assert score_with - score_without == pytest.approx(0.05)

    def test_no_stealth_incident_no_bonus(self):
        """Without a stealth incident in the incidents list, bonus is 0."""
        state = _make_state("hard", _STEALTH_CORRECT_TRIAGE, _GT_HARD, incidents=[])
        score_no_stealth = grade_episode("hard", state)

        state_with_stealth = _make_state(
            "hard", _STEALTH_CORRECT_TRIAGE, _GT_HARD, incidents=_STEALTH_INCIDENT
        )
        score_with_stealth = grade_episode("hard", state_with_stealth)
        assert score_with_stealth > score_no_stealth

    def test_incident_link_weighted_higher_than_medium(self):
        """Hard task weights incident linking at 0.25 vs medium's 0.20."""
        # Build identical scenarios for medium and hard, compare impact of adding links
        gt = [
            {"alert_id": "x1", "true_root_cause": "deployment_bug", "true_severity": "high",
             "true_remediation": "rollback_deploy", "incident_id": "i1"},
            {"alert_id": "x2", "true_root_cause": "deployment_bug", "true_severity": "high",
             "true_remediation": "rollback_deploy", "incident_id": "i1"},
        ]
        triage = [
            {"alert_id": "x1", "action_type": "triage", "root_cause": "deployment_bug",
             "severity": "high", "remediation": "rollback_deploy"},
            {"alert_id": "x2", "action_type": "triage", "root_cause": "deployment_bug",
             "severity": "high", "remediation": "rollback_deploy"},
        ]
        link = [{"action_type": "link_alerts", "alert_ids": ["x1", "x2"], "incident_label": "i1"}]

        med_no_link = grade_episode("medium", _make_state("medium", triage, gt))
        med_with_link = grade_episode("medium", _make_state("medium", triage + link, gt))
        hard_no_link = grade_episode("hard", _make_state("hard", triage, gt))
        hard_with_link = grade_episode("hard", _make_state("hard", triage + link, gt))

        # Linking should improve hard score more than medium score
        assert (hard_with_link - hard_no_link) > (med_with_link - med_no_link)

    def test_score_in_range(self):
        """Hard grader always returns [0.0, 1.0]."""
        for decisions in [[], _STEALTH_CORRECT_TRIAGE, _STEALTH_WRONG_TRIAGE]:
            score = grade_episode("hard", _make_state("hard", decisions, _GT_HARD))
            assert 0.0 <= score <= 1.0
