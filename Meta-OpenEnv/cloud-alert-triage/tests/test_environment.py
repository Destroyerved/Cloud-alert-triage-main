"""
tests/test_environment.py
--------------------------
Tests for server/environment.py — AlertTriageEnv core logic.
Implemented in Phase 4.  Run from cloud-alert-triage/ root:

    pytest tests/test_environment.py -v

Design notes
------------
- compute_reward (Phase 5 stub) returns 0.0 for valid triage / link / skip.
- grade_episode  (Phase 6 stub) returns 0.0.
- All tests are written so they pass with these stubs in place.
- Penalty-path tests (-0.10 / -0.15) work because the environment handles
  those cases directly, before calling compute_reward.
- The cumulative_reward tracking test triggers the -0.15 double-triage path
  to guarantee a non-zero value regardless of stub state.
"""

from __future__ import annotations

import pytest

# ── Guard: skip the whole module if the environment is still a stub ──────────
try:
    from server.environment import AlertTriageEnv
    from server.models import Action, EnvironmentState, Observation, StepResult
    AVAILABLE = True
except Exception:
    AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not AVAILABLE,
    reason="AlertTriageEnv not yet implemented — Phase 4",
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def env() -> AlertTriageEnv:
    """Return a fresh AlertTriageEnv instance (not yet reset)."""
    return AlertTriageEnv()


@pytest.fixture
def easy_env() -> AlertTriageEnv:
    """Return an AlertTriageEnv already reset to task='easy', seed=42."""
    e = AlertTriageEnv()
    e.reset("easy", 42)
    return e


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _triage(alert_id: str, **kwargs) -> dict:
    """Build a minimal valid triage action dict."""
    return {
        "action_type": "triage",
        "alert_id":    alert_id,
        "root_cause":  kwargs.get("root_cause",  "resource_exhaustion"),
        "severity":    kwargs.get("severity",    "high"),
        "remediation": kwargs.get("remediation", "scale_up"),
    }


def _skip(alert_id: str) -> dict:
    return {"action_type": "skip", "alert_id": alert_id}


# ─────────────────────────────────────────────────────────────────────────────
# reset()
# ─────────────────────────────────────────────────────────────────────────────

class TestReset:

    def test_easy_returns_observation_type(self, env):
        obs = env.reset("easy", 42)
        assert isinstance(obs, Observation)

    def test_easy_alert_count(self, env):
        obs = env.reset("easy", 42)
        assert len(obs.alerts) == 5

    def test_easy_pending_count(self, env):
        obs = env.reset("easy", 42)
        assert obs.pending_count == 5

    def test_easy_step_number_zero(self, env):
        obs = env.reset("easy", 42)
        assert obs.step_number == 0

    def test_easy_max_steps(self, env):
        obs = env.reset("easy", 42)
        assert obs.max_steps == 10

    def test_easy_empty_feedback(self, env):
        obs = env.reset("easy", 42)
        assert obs.feedback == ""

    def test_medium_alert_count(self, env):
        obs = env.reset("medium", 42)
        assert len(obs.alerts) == 15
        assert obs.pending_count == 15

    def test_hard_alert_count(self, env):
        obs = env.reset("hard", 42)
        assert len(obs.alerts) == 30
        assert obs.pending_count == 30

    def test_service_map_present(self, env):
        obs = env.reset("easy", 42)
        assert isinstance(obs.service_map, dict)
        assert len(obs.service_map) > 0

    def test_all_alerts_not_triaged_on_reset(self, env):
        obs = env.reset("easy", 42)
        assert all(not a.triaged for a in obs.alerts)

    def test_invalid_task_id_raises_value_error(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset("impossible", 42)

    def test_reset_clears_prior_episode(self, env):
        """Calling reset twice starts a clean episode."""
        env.reset("easy", 42)
        obs = env.reset("easy", 99)   # different seed
        assert obs.step_number == 0
        assert obs.pending_count == 5

    def test_same_seed_produces_same_first_alert_id(self, env):
        obs1 = env.reset("easy", 42)
        obs2 = env.reset("easy", 42)
        assert obs1.alerts[0].alert_id == obs2.alerts[0].alert_id

    def test_different_seeds_produce_different_scenarios(self, env):
        obs1 = env.reset("easy", 42)
        obs2 = env.reset("easy", 99)
        msgs1 = [a.message for a in obs1.alerts]
        msgs2 = [a.message for a in obs2.alerts]
        assert msgs1 != msgs2


# ─────────────────────────────────────────────────────────────────────────────
# step() — before reset
# ─────────────────────────────────────────────────────────────────────────────

class TestStepBeforeReset:

    def test_step_before_reset_raises_runtime_error(self, env):
        """step() called before any reset() must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="No active episode"):
            env.step(_skip("alert-001"))


# ─────────────────────────────────────────────────────────────────────────────
# step() — valid actions
# ─────────────────────────────────────────────────────────────────────────────

class TestStepValidActions:

    def test_triage_returns_step_result_type(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = easy_env.step(_triage(obs.alerts[0].alert_id))
        assert isinstance(result, StepResult)

    def test_triage_decreases_pending_count(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = easy_env.step(_triage(obs.alerts[0].alert_id))
        assert result.observation.pending_count == 4

    def test_triage_advances_step_number(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = easy_env.step(_triage(obs.alerts[0].alert_id))
        assert result.observation.step_number == 1

    def test_triage_reward_is_float(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = easy_env.step(_triage(obs.alerts[0].alert_id))
        assert isinstance(result.reward, float)

    def test_triage_done_false_mid_episode(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = easy_env.step(_triage(obs.alerts[0].alert_id))
        assert result.done is False

    def test_triage_marks_alert_triaged(self, easy_env):
        obs = easy_env.reset("easy", 42)
        aid = obs.alerts[0].alert_id
        easy_env.step(_triage(aid))
        # Observation returned by step should show the alert as triaged
        result = easy_env.step(_triage(obs.alerts[1].alert_id))
        triaged_ids = {a.alert_id for a in result.observation.alerts if a.triaged}
        assert aid in triaged_ids

    def test_skip_decreases_pending(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = easy_env.step(_skip(obs.alerts[0].alert_id))
        assert result.observation.pending_count == 4

    def test_link_alerts_does_not_decrease_pending(self, easy_env):
        """link_alerts is a grouping action; it does not consume/triage alerts."""
        obs = easy_env.reset("easy", 42)
        result = easy_env.step({
            "action_type":    "link_alerts",
            "alert_ids":      [obs.alerts[0].alert_id, obs.alerts[1].alert_id],
            "incident_label": "test-incident",
        })
        assert result.observation.pending_count == 5   # unchanged

    def test_link_alerts_advances_step(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = easy_env.step({
            "action_type":    "link_alerts",
            "alert_ids":      [obs.alerts[0].alert_id, obs.alerts[1].alert_id],
            "incident_label": "test-incident",
        })
        assert result.observation.step_number == 1

    def test_feedback_is_string(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = easy_env.step(_triage(obs.alerts[0].alert_id))
        assert isinstance(result.observation.feedback, str)
        assert len(result.observation.feedback) > 0

    def test_info_empty_mid_episode(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = easy_env.step(_triage(obs.alerts[0].alert_id))
        assert result.info == {}


# ─────────────────────────────────────────────────────────────────────────────
# step() — edge cases / penalties
# ─────────────────────────────────────────────────────────────────────────────

class TestStepEdgeCases:

    def test_double_triage_returns_negative_reward(self, easy_env):
        """Triaging the same alert twice returns the −0.15 penalty."""
        obs = easy_env.reset("easy", 42)
        aid = obs.alerts[0].alert_id
        easy_env.step(_triage(aid))
        result = easy_env.step(_triage(aid))
        assert result.reward == pytest.approx(-0.15)

    def test_double_triage_does_not_decrease_pending(self, easy_env):
        """Second triage on the same alert must not double-count."""
        obs = easy_env.reset("easy", 42)
        aid = obs.alerts[0].alert_id
        easy_env.step(_triage(aid))
        result = easy_env.step(_triage(aid))
        # pending_count should still be 4 (not 3)
        assert result.observation.pending_count == 4

    def test_double_skip_returns_negative_reward(self, easy_env):
        """Skipping the same alert twice returns the −0.15 penalty."""
        obs = easy_env.reset("easy", 42)
        aid = obs.alerts[0].alert_id
        easy_env.step(_skip(aid))
        result = easy_env.step(_skip(aid))
        assert result.reward == pytest.approx(-0.15)

    def test_invalid_alert_id_triage_returns_penalty(self, easy_env):
        easy_env.reset("easy", 42)
        result = easy_env.step(_triage("alert-NONEXISTENT"))
        assert result.reward == pytest.approx(-0.10)

    def test_invalid_alert_id_skip_returns_penalty(self, easy_env):
        easy_env.reset("easy", 42)
        result = easy_env.step(_skip("alert-NONEXISTENT"))
        assert result.reward == pytest.approx(-0.10)

    def test_invalid_alert_id_link_returns_penalty(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = easy_env.step({
            "action_type":    "link_alerts",
            "alert_ids":      [obs.alerts[0].alert_id, "alert-NONEXISTENT"],
            "incident_label": "bad-link",
        })
        assert result.reward == pytest.approx(-0.10)

    def test_malformed_action_dict_returns_penalty(self, easy_env):
        """A dict missing required triage fields triggers −0.10."""
        easy_env.reset("easy", 42)
        result = easy_env.step({
            "action_type": "triage",
            # missing alert_id, root_cause, severity, remediation
        })
        assert result.reward == pytest.approx(-0.10)

    def test_invalid_root_cause_value_returns_penalty(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = easy_env.step({
            "action_type": "triage",
            "alert_id":    obs.alerts[0].alert_id,
            "root_cause":  "totally_made_up",
            "severity":    "high",
            "remediation": "scale_up",
        })
        assert result.reward == pytest.approx(-0.10)

    def test_step_after_done_returns_done_no_penalty(self, easy_env):
        """Once done, further steps return done=True with reward=0.0."""
        obs = easy_env.reset("easy", 42)
        for alert in obs.alerts:
            easy_env.step(_triage(alert.alert_id))
        result = easy_env.step(_skip(obs.alerts[0].alert_id))
        assert result.done is True
        assert result.reward == 0.0

    def test_budget_exhaustion_sets_done(self):
        """Exhausting the step budget (max_steps=10) sets done=True."""
        env = AlertTriageEnv()
        env.reset("easy", 42)
        # easy max_steps = 10; take 10 steps (link_alerts doesn't triage anything)
        for _ in range(10):
            result = env.step({
                "action_type": "link_alerts",
                "alert_ids":   ["alert-001", "alert-002"],
                "incident_label": "x",
            })
        assert result.done is True


# ─────────────────────────────────────────────────────────────────────────────
# step() — full episode completion
# ─────────────────────────────────────────────────────────────────────────────

class TestFullEpisode:

    def test_easy_completes_after_five_triages(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = None
        for alert in obs.alerts:
            result = easy_env.step(_triage(alert.alert_id))
        assert result.done is True
        assert result.observation.pending_count == 0

    def test_grader_score_in_info_when_done(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = None
        for alert in obs.alerts:
            result = easy_env.step(_triage(alert.alert_id))
        assert "grader_score" in result.info
        assert isinstance(result.info["grader_score"], float)
        assert 0.0 <= result.info["grader_score"] <= 1.0

    def test_info_empty_before_done(self, easy_env):
        obs = easy_env.reset("easy", 42)
        result = easy_env.step(_triage(obs.alerts[0].alert_id))
        assert result.info == {}

    def test_cumulative_reward_tracked_across_steps(self, easy_env):
        """Cumulative reward reflects all steps including any penalties."""
        obs = easy_env.reset("easy", 42)
        aid = obs.alerts[0].alert_id
        easy_env.step(_triage(aid))
        after_step1 = easy_env.state().cumulative_reward
        easy_env.step(_triage(aid))       # double triage: penalty = -0.15
        state = easy_env.state()
        assert state.cumulative_reward == pytest.approx(after_step1 - 0.15)

    def test_reset_after_completion_starts_fresh(self, easy_env):
        obs = easy_env.reset("easy", 42)
        for alert in obs.alerts:
            easy_env.step(_triage(alert.alert_id))
        # Re-reset
        obs2 = easy_env.reset("easy", 42)
        assert obs2.step_number == 0
        assert obs2.pending_count == 5
        assert all(not a.triaged for a in obs2.alerts)


# ─────────────────────────────────────────────────────────────────────────────
# state()
# ─────────────────────────────────────────────────────────────────────────────

class TestState:

    def test_state_returns_environment_state_type(self, easy_env):
        easy_env.reset("easy", 42)
        assert isinstance(easy_env.state(), EnvironmentState)

    def test_state_task_id_and_seed(self, easy_env):
        easy_env.reset("easy", 42)
        s = easy_env.state()
        assert s.task_id == "easy"
        assert s.seed == 42

    def test_state_alert_count(self, easy_env):
        easy_env.reset("easy", 42)
        s = easy_env.state()
        assert len(s.alerts) == 5

    def test_state_ground_truth_count(self, easy_env):
        easy_env.reset("easy", 42)
        s = easy_env.state()
        assert len(s.ground_truth) == 5

    def test_state_ground_truth_fields(self, easy_env):
        easy_env.reset("easy", 42)
        s = easy_env.state()
        for gt in s.ground_truth:
            assert "alert_id" in gt
            assert "true_root_cause" in gt
            assert "true_severity" in gt
            assert "true_remediation" in gt

    def test_state_incidents_present(self):
        env = AlertTriageEnv()
        env.reset("medium", 42)
        s = env.state()
        assert len(s.incidents) == 2

    def test_state_done_false_initially(self, easy_env):
        easy_env.reset("easy", 42)
        assert easy_env.state().done is False

    def test_state_done_true_after_completion(self, easy_env):
        obs = easy_env.reset("easy", 42)
        for alert in obs.alerts:
            easy_env.step(_triage(alert.alert_id))
        assert easy_env.state().done is True

    def test_state_grader_score_none_mid_episode(self, easy_env):
        easy_env.reset("easy", 42)
        assert easy_env.state().grader_score is None

    def test_state_grader_score_populated_when_done(self, easy_env):
        obs = easy_env.reset("easy", 42)
        for alert in obs.alerts:
            easy_env.step(_triage(alert.alert_id))
        s = easy_env.state()
        assert s.grader_score is not None
        assert 0.0 <= s.grader_score <= 1.0

    def test_state_cumulative_reward_with_penalty(self, easy_env):
        """Verifies cumulative_reward is tracked via the -0.15 double-triage path."""
        obs = easy_env.reset("easy", 42)
        aid = obs.alerts[0].alert_id
        easy_env.step(_triage(aid))
        after_step1 = easy_env.state().cumulative_reward
        easy_env.step(_triage(aid))   # -0.15 penalty
        assert easy_env.state().cumulative_reward == pytest.approx(after_step1 - 0.15)

    def test_state_agent_decisions_recorded(self, easy_env):
        obs = easy_env.reset("easy", 42)
        easy_env.step(_triage(obs.alerts[0].alert_id))
        easy_env.step(_triage(obs.alerts[1].alert_id))
        s = easy_env.state()
        assert len(s.agent_decisions) == 2

    def test_state_step_number_increments(self, easy_env):
        obs = easy_env.reset("easy", 42)
        easy_env.step(_triage(obs.alerts[0].alert_id))
        easy_env.step(_triage(obs.alerts[1].alert_id))
        assert easy_env.state().step_number == 2

    def test_state_max_steps_correct(self, easy_env):
        easy_env.reset("easy", 42)
        assert easy_env.state().max_steps == 10
