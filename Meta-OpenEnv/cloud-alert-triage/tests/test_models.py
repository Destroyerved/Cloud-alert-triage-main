"""
tests/test_models.py
Tests for all Pydantic v2 models in server/models.py.
Run with: pytest tests/test_models.py -v
"""

import pytest
from pydantic import ValidationError

from server.models import (
    Alert,
    Action,
    Observation,
    StepResult,
    EnvironmentState,
    ResetRequest,
    TaskConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _alert(alert_id: str = "alert-001") -> Alert:
    return Alert(
        alert_id=alert_id,
        timestamp="2024-01-15T10:23:00Z",
        service="api-gateway",
        metric="cpu_usage_percent",
        metric_value=94.5,
        threshold=80.0,
        message="CPU spike on api-gateway",
    )


def _obs(alerts: list[Alert] | None = None, step: int = 0) -> Observation:
    return Observation(
        alerts=alerts or [],
        service_map={"api-gateway": ["user-service"], "user-service": []},
        pending_count=len(alerts or []),
        step_number=step,
        max_steps=10,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Alert
# ─────────────────────────────────────────────────────────────────────────────

class TestAlert:

    def test_valid_minimal(self):
        a = _alert()
        assert a.alert_id == "alert-001"
        assert a.triaged is False
        assert a.context is None
        assert a.agent_decision is None

    def test_valid_all_fields(self):
        a = Alert(
            alert_id="alert-002",
            timestamp="2024-01-15T10:24:00Z",
            service="order-service",
            metric="error_rate_percent",
            metric_value=15.2,
            threshold=5.0,
            message="Error rate spike",
            context="Deploy v2.3.1 rolled out 10 min ago",
            triaged=True,
            agent_decision={"root_cause": "deployment_bug", "severity": "high"},
        )
        assert a.triaged is True
        assert a.agent_decision["root_cause"] == "deployment_bug"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            # 'service' is omitted
            Alert(
                alert_id="alert-003",
                timestamp="2024-01-15T10:00:00Z",
                metric="cpu_usage_percent",
                metric_value=50.0,
                threshold=80.0,
                message="test",
            )

    def test_wrong_type_for_metric_value(self):
        with pytest.raises(ValidationError):
            Alert(
                alert_id="alert-004",
                timestamp="2024-01-15T10:00:00Z",
                service="api-gateway",
                metric="cpu_usage_percent",
                metric_value="not-a-float",   # wrong type
                threshold=80.0,
                message="test",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Action — valid constructions
# ─────────────────────────────────────────────────────────────────────────────

class TestActionValid:

    def test_valid_triage(self):
        a = Action(
            action_type="triage",
            alert_id="alert-001",
            root_cause="resource_exhaustion",
            severity="high",
            remediation="scale_up",
        )
        assert a.action_type == "triage"
        assert a.root_cause == "resource_exhaustion"
        assert a.severity == "high"
        assert a.remediation == "scale_up"

    def test_valid_link_alerts(self):
        a = Action(
            action_type="link_alerts",
            alert_ids=["alert-001", "alert-002", "alert-003"],
            incident_label="incident-alpha",
        )
        assert len(a.alert_ids) == 3
        assert a.incident_label == "incident-alpha"

    def test_valid_skip(self):
        a = Action(action_type="skip", alert_id="alert-005")
        assert a.action_type == "skip"
        assert a.alert_id == "alert-005"

    def test_triage_all_root_cause_values(self):
        valid_causes = [
            "resource_exhaustion", "network_failure", "deployment_bug",
            "config_error", "dependency_outage", "false_alarm",
        ]
        for cause in valid_causes:
            a = Action(
                action_type="triage",
                alert_id="alert-001",
                root_cause=cause,
                severity="medium",
                remediation="acknowledge_and_monitor",
            )
            assert a.root_cause == cause

    def test_triage_all_severity_values(self):
        for sev in ("critical", "high", "medium", "low"):
            a = Action(
                action_type="triage",
                alert_id="alert-001",
                root_cause="config_error",
                severity=sev,
                remediation="fix_config",
            )
            assert a.severity == sev

    def test_triage_all_remediation_values(self):
        valid_remediations = [
            "restart_service", "scale_up", "rollback_deploy", "fix_config",
            "escalate_to_team", "acknowledge_and_monitor", "dismiss",
        ]
        for rem in valid_remediations:
            a = Action(
                action_type="triage",
                alert_id="alert-001",
                root_cause="network_failure",
                severity="low",
                remediation=rem,
            )
            assert a.remediation == rem


# ─────────────────────────────────────────────────────────────────────────────
# Action — invalid constructions (enum violations)
# ─────────────────────────────────────────────────────────────────────────────

class TestActionInvalidEnums:

    def test_invalid_action_type(self):
        with pytest.raises(ValidationError):
            Action(action_type="do_nothing")

    def test_invalid_root_cause(self):
        with pytest.raises(ValidationError):
            Action(
                action_type="triage",
                alert_id="alert-001",
                root_cause="aliens",            # not a valid category
                severity="high",
                remediation="scale_up",
            )

    def test_invalid_severity(self):
        with pytest.raises(ValidationError):
            Action(
                action_type="triage",
                alert_id="alert-001",
                root_cause="resource_exhaustion",
                severity="urgent",              # not a valid level
                remediation="scale_up",
            )

    def test_invalid_remediation(self):
        with pytest.raises(ValidationError):
            Action(
                action_type="triage",
                alert_id="alert-001",
                root_cause="resource_exhaustion",
                severity="high",
                remediation="pray",             # not a valid action
            )


# ─────────────────────────────────────────────────────────────────────────────
# Action — missing required fields per action_type
# ─────────────────────────────────────────────────────────────────────────────

class TestActionMissingFields:

    def test_triage_missing_alert_id(self):
        with pytest.raises(ValidationError):
            Action(
                action_type="triage",
                root_cause="resource_exhaustion",
                severity="high",
                remediation="scale_up",
            )

    def test_triage_missing_root_cause(self):
        with pytest.raises(ValidationError):
            Action(
                action_type="triage",
                alert_id="alert-001",
                severity="high",
                remediation="scale_up",
            )

    def test_triage_missing_severity(self):
        with pytest.raises(ValidationError):
            Action(
                action_type="triage",
                alert_id="alert-001",
                root_cause="resource_exhaustion",
                remediation="scale_up",
            )

    def test_triage_missing_remediation(self):
        with pytest.raises(ValidationError):
            Action(
                action_type="triage",
                alert_id="alert-001",
                root_cause="resource_exhaustion",
                severity="high",
            )

    def test_link_alerts_missing_alert_ids(self):
        with pytest.raises(ValidationError):
            Action(action_type="link_alerts", incident_label="inc-1")

    def test_link_alerts_only_one_id(self):
        with pytest.raises(ValidationError):
            Action(
                action_type="link_alerts",
                alert_ids=["alert-001"],        # need ≥ 2
                incident_label="inc-1",
            )

    def test_link_alerts_missing_label(self):
        with pytest.raises(ValidationError):
            Action(
                action_type="link_alerts",
                alert_ids=["alert-001", "alert-002"],
            )

    def test_skip_missing_alert_id(self):
        with pytest.raises(ValidationError):
            Action(action_type="skip")


# ─────────────────────────────────────────────────────────────────────────────
# Observation — JSON round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestObservation:

    def test_valid_observation(self):
        obs = _obs([_alert("alert-001"), _alert("alert-002")])
        assert obs.pending_count == 2
        assert obs.feedback == ""
        assert obs.step_number == 0

    def test_json_round_trip(self):
        obs = Observation(
            alerts=[_alert("alert-001"), _alert("alert-002")],
            service_map={
                "api-gateway": ["user-service", "order-service"],
                "user-service": ["postgres-primary"],
                "order-service": [],
            },
            pending_count=2,
            step_number=3,
            max_steps=10,
            feedback="Consider checking upstream dependencies",
        )
        json_str = obs.model_dump_json()
        restored = Observation.model_validate_json(json_str)
        assert restored == obs
        assert restored.feedback == "Consider checking upstream dependencies"
        assert len(restored.alerts) == 2
        assert restored.service_map["api-gateway"] == ["user-service", "order-service"]


# ─────────────────────────────────────────────────────────────────────────────
# StepResult
# ─────────────────────────────────────────────────────────────────────────────

class TestStepResult:

    def test_valid_in_progress(self):
        sr = StepResult(observation=_obs(), reward=0.8, done=False)
        assert sr.reward == 0.8
        assert sr.done is False
        assert sr.info == {}

    def test_valid_done_with_grader_score(self):
        sr = StepResult(
            observation=_obs(),
            reward=0.5,
            done=True,
            info={"grader_score": 0.92},
        )
        assert sr.done is True
        assert sr.info["grader_score"] == 0.92


# ─────────────────────────────────────────────────────────────────────────────
# EnvironmentState
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvironmentState:

    def test_valid_state_in_progress(self):
        state = EnvironmentState(
            task_id="easy",
            seed=42,
            step_number=3,
            max_steps=10,
            done=False,
            alerts=[_alert()],
            ground_truth=[{"alert_id": "alert-001", "true_root_cause": "resource_exhaustion"}],
            agent_decisions=[],
            incidents=[],
            cumulative_reward=1.6,
        )
        assert state.grader_score is None
        assert state.cumulative_reward == 1.6
        assert state.task_id == "easy"

    def test_valid_state_done(self):
        state = EnvironmentState(
            task_id="hard",
            seed=99,
            step_number=45,
            max_steps=45,
            done=True,
            alerts=[],
            ground_truth=[],
            agent_decisions=[],
            incidents=[],
            cumulative_reward=12.3,
            grader_score=0.72,
        )
        assert state.done is True
        assert state.grader_score == 0.72


# ─────────────────────────────────────────────────────────────────────────────
# ResetRequest
# ─────────────────────────────────────────────────────────────────────────────

class TestResetRequest:

    def test_defaults(self):
        rr = ResetRequest()
        assert rr.task_id == "easy"
        assert rr.seed == 42

    def test_custom(self):
        rr = ResetRequest(task_id="hard", seed=99)
        assert rr.task_id == "hard"
        assert rr.seed == 99


# ─────────────────────────────────────────────────────────────────────────────
# TaskConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskConfig:

    def test_valid_task_config(self):
        tc = TaskConfig(
            task_id="medium",
            title="Correlated Incident Response",
            description="15 alerts with 2 correlated incidents and false alarms.",
            difficulty="medium",
            default_seed=42,
            num_alerts=15,
            num_incidents=2,
            noise_alerts=2,
            max_steps=25,
        )
        assert tc.task_id == "medium"
        assert tc.num_alerts == 15
        assert tc.max_steps == 25

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            TaskConfig(
                task_id="easy",
                # title missing
                description="test",
                difficulty="easy",
                default_seed=42,
                num_alerts=5,
                num_incidents=0,
                noise_alerts=0,
                max_steps=10,
            )
