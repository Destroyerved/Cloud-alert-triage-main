"""
server/models.py
All Pydantic v2 models for the Cloud Alert Triage environment.
Implemented in Phase 2.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator, model_validator

from server.config import (
    ACTION_TYPES,
    REMEDIATION_ACTIONS,
    ROOT_CAUSE_CATEGORIES,
    SEVERITY_LEVELS,
)


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    """Represents a single monitoring alert emitted by the infrastructure."""

    alert_id: str
    timestamp: str                          # ISO-8601 string, e.g. "2024-01-15T10:23:00Z"
    service: str                            # originating microservice name
    metric: str                             # e.g. "cpu_usage_percent"
    metric_value: float                     # observed value
    threshold: float                        # threshold that was breached
    message: str                            # human-readable alert text
    context: str | None = None              # optional extra context (recent deploy, etc.)
    triaged: bool = False                   # True once the agent has acted on this alert
    agent_decision: dict[str, Any] | None = None  # agent's triage result if triaged


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Everything the agent sees after reset() or step()."""

    alerts: list[Alert]
    service_map: dict[str, list[str]]       # adjacency list: service → its dependencies
    pending_count: int                      # number of un-triaged alerts remaining
    step_number: int                        # current step (0-indexed)
    max_steps: int                          # step budget for this episode
    feedback: str = ""                      # short hint after the last action


# ---------------------------------------------------------------------------
# Action  (single model with action_type discriminator)
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    Unified action model.  action_type is the discriminator; required fields
    differ per type and are enforced by the model_validator below.

    triage     → alert_id, root_cause, severity, remediation
    link_alerts→ alert_ids (≥2 items), incident_label
    skip       → alert_id
    """

    action_type: str

    # triage fields
    alert_id: str | None = None
    root_cause: str | None = None
    severity: str | None = None
    remediation: str | None = None

    # link_alerts fields
    alert_ids: list[str] | None = None
    incident_label: str | None = None

    # ── field-level enum validators ─────────────────────────────────────────

    @field_validator("action_type")
    @classmethod
    def _check_action_type(cls, v: str) -> str:
        if v not in ACTION_TYPES:
            raise ValueError(
                f"action_type must be one of {ACTION_TYPES}, got '{v}'"
            )
        return v

    @field_validator("root_cause")
    @classmethod
    def _check_root_cause(cls, v: str | None) -> str | None:
        if v is not None and v not in ROOT_CAUSE_CATEGORIES:
            raise ValueError(
                f"root_cause must be one of {ROOT_CAUSE_CATEGORIES}, got '{v}'"
            )
        return v

    @field_validator("severity")
    @classmethod
    def _check_severity(cls, v: str | None) -> str | None:
        if v is not None and v not in SEVERITY_LEVELS:
            raise ValueError(
                f"severity must be one of {SEVERITY_LEVELS}, got '{v}'"
            )
        return v

    @field_validator("remediation")
    @classmethod
    def _check_remediation(cls, v: str | None) -> str | None:
        if v is not None and v not in REMEDIATION_ACTIONS:
            raise ValueError(
                f"remediation must be one of {REMEDIATION_ACTIONS}, got '{v}'"
            )
        return v

    # ── cross-field validator: required fields per action_type ──────────────

    @model_validator(mode="after")
    def _check_required_fields(self) -> Action:
        if self.action_type == "triage":
            missing = [
                name
                for name in ("alert_id", "root_cause", "severity", "remediation")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"triage action is missing required fields: {missing}"
                )

        elif self.action_type == "link_alerts":
            if self.alert_ids is None:
                raise ValueError("link_alerts action requires 'alert_ids'")
            if len(self.alert_ids) < 2:
                raise ValueError(
                    "link_alerts requires at least 2 alert_ids, "
                    f"got {len(self.alert_ids)}"
                )
            if self.incident_label is None:
                raise ValueError("link_alerts action requires 'incident_label'")

        elif self.action_type == "skip":
            if self.alert_id is None:
                raise ValueError("skip action requires 'alert_id'")

        return self


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Response returned by step()."""

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = {}              # includes grader_score when done=True


# ---------------------------------------------------------------------------
# EnvironmentState
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    """
    Full internal state including hidden ground truth.
    Returned by GET /state.  Not exposed to the agent during inference.
    """

    task_id: str
    seed: int
    step_number: int
    max_steps: int
    done: bool
    alerts: list[Alert]
    ground_truth: list[dict[str, Any]]     # one entry per alert with true labels
    agent_decisions: list[dict[str, Any]]  # recorded decisions from agent
    incidents: list[dict[str, Any]]        # true incident groupings
    cumulative_reward: float
    grader_score: float | None = None      # populated at episode end


# ---------------------------------------------------------------------------
# ResetRequest
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Request body for POST /reset."""

    task_id: str = "easy"
    seed: int = 42


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------

class TaskConfig(BaseModel):
    """Static task metadata loaded from tasks/task_*.json."""

    task_id: str
    title: str
    description: str
    difficulty: str
    default_seed: int
    num_alerts: int
    num_incidents: int
    noise_alerts: int
    max_steps: int
