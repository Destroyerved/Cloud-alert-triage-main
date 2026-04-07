"""
server/config.py
All constants, enums, and configuration for the Cloud Alert Triage environment.
"""

# Valid root cause categories for alert classification
ROOT_CAUSE_CATEGORIES: list[str] = [
    "resource_exhaustion",
    "network_failure",
    "deployment_bug",
    "config_error",
    "dependency_outage",
    "false_alarm",
]

# Severity levels (ordered from most to least severe)
SEVERITY_LEVELS: list[str] = ["critical", "high", "medium", "low"]

# Valid remediation actions
REMEDIATION_ACTIONS: list[str] = [
    "restart_service",
    "scale_up",
    "rollback_deploy",
    "fix_config",
    "escalate_to_team",
    "acknowledge_and_monitor",
    "dismiss",
]

# Valid action types for the agent
ACTION_TYPES: list[str] = ["triage", "link_alerts", "skip"]

# Default server port (Hugging Face Spaces standard)
DEFAULT_PORT: int = 7860

# Maximum steps allowed per task
MAX_STEPS_BY_TASK: dict[str, int] = {
    "easy": 10,
    "medium": 25,
    "hard": 45,
}

# Severity numeric rank for proximity scoring (lower = more severe)
SEVERITY_ORDER: dict[str, int] = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}

# ---------------------------------------------------------------------------
# Dynamic cascade mechanic
# ---------------------------------------------------------------------------
# After CASCADE_TRIGGER_STEP environment steps, any original critical/high
# alert that is still untriaged spawns exactly one new dependent alert on a
# downstream service.  Dynamic alerts participate in per-step rewards but are
# excluded from the final grader score for deadline safety.
CASCADE_TRIGGER_STEP: int = 5
CASCADE_ELIGIBLE_SEVERITIES: frozenset[str] = frozenset({"critical", "high"})
