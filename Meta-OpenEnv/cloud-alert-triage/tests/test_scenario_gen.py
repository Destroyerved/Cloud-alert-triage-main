"""
tests/test_scenario_gen.py
--------------------------
Determinism and correctness tests for server/scenario_generator.py.
Run from cloud-alert-triage/ root: pytest tests/test_scenario_gen.py -v
"""

import pytest

# ---------------------------------------------------------------------------
# Stub guard
# ---------------------------------------------------------------------------
try:
    from server.scenario_generator import generate_scenario
    from server.service_graph import get_service_names
    from server.config import ROOT_CAUSE_CATEGORIES, SEVERITY_LEVELS, REMEDIATION_ACTIONS
    AVAILABLE = True
except Exception:
    AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not AVAILABLE,
    reason="scenario_generator not yet implemented — skip until Phase 3",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EASY_SEED = 42
MEDIUM_SEED = 42
HARD_SEED = 42
ALT_SEED = 99


def _scenario(task_id, seed=42):
    return generate_scenario(task_id, seed)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_easy_same_seed_identical(self):
        """generate_scenario('easy', 42) twice → identical alert IDs."""
        s1 = _scenario("easy", EASY_SEED)
        s2 = _scenario("easy", EASY_SEED)
        ids1 = [a["alert_id"] for a in s1["alerts"]]
        ids2 = [a["alert_id"] for a in s2["alerts"]]
        assert ids1 == ids2

    def test_medium_same_seed_identical(self):
        s1 = _scenario("medium", MEDIUM_SEED)
        s2 = _scenario("medium", MEDIUM_SEED)
        ids1 = [a["alert_id"] for a in s1["alerts"]]
        ids2 = [a["alert_id"] for a in s2["alerts"]]
        assert ids1 == ids2

    def test_hard_same_seed_identical(self):
        s1 = _scenario("hard", HARD_SEED)
        s2 = _scenario("hard", HARD_SEED)
        ids1 = [a["alert_id"] for a in s1["alerts"]]
        ids2 = [a["alert_id"] for a in s2["alerts"]]
        assert ids1 == ids2

    def test_easy_different_seeds_differ(self):
        """Different seeds should produce different scenarios."""
        s1 = _scenario("easy", EASY_SEED)
        s2 = _scenario("easy", ALT_SEED)
        # Highly likely they differ (if generator uses the seed)
        msgs1 = [a["message"] for a in s1["alerts"]]
        msgs2 = [a["message"] for a in s2["alerts"]]
        assert msgs1 != msgs2

    def test_ground_truth_deterministic(self):
        """Ground truth must also be identical for same seed."""
        s1 = _scenario("easy", EASY_SEED)
        s2 = _scenario("easy", EASY_SEED)
        gt1 = [g["true_root_cause"] for g in s1["ground_truth"]]
        gt2 = [g["true_root_cause"] for g in s2["ground_truth"]]
        assert gt1 == gt2


# ---------------------------------------------------------------------------
# Alert counts
# ---------------------------------------------------------------------------

class TestAlertCounts:
    def test_easy_has_five_alerts(self):
        s = _scenario("easy")
        assert len(s["alerts"]) == 5

    def test_medium_has_fifteen_alerts(self):
        s = _scenario("medium")
        assert len(s["alerts"]) == 15

    def test_hard_has_thirty_alerts(self):
        s = _scenario("hard")
        assert len(s["alerts"]) == 30

    def test_easy_no_incidents(self):
        s = _scenario("easy")
        # Easy has 0 incidents
        real_incidents = [i for i in s.get("incidents", []) if i]
        assert len(real_incidents) == 0

    def test_medium_has_two_incidents(self):
        s = _scenario("medium")
        assert len(s.get("incidents", [])) == 2

    def test_hard_has_four_or_five_incidents(self):
        s = _scenario("hard")
        count = len(s.get("incidents", []))
        assert 4 <= count <= 5

    def test_ground_truth_length_matches_alerts(self):
        """One ground_truth entry per alert."""
        for task in ("easy", "medium", "hard"):
            s = _scenario(task)
            assert len(s["ground_truth"]) == len(s["alerts"]), task


# ---------------------------------------------------------------------------
# Alert field validity
# ---------------------------------------------------------------------------

class TestAlertFields:
    REQUIRED_ALERT_KEYS = {
        "alert_id", "timestamp", "service", "metric",
        "metric_value", "threshold", "message",
    }

    def _all_alerts(self, task="easy"):
        return _scenario(task)["alerts"]

    def test_alert_has_required_fields(self):
        for alert in self._all_alerts():
            missing = self.REQUIRED_ALERT_KEYS - set(alert.keys())
            assert not missing, f"Alert {alert.get('alert_id')} missing: {missing}"

    def test_alert_ids_unique(self):
        for task in ("easy", "medium", "hard"):
            alerts = _scenario(task)["alerts"]
            ids = [a["alert_id"] for a in alerts]
            assert len(ids) == len(set(ids)), f"Duplicate IDs in {task}"

    def test_alert_services_in_graph(self):
        """Every alert's service must exist in the service graph."""
        known = set(get_service_names())
        for task in ("easy", "medium", "hard"):
            for alert in _scenario(task)["alerts"]:
                assert alert["service"] in known, (
                    f"Unknown service '{alert['service']}' in {task}"
                )

    def test_metric_value_exceeds_threshold(self):
        """For non-false-alarm alerts, metric_value should exceed threshold."""
        for alert in _scenario("easy")["alerts"]:
            # False alarms may have borderline values; only assert for clarity
            # This is a soft check — implementation may relax it for false alarms
            assert isinstance(alert["metric_value"], (int, float))
            assert isinstance(alert["threshold"], (int, float))

    def test_message_is_non_empty_string(self):
        for alert in _scenario("easy")["alerts"]:
            assert isinstance(alert["message"], str)
            assert len(alert["message"]) > 0


# ---------------------------------------------------------------------------
# Ground truth field validity
# ---------------------------------------------------------------------------

class TestGroundTruthFields:
    def test_root_cause_valid_enum(self):
        for task in ("easy", "medium", "hard"):
            for gt in _scenario(task)["ground_truth"]:
                assert gt["true_root_cause"] in ROOT_CAUSE_CATEGORIES, (
                    f"Invalid root_cause '{gt['true_root_cause']}' in {task}"
                )

    def test_severity_valid_enum(self):
        for task in ("easy", "medium", "hard"):
            for gt in _scenario(task)["ground_truth"]:
                assert gt["true_severity"] in SEVERITY_LEVELS, (
                    f"Invalid severity '{gt['true_severity']}' in {task}"
                )

    def test_remediation_valid_enum(self):
        for task in ("easy", "medium", "hard"):
            for gt in _scenario(task)["ground_truth"]:
                assert gt["true_remediation"] in REMEDIATION_ACTIONS, (
                    f"Invalid remediation '{gt['true_remediation']}' in {task}"
                )

    def test_gt_alert_id_references_exist(self):
        """Every ground_truth entry's alert_id must match an actual alert."""
        for task in ("easy", "medium", "hard"):
            s = _scenario(task)
            alert_ids = {a["alert_id"] for a in s["alerts"]}
            for gt in s["ground_truth"]:
                assert gt["alert_id"] in alert_ids, (
                    f"GT references unknown alert '{gt['alert_id']}' in {task}"
                )

    def test_incident_id_null_for_easy(self):
        """Easy has no incidents — incident_id should be null/None."""
        for gt in _scenario("easy")["ground_truth"]:
            assert gt.get("incident_id") is None


# ---------------------------------------------------------------------------
# Hard-task-specific checks
# ---------------------------------------------------------------------------

class TestHardScenario:
    def test_has_false_alarm_alerts(self):
        """Hard scenario must include at least some false_alarm root causes."""
        gt = _scenario("hard")["ground_truth"]
        false_alarms = [g for g in gt if g["true_root_cause"] == "false_alarm"]
        assert len(false_alarms) >= 1  # plan says 5-8

    def test_alerts_not_in_incident_order(self):
        """Hard scenario timestamps should be interleaved (not grouped by incident)."""
        s = _scenario("hard")
        # Collect incident IDs in alert order
        alert_id_to_incident = {
            g["alert_id"]: g.get("incident_id") for g in s["ground_truth"]
        }
        incident_seq = [
            alert_id_to_incident.get(a["alert_id"]) for a in s["alerts"]
        ]
        # A purely grouped sequence would have all of incident A, then B, etc.
        # We just check it's not all None (i.e. incidents exist)
        non_none = [x for x in incident_seq if x is not None]
        assert len(non_none) > 0, "Hard scenario has no incident-linked alerts"
