"""
tests/test_api.py

FastAPI endpoint integration tests using FastAPI TestClient.
Run from cloud-alert-triage/ root:  pytest tests/test_api.py -v

Test groups
-----------
    TestHealth          -- GET /health
    TestReset           -- POST /reset
    TestStep            -- POST /step
    TestStepBeforeReset -- POST /step before any /reset
    TestFullEpisode     -- complete episode via HTTP
    TestState           -- GET /state

Fixture design
--------------
- ``client``  (module-scoped)  -- one TestClient for the whole module.
- ``fresh``   (function-scoped) -- calls /reset(easy, 42) before each test
                                   that needs a clean episode.
- ``first_alert_id`` helper   -- returns the first alert_id from a reset.

NOTE: The TestClient runs the ASGI app in-process; no network is involved.
      The global ``env`` in server/app.py is shared across all tests in a
      module-scoped client, so tests that mutate episode state should call
      /reset explicitly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import app

# ---------------------------------------------------------------------------
# Module-level client (single instance; faster than recreating per test)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_easy(client: TestClient, seed: int = 42) -> dict:
    """POST /reset for easy task and return parsed response body."""
    resp = client.post("/reset", json={"task_id": "easy", "seed": seed})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _first_alert_id(client: TestClient, seed: int = 42) -> str:
    data = _reset_easy(client, seed)
    obs = data.get("observation", data)
    return obs["alerts"][0]["alert_id"]


def _all_alert_ids(client: TestClient, seed: int = 42) -> list[str]:
    data = _reset_easy(client, seed)
    obs = data.get("observation", data)
    return [a["alert_id"] for a in obs["alerts"]]


# ---------------------------------------------------------------------------
# TestHealth
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_200(self, client):
        """GET /health → 200."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self, client):
        """GET /health body must be {\"status\": \"ok\"}."""
        data = client.get("/health").json()
        assert data.get("status") == "ok"

    def test_health_no_side_effects(self, client):
        """Calling /health twice must both succeed."""
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:

    def test_reset_easy_200(self, client):
        resp = client.post("/reset", json={"task_id": "easy", "seed": 42})
        assert resp.status_code == 200

    def test_reset_medium_200(self, client):
        resp = client.post("/reset", json={"task_id": "medium", "seed": 1})
        assert resp.status_code == 200

    def test_reset_hard_200(self, client):
        resp = client.post("/reset", json={"task_id": "hard", "seed": 7})
        assert resp.status_code == 200

    def test_reset_response_has_observation_key(self, client):
        data = _reset_easy(client)
        assert "observation" in data

    def test_reset_observation_has_alerts(self, client):
        obs = _reset_easy(client)["observation"]
        assert "alerts" in obs
        assert isinstance(obs["alerts"], list)
        assert len(obs["alerts"]) > 0

    def test_reset_easy_has_five_alerts(self, client):
        obs = _reset_easy(client)["observation"]
        assert len(obs["alerts"]) == 5

    def test_reset_medium_has_fifteen_alerts(self, client):
        resp = client.post("/reset", json={"task_id": "medium", "seed": 42})
        obs = resp.json()["observation"]
        assert len(obs["alerts"]) == 15

    def test_reset_hard_has_thirty_alerts(self, client):
        resp = client.post("/reset", json={"task_id": "hard", "seed": 42})
        obs = resp.json()["observation"]
        assert len(obs["alerts"]) == 30

    def test_reset_observation_has_service_map(self, client):
        obs = _reset_easy(client)["observation"]
        assert "service_map" in obs
        assert isinstance(obs["service_map"], dict)
        assert len(obs["service_map"]) > 0

    def test_reset_observation_has_pending_count(self, client):
        obs = _reset_easy(client)["observation"]
        assert "pending_count" in obs
        assert obs["pending_count"] == 5   # easy: 5 alerts

    def test_reset_observation_has_step_number_zero(self, client):
        obs = _reset_easy(client)["observation"]
        assert obs.get("step_number") == 0

    def test_reset_observation_has_max_steps(self, client):
        obs = _reset_easy(client)["observation"]
        assert "max_steps" in obs
        assert obs["max_steps"] == 10      # easy task max_steps

    def test_reset_observation_has_feedback_empty(self, client):
        obs = _reset_easy(client)["observation"]
        # Feedback should be empty string on fresh reset
        assert obs.get("feedback", "") == ""

    def test_reset_alerts_have_required_fields(self, client):
        obs = _reset_easy(client)["observation"]
        required = {"alert_id", "timestamp", "service", "metric",
                    "metric_value", "threshold", "message"}
        for alert in obs["alerts"]:
            assert required.issubset(alert.keys()), (
                f"Alert missing fields: {required - alert.keys()}"
            )

    def test_reset_alerts_not_triaged(self, client):
        obs = _reset_easy(client)["observation"]
        for alert in obs["alerts"]:
            assert alert["triaged"] is False

    def test_reset_deterministic_same_seed(self, client):
        """Same task + seed → identical alert IDs."""
        obs1 = _reset_easy(client, 42)["observation"]
        obs2 = _reset_easy(client, 42)["observation"]
        ids1 = [a["alert_id"] for a in obs1["alerts"]]
        ids2 = [a["alert_id"] for a in obs2["alerts"]]
        assert ids1 == ids2

    def test_reset_different_seeds_produce_different_scenarios(self, client):
        obs1 = _reset_easy(client, 42)["observation"]
        obs2 = _reset_easy(client, 99)["observation"]
        msgs1 = [a["message"] for a in obs1["alerts"]]
        msgs2 = [a["message"] for a in obs2["alerts"]]
        assert msgs1 != msgs2

    def test_reset_invalid_task_id_422(self, client):
        resp = client.post("/reset", json={"task_id": "nonexistent", "seed": 42})
        assert resp.status_code == 422

    def test_reset_missing_body_uses_defaults(self, client):
        """Empty body should use default task_id=easy and seed=42."""
        resp = client.post("/reset", json={})
        assert resp.status_code == 200

    def test_reset_clears_previous_episode(self, client):
        """A second /reset must return step_number=0 and full pending_count."""
        # Do a step in the first episode
        aid = _first_alert_id(client, 42)
        client.post("/step", json={
            "action_type": "triage",
            "alert_id": aid,
            "root_cause": "resource_exhaustion",
            "severity": "high",
            "remediation": "scale_up",
        })
        # Reset again — should be back to fresh state
        obs = _reset_easy(client, 42)["observation"]
        assert obs["step_number"] == 0
        assert obs["pending_count"] == 5


# ---------------------------------------------------------------------------
# TestStep
# ---------------------------------------------------------------------------

class TestStep:

    def test_step_triage_200(self, client):
        aid = _first_alert_id(client)
        resp = client.post("/step", json={
            "action_type": "triage",
            "alert_id": aid,
            "root_cause": "resource_exhaustion",
            "severity": "high",
            "remediation": "scale_up",
        })
        assert resp.status_code == 200

    def test_step_response_has_required_fields(self, client):
        aid = _first_alert_id(client)
        data = client.post("/step", json={
            "action_type": "triage",
            "alert_id": aid,
            "root_cause": "resource_exhaustion",
            "severity": "high",
            "remediation": "scale_up",
        }).json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_reward_is_numeric(self, client):
        aid = _first_alert_id(client)
        data = client.post("/step", json={
            "action_type": "triage",
            "alert_id": aid,
            "root_cause": "deployment_bug",
            "severity": "critical",
            "remediation": "rollback_deploy",
        }).json()
        assert isinstance(data["reward"], (int, float))

    def test_step_done_false_after_first_of_five(self, client):
        aid = _first_alert_id(client)
        data = client.post("/step", json={
            "action_type": "triage",
            "alert_id": aid,
            "root_cause": "resource_exhaustion",
            "severity": "high",
            "remediation": "scale_up",
        }).json()
        assert data["done"] is False

    def test_step_observation_pending_count_decrements(self, client):
        _reset_easy(client)
        aids = _all_alert_ids(client)   # also resets
        data = client.post("/step", json={
            "action_type": "triage",
            "alert_id": aids[0],
            "root_cause": "resource_exhaustion",
            "severity": "high",
            "remediation": "scale_up",
        }).json()
        obs = data.get("observation", {})
        assert obs["pending_count"] == 4   # 5 - 1

    def test_step_triage_marks_alert_as_triaged(self, client):
        _reset_easy(client)
        aids = _all_alert_ids(client)
        data = client.post("/step", json={
            "action_type": "triage",
            "alert_id": aids[0],
            "root_cause": "resource_exhaustion",
            "severity": "high",
            "remediation": "scale_up",
        }).json()
        obs = data["observation"]
        triaged_alert = next(a for a in obs["alerts"] if a["alert_id"] == aids[0])
        assert triaged_alert["triaged"] is True

    def test_step_skip_action_200(self, client):
        aid = _first_alert_id(client)
        resp = client.post("/step", json={
            "action_type": "skip",
            "alert_id": aid,
        })
        assert resp.status_code == 200

    def test_step_skip_also_decrements_pending(self, client):
        _reset_easy(client)
        aids = _all_alert_ids(client)
        data = client.post("/step", json={
            "action_type": "skip",
            "alert_id": aids[0],
        }).json()
        assert data["observation"]["pending_count"] == 4

    def test_step_link_alerts_200(self, client):
        _reset_easy(client)
        aids = _all_alert_ids(client)
        resp = client.post("/step", json={
            "action_type": "link_alerts",
            "alert_ids": [aids[0], aids[1]],
            "incident_label": "test-incident",
        })
        assert resp.status_code == 200

    def test_step_link_does_not_decrement_pending(self, client):
        """link_alerts is grouping-only; pending_count must not change."""
        _reset_easy(client)
        aids = _all_alert_ids(client)
        data = client.post("/step", json={
            "action_type": "link_alerts",
            "alert_ids": [aids[0], aids[1]],
            "incident_label": "test-incident",
        }).json()
        assert data["observation"]["pending_count"] == 5

    def test_step_invalid_action_missing_alert_id_422(self, client):
        _reset_easy(client)
        resp = client.post("/step", json={
            "action_type": "triage",
            # alert_id intentionally missing
            "root_cause": "resource_exhaustion",
            "severity": "high",
            "remediation": "scale_up",
        })
        assert resp.status_code == 422

    def test_step_invalid_root_cause_422(self, client):
        aid = _first_alert_id(client)
        resp = client.post("/step", json={
            "action_type": "triage",
            "alert_id": aid,
            "root_cause": "not_a_valid_cause",
            "severity": "high",
            "remediation": "scale_up",
        })
        assert resp.status_code == 422

    def test_step_invalid_severity_422(self, client):
        aid = _first_alert_id(client)
        resp = client.post("/step", json={
            "action_type": "triage",
            "alert_id": aid,
            "root_cause": "resource_exhaustion",
            "severity": "extreme",       # not a valid level
            "remediation": "scale_up",
        })
        assert resp.status_code == 422

    def test_step_feedback_non_empty_after_triage(self, client):
        aid = _first_alert_id(client)
        data = client.post("/step", json={
            "action_type": "triage",
            "alert_id": aid,
            "root_cause": "resource_exhaustion",
            "severity": "high",
            "remediation": "scale_up",
        }).json()
        feedback = data.get("observation", {}).get("feedback", "")
        assert isinstance(feedback, str)
        assert len(feedback) > 0

    def test_step_step_number_increments(self, client):
        _reset_easy(client)
        aids = _all_alert_ids(client)
        data = client.post("/step", json={
            "action_type": "triage",
            "alert_id": aids[0],
            "root_cause": "resource_exhaustion",
            "severity": "high",
            "remediation": "scale_up",
        }).json()
        assert data["observation"]["step_number"] == 1


# ---------------------------------------------------------------------------
# TestStepBeforeReset
# ---------------------------------------------------------------------------

class TestStepBeforeReset:

    def test_step_before_reset_returns_400(self):
        """
        A fresh app with no prior /reset should return 400 on /step.
        We create an isolated TestClient with a fresh app instance to ensure
        no prior episode exists.
        """
        from fastapi.testclient import TestClient as TC
        from server.environment import AlertTriageEnv
        from server import app as app_module
        import server.app as app_mod

        # Create a temporary clean env to simulate "no episode" state
        original_env = app_mod.env
        app_mod.env = AlertTriageEnv()   # fresh instance — _active=False
        try:
            with TC(app_mod.app) as c:
                resp = c.post("/step", json={
                    "action_type": "triage",
                    "alert_id": "alert-001",
                    "root_cause": "resource_exhaustion",
                    "severity": "high",
                    "remediation": "scale_up",
                })
                assert resp.status_code == 400
        finally:
            app_mod.env = original_env   # restore


# ---------------------------------------------------------------------------
# TestFullEpisode
# ---------------------------------------------------------------------------

class TestFullEpisode:

    def test_easy_episode_completes(self, client):
        """Triaging all 5 easy alerts sets done=True on the last step."""
        resp = client.post("/reset", json={"task_id": "easy", "seed": 42})
        obs = resp.json()["observation"]
        alerts = obs["alerts"]

        done = False
        data: dict = {}
        for alert in alerts:
            data = client.post("/step", json={
                "action_type": "triage",
                "alert_id": alert["alert_id"],
                "root_cause": "resource_exhaustion",
                "severity": "medium",
                "remediation": "acknowledge_and_monitor",
            }).json()
            done = data.get("done", False)

        assert done is True

    def test_episode_done_info_has_grader_score(self, client):
        """When done=True, info must contain grader_score in [0, 1]."""
        resp = client.post("/reset", json={"task_id": "easy", "seed": 42})
        alerts = resp.json()["observation"]["alerts"]

        data: dict = {}
        for alert in alerts:
            data = client.post("/step", json={
                "action_type": "triage",
                "alert_id": alert["alert_id"],
                "root_cause": "resource_exhaustion",
                "severity": "medium",
                "remediation": "acknowledge_and_monitor",
            }).json()

        assert "grader_score" in data.get("info", {})
        score = data["info"]["grader_score"]
        assert 0.0 <= score <= 1.0

    def test_step_after_done_returns_done_true(self, client):
        """Steps after episode end return done=True and reward=0."""
        resp = client.post("/reset", json={"task_id": "easy", "seed": 42})
        alerts = resp.json()["observation"]["alerts"]
        for alert in alerts:
            client.post("/step", json={
                "action_type": "triage",
                "alert_id": alert["alert_id"],
                "root_cause": "resource_exhaustion",
                "severity": "medium",
                "remediation": "acknowledge_and_monitor",
            })
        # One more step after episode is done
        data = client.post("/step", json={
            "action_type": "skip",
            "alert_id": alerts[0]["alert_id"],
        }).json()
        assert data["done"] is True
        assert data["reward"] == 0.0

    def test_budget_exhaustion_sets_done(self, client):
        """Exceeding max_steps (10 for easy) triggers done without triaging all alerts."""
        resp = client.post("/reset", json={"task_id": "easy", "seed": 42})
        alerts = resp.json()["observation"]["alerts"]
        # Use 10 link_alerts steps (they don't reduce pending_count)
        # to exhaust the budget without triaging anything
        for _ in range(10):
            data = client.post("/step", json={
                "action_type": "link_alerts",
                "alert_ids": [alerts[0]["alert_id"], alerts[1]["alert_id"]],
                "incident_label": "x",
            }).json()
        assert data["done"] is True

    def test_perfect_easy_episode_reward_sum(self, client):
        """
        Perfect easy run: all root_cause+severity+remediation correct.
        Each triage gives +0.80 → total >= 4.0 for 5 alerts
        (budget pressure may apply on last steps; total should still be > 0).
        """
        from server.scenario_generator import generate_scenario
        scenario = generate_scenario("easy", 42)
        gt_map = {g["alert_id"]: g for g in scenario["ground_truth"]}

        resp = client.post("/reset", json={"task_id": "easy", "seed": 42})
        alerts = resp.json()["observation"]["alerts"]

        total_reward = 0.0
        for alert in alerts:
            gt = gt_map[alert["alert_id"]]
            data = client.post("/step", json={
                "action_type": "triage",
                "alert_id":    alert["alert_id"],
                "root_cause":  gt["true_root_cause"],
                "severity":    gt["true_severity"],
                "remediation": gt["true_remediation"],
            }).json()
            total_reward += data["reward"]

        # 5 perfect triages × 0.80 = 4.00; budget pressure on last steps
        # may reduce slightly, but total should be well above 0.
        assert total_reward > 2.0


# ---------------------------------------------------------------------------
# TestState
# ---------------------------------------------------------------------------

class TestState:

    def test_state_200_after_reset(self, client):
        _reset_easy(client)
        resp = client.get("/state")
        assert resp.status_code == 200

    def test_state_contains_ground_truth(self, client):
        _reset_easy(client)
        data = client.get("/state").json()
        assert "ground_truth" in data
        assert isinstance(data["ground_truth"], list)
        assert len(data["ground_truth"]) == 5

    def test_state_contains_task_id(self, client):
        client.post("/reset", json={"task_id": "medium", "seed": 7})
        data = client.get("/state").json()
        assert data.get("task_id") == "medium"

    def test_state_contains_seed(self, client):
        client.post("/reset", json={"task_id": "easy", "seed": 123})
        data = client.get("/state").json()
        assert data.get("seed") == 123

    def test_state_step_number_updates(self, client):
        _reset_easy(client)
        aids = _all_alert_ids(client)
        client.post("/step", json={
            "action_type": "triage",
            "alert_id": aids[0],
            "root_cause": "resource_exhaustion",
            "severity": "high",
            "remediation": "scale_up",
        })
        data = client.get("/state").json()
        assert data["step_number"] == 1

    def test_state_incidents_present(self, client):
        _reset_easy(client)
        data = client.get("/state").json()
        assert "incidents" in data
        # Easy task has 0 incidents
        assert data["incidents"] == []

    def test_state_ground_truth_has_correct_fields(self, client):
        _reset_easy(client)
        gt = client.get("/state").json()["ground_truth"]
        required = {"alert_id", "true_root_cause", "true_severity",
                    "true_remediation", "incident_id"}
        for entry in gt:
            assert required.issubset(entry.keys()), (
                f"GT entry missing fields: {required - entry.keys()}"
            )

    def test_state_before_reset_returns_400(self):
        """GET /state before any /reset must return 400."""
        import server.app as app_mod
        from fastapi.testclient import TestClient as TC
        from server.environment import AlertTriageEnv
        original_env = app_mod.env
        app_mod.env = AlertTriageEnv()   # fresh — _active=False
        try:
            with TC(app_mod.app) as c:
                resp = c.get("/state")
                assert resp.status_code == 400
        finally:
            app_mod.env = original_env

    def test_state_done_false_before_episode_ends(self, client):
        _reset_easy(client)
        data = client.get("/state").json()
        assert data["done"] is False

    def test_state_grader_score_none_during_episode(self, client):
        _reset_easy(client)
        data = client.get("/state").json()
        assert data.get("grader_score") is None
