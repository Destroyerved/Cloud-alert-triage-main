#!/usr/bin/env python3
"""
scripts/smoke_test.py
---------------------
End-to-end smoke test for the cloud-alert-triage environment.
Requires the server to be running on localhost:7860 (or ENV_URL).

Usage:
    # Terminal 1: start server
    uvicorn server.app:app --port 7860

    # Terminal 2: run smoke test
    python scripts/smoke_test.py

Exit code 0 = all assertions passed.
Exit code 1 = one or more assertions failed.
"""

import json
import os
import sys

try:
    import httpx
except ImportError:
    print("[ERROR] httpx not installed. Run: pip install httpx")
    sys.exit(1)

BASE_URL = os.environ.get("ENV_URL", "http://localhost:7860")

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {label}")
        PASS += 1
    else:
        print(f"  [FAIL] {label}{(' — ' + detail) if detail else ''}")
        FAIL += 1


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> int:
    print(f"\nCloud Alert Triage — Smoke Test")
    print(f"Target: {BASE_URL}")

    client = httpx.Client(base_url=BASE_URL, timeout=15.0)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    section("Health Check")
    try:
        r = client.get("/health")
        check("GET /health returns 200", r.status_code == 200)
        data = r.json()
        check("Body contains status=ok", data.get("status") == "ok", str(data))
    except Exception as e:
        check("GET /health reachable", False, str(e))
        print("\n[ABORT] Server unreachable. Is it running?")
        return 1

    # ------------------------------------------------------------------
    # Reset — easy
    # ------------------------------------------------------------------
    section("Reset (easy, seed=42)")
    r = client.post("/reset", json={"task_id": "easy", "seed": 42})
    check("POST /reset returns 200", r.status_code == 200, f"Got {r.status_code}")
    reset_data = r.json()

    # Unwrap observation whether it's nested or flat
    obs = reset_data.get("observation", reset_data)
    check("Response has alerts list", isinstance(obs.get("alerts"), list))
    alerts = obs.get("alerts", [])
    check("Easy scenario has 5 alerts", len(alerts) == 5, f"Got {len(alerts)}")
    check("pending_count is 5", obs.get("pending_count") == 5)
    check("step_number is 0", obs.get("step_number") == 0)
    check("service_map is dict", isinstance(obs.get("service_map"), dict))

    # ------------------------------------------------------------------
    # Step — triage all 5 alerts
    # ------------------------------------------------------------------
    section("Step through all easy alerts")
    cumulative_reward = 0.0
    done = False
    final_data = {}

    for i, alert in enumerate(alerts):
        action = {
            "action_type": "triage",
            "alert_id": alert["alert_id"],
            "root_cause": "resource_exhaustion",
            "severity": "medium",
            "remediation": "acknowledge_and_monitor",
        }
        r = client.post("/step", json=action)
        check(
            f"Step {i+1}: POST /step returns 200",
            r.status_code == 200,
            f"Got {r.status_code}",
        )
        step_data = r.json()
        reward = step_data.get("reward", None)
        check(
            f"Step {i+1}: reward is numeric",
            isinstance(reward, (int, float)),
            f"reward={reward}",
        )
        if isinstance(reward, (int, float)):
            cumulative_reward += reward
        done = step_data.get("done", False)
        final_data = step_data

    check("done=True after last step", done is True, f"done={done}")
    check(
        "info contains grader_score",
        "grader_score" in final_data.get("info", {}),
        str(final_data.get("info")),
    )

    grader_score = final_data.get("info", {}).get("grader_score")
    if grader_score is not None:
        check(
            "grader_score in [0.0, 1.0]",
            0.0 <= grader_score <= 1.0,
            f"score={grader_score}",
        )

    check("cumulative_reward > 0", cumulative_reward > 0, f"total={cumulative_reward:.3f}")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    section("State Endpoint")
    r = client.get("/state")
    check("GET /state returns 200", r.status_code == 200, f"Got {r.status_code}")
    state_data = r.json()
    check("state has ground_truth", "ground_truth" in state_data)
    check("state has task_id", state_data.get("task_id") == "easy")
    check("state.done is True", state_data.get("done") is True)

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------
    section("Determinism (same seed => same alerts)")
    r1 = client.post("/reset", json={"task_id": "easy", "seed": 42})
    r2 = client.post("/reset", json={"task_id": "easy", "seed": 42})
    obs1 = r1.json().get("observation", r1.json())
    obs2 = r2.json().get("observation", r2.json())
    ids1 = [a["alert_id"] for a in obs1.get("alerts", [])]
    ids2 = [a["alert_id"] for a in obs2.get("alerts", [])]
    check("Same seed => same alert IDs", ids1 == ids2, f"{ids1} vs {ids2}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n=== Smoke Test Summary ===")
    print(f"  Passed : {PASS}")
    print(f"  Failed : {FAIL}")
    if FAIL == 0:
        print("\nAll smoke tests passed.\n")
        return 0
    else:
        print(f"\n{FAIL} test(s) failed. Fix before submitting.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
