"""
server/environment.py

Core AlertTriageEnv class for the Cloud Alert Triage environment.
Implemented in Phase 4.  Dynamic cascade mechanic added in Phase B.

Public API
----------
    env    = AlertTriageEnv()
    obs    = env.reset(task_id, seed)    -> Observation
    result = env.step(action_or_dict)    -> StepResult
    state  = env.state()                 -> EnvironmentState

Episode lifecycle
-----------------
1. reset(task_id, seed)  — initialise a new episode; clears any prior state
2. step(action) × N      — agent acts until done=True or budget exhausted
3. state()               — inspect full hidden state (for /state endpoint)

Dynamic cascade mechanic (Phase B)
-----------------------------------
After CASCADE_TRIGGER_STEP environment steps, any *original* critical/high
alert that is still untriaged spawns exactly one new dependent alert on a
downstream service chosen deterministically from the service graph.

- Each originating alert spawns at most one dynamic alert (tracked via
  ``_spawned_from``).
- Dynamic alerts participate in per-step rewards (they behave like normal
  alerts for the reward function).
- Dynamic alerts are **excluded** from the final grader score via the
  ``dynamic_alert_ids`` set passed through the state snapshot.
- Alert IDs for dynamic alerts use the prefix ``dyn-``.

Design notes
------------
- Single-instance design (one global env shared by the API server).
- Not thread-safe — acceptable for a single-worker hackathon deployment.
- Rewards are delegated to server/rewards.py (Phase 5 stub → real impl).
- Grading is delegated to server/grading.py (Phase 6 stub → real impl).
- Edge cases handled directly in the environment layer (not in rewards):
    • step() before reset()   → RuntimeError
    • invalid alert_id        → −0.10, feedback note, step still counts
    • already-triaged alert   → −0.15, feedback note, step still counts
    • step() after done       → reward=0.0, done=True, no state mutation
    • invalid action format   → −0.10 (Pydantic validation failure)
"""

from __future__ import annotations

from typing import Any

from server.config import (
    CASCADE_ELIGIBLE_SEVERITIES,
    CASCADE_TRIGGER_STEP,
    MAX_STEPS_BY_TASK,
    SEVERITY_ORDER,
)
from server.grading import grade_episode
from server.models import Action, Alert, EnvironmentState, Observation, StepResult
from server.rewards import compute_reward
from server.scenario_generator import generate_scenario
from server.service_graph import get_dependents, get_graph_as_adjacency_list


class AlertTriageEnv:
    """
    Stateful environment for the Cloud Alert Triage task.

    Attributes (all private; access via public methods only)
    ---------
    _active          : bool        — True after the first reset()
    _task_id         : str         — current task ("easy" / "medium" / "hard")
    _seed            : int         — seed used for scenario generation
    _alerts          : list[Alert] — mutable alert objects (triaged flag updated in-place)
    _ground_truth    : list[dict]  — one GT entry per alert (never mutated at init;
                                     dynamic alerts append new entries at runtime)
    _incidents       : list[dict]  — true incident groupings (never mutated)
    _agent_decisions : list[dict]  — ordered log of every recorded decision
    _agent_links     : list[dict]  — subset of decisions: link_alerts only
    _step_count      : int         — number of steps taken so far
    _max_steps       : int         — step budget for current task
    _done            : bool        — True once episode ends
    _cumulative_reward: float      — running sum of per-step rewards
    _grader_score    : float|None  — set when done=True
    _service_map     : dict        — adjacency list (static; same every episode)
    _dynamic_alert_ids : set[str]  — IDs of alerts spawned by the cascade mechanic
    _spawned_from      : set[str]  — original alert IDs that have already spawned
    _original_alert_ids: set[str]  — IDs of alerts from the initial scenario
    _cascade_feedback  : str       — feedback text about newly spawned alerts
    """

    def __init__(self) -> None:
        self._active: bool = False
        self._task_id: str = ""
        self._seed: int = 0
        self._alerts: list[Alert] = []
        self._ground_truth: list[dict[str, Any]] = []
        self._incidents: list[dict[str, Any]] = []
        self._agent_decisions: list[dict[str, Any]] = []
        self._agent_links: list[dict[str, Any]] = []
        self._step_count: int = 0
        self._max_steps: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._grader_score: float | None = None
        # Dynamic cascade state
        self._dynamic_alert_ids: set[str] = set()
        self._spawned_from: set[str] = set()
        self._original_alert_ids: set[str] = set()
        self._cascade_feedback: str = ""
        # Static; build once at construction time.
        self._service_map: dict[str, list[str]] = get_graph_as_adjacency_list()

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, task_id: str, seed: int) -> Observation:
        """
        Start a new episode, discarding any in-progress episode.

        Parameters
        ----------
        task_id : "easy" | "medium" | "hard"
        seed    : integer RNG seed — same task_id + seed → identical scenario

        Returns
        -------
        Observation
            step_number=0, pending_count=total alerts, feedback=""

        Raises
        ------
        ValueError
            If task_id is not one of the recognised task names.
        """
        if task_id not in MAX_STEPS_BY_TASK:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid values: {sorted(MAX_STEPS_BY_TASK.keys())}"
            )

        scenario = generate_scenario(task_id, seed)

        self._task_id = task_id
        self._seed = seed
        # Construct mutable Alert objects from the generator's dicts.
        self._alerts = [Alert(**a) for a in scenario["alerts"]]
        # Ground truth and incidents are stored as plain dicts (never shown to agent).
        self._ground_truth = [dict(g) for g in scenario["ground_truth"]]
        self._incidents = [dict(i) for i in scenario["incidents"]]
        self._agent_decisions = []
        self._agent_links = []
        self._step_count = 0
        self._max_steps = MAX_STEPS_BY_TASK[task_id]
        self._done = False
        self._cumulative_reward = 0.0
        self._grader_score = None
        # Dynamic cascade: track which alerts are original vs spawned
        self._original_alert_ids = {a.alert_id for a in self._alerts}
        self._dynamic_alert_ids = set()
        self._spawned_from = set()
        
        # Partial observability: mask metric values for some alerts
        from server.config import PARTIAL_OBSERVABILITY_ENABLED, PARTIAL_OBSERVABILITY_TASKS
        if PARTIAL_OBSERVABILITY_ENABLED and task_id in PARTIAL_OBSERVABILITY_TASKS:
            # Mask 50% of alerts to require investigation
            import random as rng_module
            rng = rng_module.Random(seed + 1000)  # Different seed for masking
            alert_list = list(self._alerts)
            rng.shuffle(alert_list)
            # Mask first half of shuffled alerts
            for alert in alert_list[:len(alert_list)//2]:
                alert.masked = True
        
        self._cascade_feedback = ""
        self._active = True

        return self._build_observation(feedback="")

    def step(self, action: Action | dict[str, Any]) -> StepResult:
        """
        Apply one action and advance the episode.

        Parameters
        ----------
        action : Action model instance **or** a plain dict that can be coerced
                 into one via Pydantic.  The dict path is the primary path used
                 by the HTTP API and tests.

        Returns
        -------
        StepResult
            (observation, reward, done, info)
            When done=True, info contains {"grader_score": float}.

        Raises
        ------
        RuntimeError
            If called before any reset().
        """
        if not self._active:
            raise RuntimeError(
                "No active episode. Call reset(task_id, seed) before step()."
            )

        # ── already done — no state change, return grader_score as reward ──────
        if self._done:
            info = self._make_info()
            safe_score = float(info.get("grader_score", 0.5))
            return StepResult(
                observation=self._build_observation(
                    feedback="Episode already complete."
                ),
                reward=safe_score,
                done=True,
                info=info,
            )

        # ── coerce dict → Action (Pydantic validates enum values & required fields)
        if isinstance(action, dict):
            try:
                action = Action(**action)
            except Exception as exc:
                # Invalid action format: penalise, count the step, check done.
                return self._record_invalid_action(str(exc))

        # ── dispatch valid action ─────────────────────────────────────────────
        reward, feedback = self._dispatch(action)

        self._step_count += 1
        self._cumulative_reward += reward

        # ── dynamic cascade: spawn dependent alerts if threshold reached ─────
        self._maybe_spawn_cascade_alerts()
        if self._cascade_feedback:
            feedback = f"{feedback} {self._cascade_feedback}"
            self._cascade_feedback = ""

        self._update_done()

        # When done=True, override the per-step reward with grader_score.
        # Evaluators read the reward field of the done=True step as the task
        # score — it must be strictly in (0, 1).
        info = self._make_info()
        if self._done:
            reward = float(info.get("grader_score", 0.5))

        return StepResult(
            observation=self._build_observation(feedback=feedback),
            reward=reward,
            done=self._done,
            info=info,
        )

    def state(self) -> EnvironmentState:
        """
        Return the full internal state, including hidden ground truth.

        Intended for the GET /state debug endpoint.  inference.py must NOT
        call this — it would give the agent access to ground truth.
        """
        return EnvironmentState(
            task_id=self._task_id,
            seed=self._seed,
            step_number=self._step_count,
            max_steps=self._max_steps,
            done=self._done,
            alerts=list(self._alerts),
            ground_truth=list(self._ground_truth),
            agent_decisions=list(self._agent_decisions),
            incidents=list(self._incidents),
            cumulative_reward=self._cumulative_reward,
            grader_score=self._grader_score,
            dynamic_alert_ids=set(self._dynamic_alert_ids),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Action dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def _dispatch(self, action: Action) -> tuple[float, str]:
        if action.action_type == "triage":
            return self._apply_triage(action)
        elif action.action_type == "link_alerts":
            return self._apply_link(action)
        elif action.action_type == "investigate":
            return self._apply_investigate(action)
        else:  # "skip" — guaranteed by Action validator
            return self._apply_skip(action)

    def _apply_investigate(self, action: Action) -> tuple[float, str]:
        """Reveal masked details for an alert (partial observability)."""
        alert = self._find_alert(action.alert_id)
        
        if alert is None:
            return (
                -0.10,
                f"Unknown alert_id '{action.alert_id}'. No investigation recorded.",
            )
        
        if alert.investigated:
            return (
                -0.05,
                f"Alert '{action.alert_id}' already investigated. Skipping.",
            )
        
        # Mark as investigated and unmask
        alert.investigated = True
        was_masked = alert.masked
        if alert.masked:
            alert.masked = False
        
        # Only reward if the alert was actually masked (partial observability reveal)
        # Investigating a non-masked alert gives a tiny epsilon — never exactly 0.0,
        # which could be misread as an out-of-range score by the evaluator.
        reward = 0.10 if was_masked else 1e-4
        msg = (
            f"Investigated {action.alert_id}: metric value revealed"
            if was_masked
            else f"Alert {action.alert_id} was already visible — no new information."
        )
        return reward, msg

    def _apply_triage(self, action: Action) -> tuple[float, str]:
        alert = self._find_alert(action.alert_id)

        if alert is None:
            return (
                -0.10,
                f"Unknown alert_id '{action.alert_id}'. No triage recorded.",
            )

        if alert.triaged:
            return (
                -0.15,
                f"Alert '{action.alert_id}' is already triaged. Penalty applied.",
            )

        # Record the decision on both the Alert object and the decisions log.
        decision: dict[str, Any] = {
            "alert_id":    action.alert_id,
            "action_type": "triage",
            "root_cause":  action.root_cause,
            "severity":    action.severity,
            "remediation": action.remediation,
        }
        alert.triaged = True
        alert.agent_decision = decision
        self._agent_decisions.append(decision)

        # compute_reward now includes budget pressure via _penalty_budget().
        # Do NOT add self._budget_penalty() here — that would double-count it.
        reward = compute_reward(decision, self._ground_truth, self._make_state_snapshot())

        return reward, self._triage_feedback(action)

    def _apply_link(self, action: Action) -> tuple[float, str]:
        # Validate all referenced alert IDs.
        for aid in action.alert_ids:
            if self._find_alert(aid) is None:
                return (
                    -0.10,
                    f"Unknown alert_id '{aid}' in link_alerts. No link recorded.",
                )

        link: dict[str, Any] = {
            "action_type":    "link_alerts",
            "alert_ids":      list(action.alert_ids),
            "incident_label": action.incident_label,
        }
        self._agent_links.append(link)
        self._agent_decisions.append(link)

        # link_alerts does NOT mark alerts as triaged — they must be triaged
        # separately.  It only records the grouping for scoring purposes.
        reward = compute_reward(link, self._ground_truth, self._make_state_snapshot())

        return (
            reward,
            f"Linked {len(action.alert_ids)} alerts as incident "
            f"'{action.incident_label}'.",
        )

    def _apply_skip(self, action: Action) -> tuple[float, str]:
        alert = self._find_alert(action.alert_id)

        if alert is None:
            return (
                -0.10,
                f"Unknown alert_id '{action.alert_id}'. No skip recorded.",
            )

        if alert.triaged:
            return (
                -0.15,
                f"Alert '{action.alert_id}' is already triaged. Penalty applied.",
            )

        decision: dict[str, Any] = {
            "alert_id":    action.alert_id,
            "action_type": "skip",
        }
        alert.triaged = True
        alert.agent_decision = decision
        self._agent_decisions.append(decision)

        reward = compute_reward(decision, self._ground_truth, self._make_state_snapshot())

        return reward, f"Skipped alert '{action.alert_id}'."

    # ─────────────────────────────────────────────────────────────────────────
    # Done detection and scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _update_done(self) -> None:
        """
        Mark done when all alerts are triaged (or skipped) OR step budget
        is exhausted.  Always sets _grader_score strictly in (EPS, 1-EPS).
        """
        all_triaged = all(a.triaged for a in self._alerts)
        budget_gone = self._step_count >= self._max_steps

        if (all_triaged or budget_gone) and not self._done:
            self._done = True
            try:
                raw = grade_episode(self._task_id, self._make_state_snapshot())
            except Exception:
                raw = 0.5
            # Triple-enforce: must be strictly inside (0, 1)
            import math as _math
            if not _math.isfinite(raw):
                raw = 0.5
            self._grader_score = max(1e-4, min(1 - 1e-4, float(raw)))

    def _make_info(self) -> dict[str, Any]:
        """Return info dict with grader_score whenever done=True.

        grader_score is STRICTLY in (1e-4, 1-1e-4) — never 0.0, never 1.0.
        No round() is applied after clamping: round() can push a float that is
        only epsilon below 1.0 up to exactly 1.0 via IEEE-754 rounding, which
        would fail the open-interval requirement.
        """
        if self._done:
            score = self._grader_score if self._grader_score is not None else 0.5
            import math as _math
            if not _math.isfinite(score):
                score = 0.5
            # Clamp to strict open interval — NO round() after this point.
            score = max(1e-4, min(1 - 1e-4, float(score)))
            return {"grader_score": score}
        return {}

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _find_alert(self, alert_id: str | None) -> Alert | None:
        if alert_id is None:
            return None
        for a in self._alerts:
            if a.alert_id == alert_id:
                return a
        return None

    def _get_ground_truth(self, alert_id: str) -> dict[str, Any] | None:
        for gt in self._ground_truth:
            if gt["alert_id"] == alert_id:
                return gt
        return None

    def _pending_count(self) -> int:
        return sum(1 for a in self._alerts if not a.triaged)

    def _budget_penalty(self) -> float:
        """
        −0.05 per step once ≥80 % of the step budget has been consumed.
        Applied on top of the base action reward.
        """
        if self._max_steps > 0 and self._step_count >= 0.8 * self._max_steps:
            return -0.05
        return 0.0

    def _triage_feedback(self, action: Action) -> str:
        """
        Short hint based on comparison with ground truth.
        Gives learning signal without revealing exact answers.
        """
        gt = self._get_ground_truth(action.alert_id)
        if gt is None:
            return "Alert triaged."

        parts: list[str] = []

        # Root cause hint
        if action.root_cause == gt["true_root_cause"]:
            parts.append("Root cause accepted.")
        else:
            parts.append("Root cause may be incorrect — review the dependency graph.")

        # Severity hint
        if action.severity == gt["true_severity"]:
            parts.append("Severity accepted.")
        else:
            true_rank = SEVERITY_ORDER.get(gt["true_severity"], 2)
            agent_rank = SEVERITY_ORDER.get(action.severity or "", 2)
            if abs(true_rank - agent_rank) == 1:
                parts.append("Severity is close but off by one level.")
            else:
                parts.append("Severity appears significantly off.")

        return " ".join(parts)

    def _record_invalid_action(self, error_msg: str) -> StepResult:
        """Handle Pydantic validation failure: −0.10 penalty, step still counts."""
        penalty = -0.10
        self._step_count += 1
        self._cumulative_reward += penalty
        self._update_done()
        info = self._make_info()
        # CRITICAL: if the episode just ended, the reward field MUST be the
        # grader_score (strictly in (0, 1)), not the raw penalty (-0.10).
        # The evaluator reads the reward field of the terminal step as the
        # task score and rejects values outside (0, 1).
        reward = float(info.get("grader_score", 0.5)) if self._done else penalty
        return StepResult(
            observation=self._build_observation(
                feedback=f"Invalid action format: {error_msg}"
            ),
            reward=reward,
            done=self._done,
            info=info,
        )

    def _build_observation(self, feedback: str) -> Observation:
        # Create alerts list, masking metric values for uninvestigated alerts if partial observability enabled
        from server.config import PARTIAL_OBSERVABILITY_ENABLED, PARTIAL_OBSERVABILITY_TASKS
        
        alerts_to_show = []
        for alert in self._alerts:
            if (PARTIAL_OBSERVABILITY_ENABLED and 
                self._task_id in PARTIAL_OBSERVABILITY_TASKS and
                alert.masked and not alert.investigated and not alert.triaged):
                # Create a masked version
                masked_alert = Alert(
                    alert_id=alert.alert_id,
                    timestamp=alert.timestamp,
                    service=alert.service,
                    metric=alert.metric,
                    metric_value=None,  # Hidden!
                    threshold=alert.threshold,
                    message=alert.message,
                    context=alert.context,
                    triaged=alert.triaged,
                    investigated=alert.investigated,
                    agent_decision=alert.agent_decision,
                    masked=True,
                )
                alerts_to_show.append(masked_alert)
            else:
                alerts_to_show.append(alert)
        
        return Observation(
            alerts=alerts_to_show,
            service_map=self._service_map,
            pending_count=self._pending_count(),
            step_number=self._step_count,
            max_steps=self._max_steps,
            feedback=feedback,
        )

    def _make_state_snapshot(self) -> dict[str, Any]:
        """
        Lightweight dict snapshot passed to rewards.compute_reward() and
        grading.grade_episode().  Contains everything Phase 5/6 will need.

        The ``dynamic_alert_ids`` set lets the grader exclude dynamically
        spawned alerts from the final episode score.
        """
        return {
            "task_id":           self._task_id,
            "seed":              self._seed,
            "step_number":       self._step_count,
            "max_steps":         self._max_steps,
            "done":              self._done,
            "ground_truth":      self._ground_truth,
            "incidents":         self._incidents,
            "agent_links":       self._agent_links,
            "agent_decisions":   self._agent_decisions,
            "cumulative_reward": self._cumulative_reward,
            "dynamic_alert_ids": self._dynamic_alert_ids,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Dynamic cascade mechanic
    # ─────────────────────────────────────────────────────────────────────────

    def _maybe_spawn_cascade_alerts(self) -> None:
        """
        After CASCADE_TRIGGER_STEP steps, check every *original* untriaged
        alert with critical/high severity.  For each eligible alert that has
        not already spawned, create one new dependent alert on a downstream
        service chosen deterministically from the service graph.

        Determinism contract
        --------------------
        - Eligible alerts are iterated in sorted alert_id order.
        - Downstream service is the first dependent (alphabetically) of the
          alert's originating service that does not already have an alert in
          the current alert set.  If all dependents are already covered, the
          first dependent is used anyway (duplicates are realistic).
        - The generated alert uses a fixed metric/threshold/message template
          so the output is fully deterministic given the same state.
        """
        if self._step_count < CASCADE_TRIGGER_STEP:
            return

        # Collect eligible alerts: original, untriaged, critical/high
        eligible: list[Alert] = []
        for alert in sorted(self._alerts, key=lambda a: a.alert_id):
            if alert.alert_id not in self._original_alert_ids:
                continue  # skip dynamic alerts
            if alert.alert_id in self._spawned_from:
                continue  # already spawned
            if alert.triaged:
                continue  # already handled by agent
            # Look up true severity from ground truth
            gt = self._get_ground_truth(alert.alert_id)
            if gt is None:
                continue
            if gt["true_severity"] not in CASCADE_ELIGIBLE_SEVERITIES:
                continue
            eligible.append(alert)

        if not eligible:
            return

        # Set of services that already have alerts (to prefer fresh services)
        existing_services = {a.service for a in self._alerts}

        # Compute the current maximum timestamp for ordering new alerts after
        max_ts = max(a.timestamp for a in self._alerts) if self._alerts else "2024-01-15T10:00:00Z"

        spawned_names: list[str] = []

        for alert in eligible:
            dependents = get_dependents(alert.service)
            if not dependents:
                continue  # top-of-chain service; no downstream to cascade to

            # Prefer a dependent service not already in the alert set
            target_svc = dependents[0]  # fallback: first alphabetically
            for dep in dependents:
                if dep not in existing_services:
                    target_svc = dep
                    break

            dyn_id = f"dyn-{alert.alert_id}"

            # Build a deterministic timestamp 1 minute after current max
            # (simple string increment for ISO-8601 lexicographic ordering)
            dyn_ts = max_ts  # same timestamp batch (deterministic)

            dyn_alert = Alert(
                alert_id=dyn_id,
                timestamp=dyn_ts,
                service=target_svc,
                metric="upstream_error_rate",
                metric_value=45.0,
                threshold=20.0,
                message=(
                    f"{target_svc} upstream errors surging — "
                    f"cascade from unresolved {alert.service} incident "
                    f"(spawned by {alert.alert_id})"
                ),
                context=(
                    f"Dynamic cascade: {alert.service} has been degraded "
                    f"for {self._step_count} steps without triage"
                ),
                triaged=False,
                agent_decision=None,
            )

            dyn_gt: dict[str, Any] = {
                "alert_id":         dyn_id,
                "true_root_cause":  "dependency_outage",
                "true_severity":    "high",
                "true_remediation": "acknowledge_and_monitor",
                "incident_id":      None,
                "dynamic_generated": True,
                "spawned_from_alert_id": alert.alert_id,
            }

            self._alerts.append(dyn_alert)
            self._ground_truth.append(dyn_gt)
            self._dynamic_alert_ids.add(dyn_id)
            self._spawned_from.add(alert.alert_id)
            existing_services.add(target_svc)
            spawned_names.append(dyn_id)

        if spawned_names:
            self._cascade_feedback = (
                f"⚠ Cascade: {len(spawned_names)} new alert(s) spawned "
                f"due to unresolved incidents: {', '.join(spawned_names)}."
            )
