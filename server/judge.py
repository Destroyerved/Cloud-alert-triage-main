"""
server/judge.py

LLM-based Reasoning Judge for the Cloud Alert Triage environment.

Unlike the deterministic grader (grading.py) which scores exact-match
accuracy on root_cause / severity / remediation labels, the judge evaluates
the *quality of the agent's reasoning process* — the causal chain from
observed metrics to identified root cause.

This produces a richer training signal for RL agents: partial credit for
good reasoning that reached a wrong label, penalties for lucky guesses with
no causal justification, and explicit feedback for improvement.

Public API
----------
    judge  = TriageJudge()
    result = judge.evaluate(episode_state, persona="senior")

    # result is a JudgeResult with:
    #   reasoning_score  : float in (0, 1)
    #   feedback         : str   (what the agent did well / poorly)
    #   component_scores : dict  (breakdown per dimension)
    #   persona          : str

Personas
--------
    "junior"    — lenient, gives hints, partial credit for any causal attempt
    "senior"    — standard SRE expectations, rewards systematic diagnosis
    "principal" — strict, penalises inefficiency, rewards elegant root-cause chains

The judge is called via POST /judge after an episode ends (done=True).
It does NOT replace the deterministic grader — it adds a complementary
signal for agents that want to optimise reasoning quality, not just labels.

Environment variable
--------------------
    JUDGE_API_KEY    : API key for the judge LLM (falls back to HF_TOKEN)
    JUDGE_MODEL      : model to use (default: claude-haiku-4-5-20251001 for speed)
    JUDGE_API_BASE   : base URL (default: https://api.anthropic.com/v1 — OpenAI compat)
    JUDGE_ENABLED    : set to "false" to disable LLM calls and return heuristic scores
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

_JUDGE_ENABLED  = os.environ.get("JUDGE_ENABLED", "true").lower() != "false"
_JUDGE_MODEL    = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5-20251001")
_JUDGE_API_BASE = os.environ.get("JUDGE_API_BASE", "https://api.anthropic.com/v1")
_JUDGE_API_KEY  = os.environ.get("JUDGE_API_KEY") or os.environ.get("HF_TOKEN", "")

# Persona prompt fragments
_PERSONA_INSTRUCTIONS: dict[str, str] = {
    "junior": (
        "You are a lenient junior SRE reviewing a trainee's triage decisions. "
        "Give partial credit for any causal reasoning attempt, even if imperfect. "
        "Provide hints for improvement. Score generously — the goal is encouragement."
    ),
    "senior": (
        "You are a senior SRE evaluating a colleague's incident triage. "
        "Expect systematic diagnosis: metric analysis → dependency reasoning → "
        "root cause identification → appropriate remediation. "
        "Award full marks only for complete causal chains."
    ),
    "principal": (
        "You are a principal SRE with very high standards. "
        "Penalise guessing, inefficiency, and missing cascade reasoning. "
        "Reward only precise, elegant, fully-justified triage decisions. "
        "Be strict — partial credit is rare."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    """Output from TriageJudge.evaluate()."""
    reasoning_score:   float                    # overall in (0, 1)
    feedback:          str                      # natural language critique
    component_scores:  dict[str, float] = field(default_factory=dict)
    persona:           str = "senior"
    latency_ms:        float = 0.0
    heuristic_fallback: bool = False            # True if LLM call failed


# ─────────────────────────────────────────────────────────────────────────────
# TriageJudge
# ─────────────────────────────────────────────────────────────────────────────

class TriageJudge:
    """
    Evaluates triage reasoning quality using an LLM judge.

    Falls back to a heuristic scorer if JUDGE_ENABLED=false, the API key
    is missing, or the LLM call fails — so the environment stays operational
    even without an API key.
    """

    def __init__(self) -> None:
        self._client = None
        if _JUDGE_ENABLED and _JUDGE_API_KEY:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=_JUDGE_API_BASE,
                    api_key=_JUDGE_API_KEY,
                )
                logger.info("TriageJudge: LLM judge enabled (model=%s)", _JUDGE_MODEL)
            except ImportError:
                logger.warning("TriageJudge: openai package not found — using heuristic fallback")
        else:
            logger.info("TriageJudge: disabled or no API key — using heuristic fallback")

    def evaluate(
        self,
        episode_state: dict[str, Any],
        persona: str = "senior",
    ) -> JudgeResult:
        """
        Evaluate the quality of an agent's triage decisions.

        Parameters
        ----------
        episode_state : dict — the full state snapshot from AlertTriageEnv.state()
                        Must include: ground_truth, agent_decisions, alerts, task_id
        persona       : "junior" | "senior" | "principal"

        Returns
        -------
        JudgeResult
        """
        if persona not in _PERSONA_INSTRUCTIONS:
            persona = "senior"

        t0 = time.monotonic()

        # Always run heuristic component scores (fast, no API call)
        heuristic = self._heuristic_scores(episode_state)

        if self._client is None:
            return JudgeResult(
                reasoning_score=heuristic["overall"],
                feedback=heuristic["feedback"],
                component_scores=heuristic["components"],
                persona=persona,
                latency_ms=0.0,
                heuristic_fallback=True,
            )

        # Try LLM evaluation
        try:
            result = self._llm_evaluate(episode_state, persona, heuristic, t0)
            return result
        except Exception as exc:
            logger.warning("TriageJudge: LLM call failed (%s) — using heuristic", exc)
            return JudgeResult(
                reasoning_score=heuristic["overall"],
                feedback=heuristic["feedback"],
                component_scores=heuristic["components"],
                persona=persona,
                latency_ms=(time.monotonic() - t0) * 1000,
                heuristic_fallback=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # LLM path
    # ─────────────────────────────────────────────────────────────────────────

    def _llm_evaluate(
        self,
        episode_state: dict[str, Any],
        persona: str,
        heuristic: dict,
        t0: float,
    ) -> JudgeResult:
        """Call the LLM judge and parse its structured response."""
        prompt = self._build_prompt(episode_state, persona, heuristic)

        response = self._client.chat.completions.create(
            model=_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=600,
        )
        raw = response.choices[0].message.content.strip()
        latency = (time.monotonic() - t0) * 1000

        # Parse JSON from response
        parsed = self._parse_llm_response(raw, heuristic)

        # Clamp to strictly (0, 1) — same rule as the grader
        score = max(1e-4, min(1.0 - 1e-4, parsed["score"]))

        return JudgeResult(
            reasoning_score=score,
            feedback=parsed["feedback"],
            component_scores=parsed.get("components", heuristic["components"]),
            persona=persona,
            latency_ms=latency,
            heuristic_fallback=False,
        )

    def _build_prompt(
        self,
        episode_state: dict[str, Any],
        persona: str,
        heuristic: dict,
    ) -> str:
        """Build the judge prompt from episode state."""
        task_id         = episode_state.get("task_id", "unknown")
        ground_truth    = episode_state.get("ground_truth", [])
        agent_decisions = episode_state.get("agent_decisions", [])
        agent_links     = episode_state.get("agent_links", [])

        # Build a concise triage summary (cap at 15 decisions to fit context)
        triage_decisions = [d for d in agent_decisions if d.get("action_type") == "triage"][:15]
        skip_decisions   = [d for d in agent_decisions if d.get("action_type") == "skip"]

        gt_by_id = {gt["alert_id"]: gt for gt in ground_truth}

        decision_lines = []
        for d in triage_decisions:
            aid  = d["alert_id"]
            gt   = gt_by_id.get(aid, {})
            correct_rc  = d.get("root_cause") == gt.get("true_root_cause")
            correct_rem = d.get("remediation") == gt.get("true_remediation")
            decision_lines.append(
                f"  {aid}: rc={d.get('root_cause')} (✓)" if correct_rc
                else f"  {aid}: rc={d.get('root_cause')} (✗ should be {gt.get('true_root_cause')})"
            )
            if not correct_rem:
                decision_lines[-1] += f" | rem={d.get('remediation')} (✗ should be {gt.get('true_remediation')})"

        heuristic_summary = json.dumps(heuristic["components"], indent=2)

        return f"""{_PERSONA_INSTRUCTIONS[persona]}

You are evaluating an AI agent's cloud infrastructure alert triage for task: {task_id.upper()}.

HEURISTIC SCORES (exact-match accuracy):
{heuristic_summary}

AGENT DECISIONS ({len(triage_decisions)} triaged, {len(skip_decisions)} skipped, {len(agent_links)} incident links):
{chr(10).join(decision_lines) if decision_lines else "  (no triage decisions recorded)"}

INCIDENT LINKS: {len(agent_links)} group(s) submitted

Evaluate the REASONING QUALITY — not just whether labels are right, but whether:
1. Causal chain: Did the agent trace metrics → root cause correctly?
2. Cascade awareness: Did it identify related alerts as incidents?
3. Prioritisation: Did it triage critical alerts first?
4. False alarm discrimination: Did it correctly identify noise?
5. Efficiency: Did it avoid redundant or contradictory actions?

Respond ONLY with valid JSON, no other text:
{{
  "score": <float between 0.05 and 0.95>,
  "feedback": "<2-3 sentences of specific, actionable critique>",
  "components": {{
    "causal_reasoning": <0.0-1.0>,
    "cascade_awareness": <0.0-1.0>,
    "prioritisation": <0.0-1.0>,
    "false_alarm_discrimination": <0.0-1.0>,
    "efficiency": <0.0-1.0>
  }}
}}"""

    def _parse_llm_response(self, raw: str, heuristic: dict) -> dict:
        """Parse LLM JSON response, falling back to heuristic on parse failure."""
        # Strip markdown code fences if present
        clean = raw.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(clean)
            if "score" not in parsed:
                raise ValueError("missing score field")
            # Clamp scores to ensure strictly between 0 and 1
            parsed["score"] = max(0.01, min(0.99, parsed["score"]))
            if "components" in parsed:
                for k in parsed["components"]:
                    parsed["components"][k] = max(0.0, min(1.0, parsed["components"][k]))
            return parsed
        except Exception:
            logger.warning("TriageJudge: failed to parse LLM response — using heuristic")
            return {"score": heuristic["overall"], "feedback": heuristic["feedback"],
                    "components": heuristic["components"]}

    # ─────────────────────────────────────────────────────────────────────────
    # Heuristic fallback scorer (no LLM required)
    # ─────────────────────────────────────────────────────────────────────────

    def _heuristic_scores(self, episode_state: dict[str, Any]) -> dict:
        """
        Fast heuristic scoring across 5 reasoning dimensions.
        Used both as a fallback and as context for the LLM judge.
        """
        ground_truth    = episode_state.get("ground_truth", [])
        agent_decisions = episode_state.get("agent_decisions", [])
        agent_links     = episode_state.get("agent_links", [])
        incidents       = episode_state.get("incidents", [])

        gt_by_id       = {gt["alert_id"]: gt for gt in ground_truth}
        triage_by_id   = {d["alert_id"]: d for d in agent_decisions if d.get("action_type") == "triage"}
        skips           = {d["alert_id"] for d in agent_decisions if d.get("action_type") == "skip"}

        n_total = len(ground_truth) or 1

        # 1. Causal reasoning — root cause accuracy
        rc_correct = sum(
            1 for gt in ground_truth
            if triage_by_id.get(gt["alert_id"], {}).get("root_cause") == gt["true_root_cause"]
        )
        causal = rc_correct / n_total

        # 2. Cascade awareness — incident link quality (simplified F1)
        incident_aids: set[str] = set()
        for inc in incidents:
            for aid in inc.get("alert_ids", []):
                incident_aids.add(aid)

        linked_aids: set[str] = set()
        for link in agent_links:
            for aid in link.get("alert_ids", []):
                linked_aids.add(aid)

        if incident_aids:
            tp = len(incident_aids & linked_aids)
            cascade = tp / len(incident_aids)
        else:
            cascade = 1.0   # no incidents → vacuously correct

        # 3. Prioritisation — did critical alerts get triaged (not skipped)?
        critical_alerts = [gt for gt in ground_truth if gt["true_severity"] == "critical"]
        if critical_alerts:
            handled_critical = sum(
                1 for gt in critical_alerts
                if gt["alert_id"] in triage_by_id
            )
            prioritisation = handled_critical / len(critical_alerts)
        else:
            prioritisation = 1.0

        # 4. False alarm discrimination — correctly skipped FAs
        fa_alerts = [gt for gt in ground_truth if gt["true_root_cause"] == "false_alarm"]
        if fa_alerts:
            correctly_skipped = sum(1 for gt in fa_alerts if gt["alert_id"] in skips)
            fa_disc = correctly_skipped / len(fa_alerts)
        else:
            fa_disc = 1.0

        # 5. Efficiency — coverage without over-triaging
        covered   = len(triage_by_id) + len(skips)
        coverage  = min(1.0, covered / n_total)
        # Penalise if very low coverage
        efficiency = coverage if coverage >= 0.5 else coverage * 0.5

        components = {
            "causal_reasoning":          round(causal,          3),
            "cascade_awareness":         round(cascade,         3),
            "prioritisation":            round(prioritisation,  3),
            "false_alarm_discrimination": round(fa_disc,        3),
            "efficiency":                round(efficiency,      3),
        }

        # Weighted overall (mirrors what the LLM judge is asked to consider)
        weights = {"causal_reasoning": 0.35, "cascade_awareness": 0.25,
                   "prioritisation": 0.15, "false_alarm_discrimination": 0.15,
                   "efficiency": 0.10}
        overall = sum(components[k] * weights[k] for k in weights)

        # Clamp strictly between 0 and 1
        overall = max(0.01, min(0.99, overall))

        # Build a concise feedback string
        weak = [k for k, v in components.items() if v < 0.6]
        strong = [k for k, v in components.items() if v >= 0.85]
        parts = []
        if strong:
            parts.append(f"Strong: {', '.join(strong)}.")
        if weak:
            parts.append(f"Needs work: {', '.join(weak)}.")
        if not parts:
            parts.append("Solid performance across all dimensions.")
        feedback = " ".join(parts)

        return {
            "overall":    round(overall, 4),
            "components": components,
            "feedback":   feedback,
        }
