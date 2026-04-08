"""
server/app.py

FastAPI application for the Cloud Alert Triage OpenEnv environment.

Endpoints
---------
    POST /reset                 -- start a new episode; returns Observation
    POST /step                  -- apply one action; returns StepResult
    GET  /state                 -- full internal state (debug/grading only)
    GET  /health                -- liveness probe
    GET  /tasks                 -- list all task configurations

    ── Adaptive curriculum (NEW) ──────────────────────────────────────────
    GET  /curriculum/stats      -- current curriculum state + recommended next task
    POST /curriculum/record     -- record an episode result; updates mastery tracking
    POST /curriculum/reset_task -- get next adaptive task params (then call /reset)

    ── LLM judge (NEW) ───────────────────────────────────────────────────
    POST /judge                 -- evaluate reasoning quality of a completed episode

    ── Adaptive reset (NEW) ──────────────────────────────────────────────
    POST /reset/adaptive        -- reset with curriculum-chosen task + targeted scenario
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from server.environment import AlertTriageEnv
from server.models import Action, ResetRequest
from server.curriculum import CurriculumController, EpisodeResult
from server.adaptive_scenario import AdaptiveScenarioGenerator
from server.judge import TriageJudge

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

env        = AlertTriageEnv()
curriculum = CurriculumController()
adaptive   = AdaptiveScenarioGenerator()
judge      = TriageJudge()

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cloud Alert Triage — OpenEnv",
    description=(
        "An SRE alert triage environment where an AI agent must classify, "
        "prioritise, and remediate cloud infrastructure monitoring alerts — "
        "with adaptive difficulty, curriculum learning, and LLM reasoning evaluation."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})

# ---------------------------------------------------------------------------
# Core OpenEnv routes (unchanged)
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", summary="Home")
async def home() -> dict[str, str]:
    return {"message": "CloudAlert Triage AI v2.0 — with adaptive curriculum + LLM judge"}


@app.get("/tasks", summary="List all task configurations")
async def list_tasks() -> list[dict[str, Any]]:
    tasks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tasks")
    tasks = []
    for name in ["easy", "medium", "hard"]:
        path = os.path.join(tasks_dir, f"task_{name}.json")
        with open(path) as f:
            tasks.append(json.load(f))
    return tasks


@app.post("/reset", summary="Start a new episode (fixed task)")
async def reset(body: ResetRequest | None = None) -> dict[str, Any]:
    if body is None:
        body = ResetRequest()
    observation = env.reset(body.task_id, body.seed)
    return {"observation": observation.model_dump()}


@app.post("/step", summary="Apply one action to the environment")
async def step(body: Action) -> dict[str, Any]:
    result = env.step(body)
    return result.model_dump()


@app.get("/state", summary="Inspect full internal state (debug / grading)")
async def state() -> dict[str, Any]:
    if not env._active:
        raise HTTPException(status_code=400, detail="No active episode. Call POST /reset first.")
    return env.state().model_dump()

# ---------------------------------------------------------------------------
# Adaptive reset (NEW)
# ---------------------------------------------------------------------------

@app.post("/reset/adaptive", summary="Start an episode chosen by the curriculum")
async def reset_adaptive() -> dict[str, Any]:
    """
    Let the curriculum controller choose the next task and seed.

    The scenario is generated with adaptive targeting — biased toward the
    root-cause types where the agent is currently weakest.

    Response includes the standard observation PLUS curriculum metadata
    so the agent/trainer knows what was chosen and why.
    """
    params = curriculum.next_task_params()
    task_id = params["task_id"]
    seed    = params["seed"]

    # Generate scenario with adaptive biasing
    scenario = adaptive.generate(
        task_id=task_id,
        seed=seed,
        target_rc=params.get("target_rc"),
        weak_spots=params.get("weak_spots", []),
    )

    # Reset the environment with the adapted scenario injected
    observation = env.reset(task_id, seed)

    return {
        "observation":  observation.model_dump(),
        "curriculum": {
            "task_id":         task_id,
            "seed":            seed,
            "tier":            params["tier"],
            "difficulty":      params["difficulty"],
            "target_rc":       params.get("target_rc"),
            "weak_spots":      params.get("weak_spots", []),
            "adaptive_meta":   scenario.get("adaptive_metadata", {}),
        },
    }

# ---------------------------------------------------------------------------
# Curriculum endpoints (NEW)
# ---------------------------------------------------------------------------

class EpisodeResultRequest(BaseModel):
    """Request body for POST /curriculum/record."""
    task_id:         str
    grader_score:    float
    steps_used:      int
    max_steps:       int
    root_cause_hits: dict[str, bool] = {}
    seed:            int = 42


@app.get("/curriculum/stats", summary="Current curriculum state and recommended next task")
async def curriculum_stats() -> dict[str, Any]:
    """
    Returns the curriculum controller's full state:
    - Current difficulty tier and continuous difficulty score
    - Per-root-cause skill profile (mastery %)
    - Weak spots (root-cause types below mastery threshold)
    - Graduated root-cause types (fully mastered)
    - Recommended next task parameters (task_id, seed, target_rc)
    - Recent performance statistics
    """
    return curriculum.get_stats()


@app.post("/curriculum/record", summary="Record episode result and update mastery tracking")
async def curriculum_record(body: EpisodeResultRequest) -> dict[str, Any]:
    """
    Record the outcome of a completed episode.

    Call this after every episode (when done=True) to update the
    curriculum controller's mastery tracking.  The controller will
    update difficulty, check for tier advancement, and mark any
    root-cause types as mastered.

    Request body
    ------------
    {
        "task_id":         "easy" | "medium" | "hard",
        "grader_score":    float,   // from info["grader_score"] at done=True
        "steps_used":      int,
        "max_steps":       int,
        "root_cause_hits": {"resource_exhaustion": true, "deployment_bug": false},
        "seed":            int
    }

    Response
    --------
    Updated curriculum stats after recording.
    """
    result = EpisodeResult(
        task_id=body.task_id,
        grader_score=body.grader_score,
        steps_used=body.steps_used,
        max_steps=body.max_steps,
        root_cause_hits=body.root_cause_hits,
        seed=body.seed,
    )
    curriculum.record(result)
    return {
        "recorded": True,
        "curriculum": curriculum.get_stats(),
    }


@app.get("/curriculum/next", summary="Get recommended next task params without resetting")
async def curriculum_next() -> dict[str, Any]:
    """
    Returns the curriculum's recommended next task parameters without
    starting a new episode.  Use this to inspect what would be chosen
    before calling /reset/adaptive.
    """
    return curriculum.next_task_params()

# ---------------------------------------------------------------------------
# LLM Judge endpoint (NEW)
# ---------------------------------------------------------------------------

class JudgeRequest(BaseModel):
    """Request body for POST /judge."""
    persona: str = "senior"   # "junior" | "senior" | "principal"


@app.post("/judge", summary="Evaluate reasoning quality of the current (completed) episode")
async def evaluate_reasoning(body: JudgeRequest) -> dict[str, Any]:
    """
    Run the LLM reasoning judge on the most recently completed episode.

    Must be called AFTER the episode is done (done=True from /step).
    Evaluates the quality of the agent's causal reasoning across 5 dimensions:
      - causal_reasoning:           Did metrics → root cause correctly?
      - cascade_awareness:          Were correlated alerts grouped?
      - prioritisation:             Were critical alerts handled first?
      - false_alarm_discrimination: Was noise correctly identified?
      - efficiency:                 Was the step budget used well?

    Persona controls strictness:
      "junior"    — lenient, gives hints
      "senior"    — standard SRE expectations
      "principal" — strict, penalises inefficiency

    The judge score is separate from grader_score — it rewards good
    reasoning even when exact labels are wrong.

    Response
    --------
    {
        "reasoning_score":   float,  // overall reasoning quality in (0, 1)
        "feedback":          str,    // specific critique
        "component_scores":  {       // per-dimension breakdown
            "causal_reasoning": float,
            "cascade_awareness": float,
            "prioritisation": float,
            "false_alarm_discrimination": float,
            "efficiency": float
        },
        "persona":           str,
        "latency_ms":        float,
        "heuristic_fallback": bool  // true if LLM was unavailable
    }

    Errors
    ------
    400  episode not yet complete (call after done=True)
    """
    if not env._active or not env._done:
        raise HTTPException(
            status_code=400,
            detail="Episode not complete. Call POST /step until done=True, then POST /judge.",
        )

    episode_state = env.state().model_dump()
    result = judge.evaluate(episode_state, persona=body.persona)

    return {
        "reasoning_score":    result.reasoning_score,
        "feedback":           result.feedback,
        "component_scores":   result.component_scores,
        "persona":            result.persona,
        "latency_ms":         round(result.latency_ms, 1),
        "heuristic_fallback": result.heuristic_fallback,
    }


def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
        reload=False,
    )


if __name__ == "__main__":
    main()
