"""
server/app.py

FastAPI application for the Cloud Alert Triage OpenEnv environment.
Implemented in Phase 7.

Endpoints
---------
    POST /reset   -- start a new episode; returns Observation
    POST /step    -- apply one action; returns StepResult
    GET  /state   -- full internal state (debug/grading only)
    GET  /health  -- liveness probe

Design notes
------------
- One global AlertTriageEnv instance shared across all requests.
  Acceptable for a single-worker hackathon deployment; not thread-safe.
- Calling POST /reset discards any in-progress episode (no lock needed).
- ValueError from env.reset() (bad task_id) → 422 Unprocessable Entity.
- RuntimeError from env.step() (step before reset) → 400 Bad Request.
- Pydantic validation errors on the request body → 422 (FastAPI default).
- CORS is fully open (allow_origins=["*"]) — fine for an open evaluation env.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from server.environment import AlertTriageEnv
from server.models import Action, ResetRequest

# ---------------------------------------------------------------------------
# Global environment instance
# ---------------------------------------------------------------------------

env = AlertTriageEnv()

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cloud Alert Triage — OpenEnv",
    description=(
        "An SRE alert triage environment where an AI agent must classify, "
        "prioritise, and remediate cloud infrastructure monitoring alerts."
    ),
    version="1.0.0",
)

# ── CORS (fully open — required for OpenEnv evaluation harness) ────────────
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
    """
    ValueError from env.reset() (unknown task_id) → 422.
    FastAPI already raises 422 for Pydantic validation errors; this handler
    covers ValueError raised inside route handlers.
    """
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
    """RuntimeError from env.step() before reset → 400."""
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    """Return 200 with {\"status\": \"ok\"} so load-balancers know the pod is alive."""
    return {"status": "ok"}


@app.post("/reset", summary="Start a new episode")
async def reset(body: ResetRequest) -> dict[str, Any]:
    """
    Initialise a new episode (discards any in-progress episode).

    Request body
    ------------
    {
        "task_id": "easy" | "medium" | "hard",
        "seed":    int
    }

    Response
    --------
    {
        "observation": { ...Observation fields... }
    }

    Errors
    ------
    422  unknown task_id
    """
    observation = env.reset(body.task_id, body.seed)
    return {"observation": observation.model_dump()}


@app.post("/step", summary="Apply one action to the environment")
async def step(body: Action) -> dict[str, Any]:
    """
    Apply one agent action and advance the episode by one step.

    Request body
    ------------
    One of:
      { "action_type": "triage", "alert_id": "...", "root_cause": "...",
        "severity": "...", "remediation": "..." }
      { "action_type": "link_alerts", "alert_ids": [...], "incident_label": "..." }
      { "action_type": "skip", "alert_id": "..." }

    Response
    --------
    {
        "observation": { ...Observation fields... },
        "reward":      float,
        "done":        bool,
        "info":        {}  |  {"grader_score": float}
    }

    Errors
    ------
    400  called before any /reset (no active episode)
    422  malformed action body (Pydantic validation failure)
    """
    result = env.step(body)
    return result.model_dump()


@app.get("/state", summary="Inspect full internal state (debug / grading)")
async def state() -> dict[str, Any]:
    """
    Return the full EnvironmentState including hidden ground truth.

    Intended for the evaluation harness and debugging only.
    The inference agent must NOT call this endpoint — it would give the
    agent access to labels it is supposed to infer.

    Response
    --------
    EnvironmentState as JSON (see server/models.py for schema).

    Errors
    ------
    400  called before any /reset
    """
    if not env._active:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )
    return env.state().model_dump()

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
