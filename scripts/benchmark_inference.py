#!/usr/bin/env python3
"""
Benchmark inference.py across tasks and seeds, reporting real grader scores.

Usage (run from the project root):
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py easy
    python scripts/benchmark_inference.py easy --seeds 42,123

Log format contract (from inference.py):
    stdout: [START] task=<t> env=cloud-alert-triage model=<m>
    stdout: [STEP]  step=<n> action=... reward=<r> done=<b> error=<e>
    stdout: [END]   success=<b> steps=<n> rewards=<r1,r2,...>
    stderr: [SCORE] task=<t> grader_score=<f>   <- grader score is on STDERR

All grader scores are guaranteed in (0.001, 0.999) -- never 0.0 or 1.0.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections import defaultdict

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Run from project root -- inference.py and server/ are both here.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SERVER_CMD = ["uvicorn", "server.app:app", "--port", "7860", "--host", "0.0.0.0"]
DEFAULT_TASKS = ["easy", "medium", "hard"]
DEFAULT_SEEDS = [42, 123, 456]


def start_server() -> subprocess.Popen:
    """Start the uvicorn server in the background and wait until healthy."""
    proc = subprocess.Popen(
        SERVER_CMD,
        cwd=PROJECT_ROOT,       # FIX: run from project root, not a subdir
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Poll /health for up to 15 s before giving up.
    deadline = time.time() + 15.0
    while time.time() < deadline:
        time.sleep(1)
        try:
            r = httpx.get("http://localhost:7860/health", timeout=3)
            if r.status_code == 200:
                return proc
        except Exception:
            pass

    proc.terminate()
    raise RuntimeError(
        "Server failed to become healthy within 15 s. "
        "Check that port 7860 is free and dependencies are installed."
    )


def run_inference(seed: int, api_key: str | None) -> dict:
    """
    Run inference.py (which always runs easy/medium/hard sequentially).
    Parse grader scores from stderr [SCORE] lines and step counts from stdout.

    Returns {"scores": {task: float}, "steps": {task: int}}.
    """
    env = os.environ.copy()
    env["ENV_URL"] = "http://localhost:7860"
    if api_key:
        env.setdefault("OPENAI_API_KEY", api_key)

    proc = subprocess.run(
        ["python", "inference.py"],
        cwd=PROJECT_ROOT,       # FIX: run from project root
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    # ── step counts from stdout ──────────────────────────────────────────────
    # [START] task=easy ...
    # [END] success=true steps=5 rewards=...
    task_order: list[str] = [m.group(1) for m in re.finditer(r"\[START\] task=(\w+)", proc.stdout)]
    end_steps: list[int] = [int(m.group(1)) for m in re.finditer(r"\[END\] success=\w+ steps=(\d+)", proc.stdout)]
    steps_by_task: dict[str, int] = {t: s for t, s in zip(task_order, end_steps)}

    # ── grader scores from stderr ────────────────────────────────────────────
    # [SCORE] task=easy grader_score=0.999
    scores: dict[str, float] = {}
    for m in re.finditer(r"\[SCORE\] task=(\w+) grader_score=([\d.]+)", proc.stderr):
        scores[m.group(1)] = float(m.group(2))

    if not scores:
        raise ValueError(
            f"No [SCORE] lines found in stderr for seed={seed}.\n"
            f"--- stdout (last 800) ---\n{proc.stdout[-800:]}\n"
            f"--- stderr (last 800) ---\n{proc.stderr[-800:]}"
        )

    # Validate strict open interval.
    for t, s in scores.items():
        if not (0.0 < s < 1.0):
            raise ValueError(
                f"Grader score {s} for task={t} seed={seed} is outside "
                f"the required open interval (0, 1). Check grading.py clamp."
            )

    return {"scores": scores, "steps": steps_by_task}


def main() -> None:
    tasks = [sys.argv[1]] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else DEFAULT_TASKS

    seeds = DEFAULT_SEEDS
    if "--seeds" in sys.argv:
        idx = sys.argv.index("--seeds")
        seeds = [int(s) for s in sys.argv[idx + 1].split(",")]

    api_key: str | None = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("HF_TOKEN")
    )

    print(f"Benchmark: tasks={tasks}  seeds={seeds}")
    key_source = (
        "OPENAI_API_KEY" if os.environ.get("OPENAI_API_KEY")
        else "GROQ_API_KEY" if os.environ.get("GROQ_API_KEY")
        else "HF_TOKEN" if os.environ.get("HF_TOKEN")
        else "NONE — inference will warn"
    )
    print(f"API key:   {key_source}\n")

    server = start_server()
    print("Server healthy. Running inference...\n")

    try:
        results: dict[str, list[float]] = defaultdict(list)

        for seed in seeds:
            print(f"Seed {seed}:")
            try:
                run_result = run_inference(seed, api_key)
                for t, score in run_result["scores"].items():
                    if t in tasks:
                        results[t].append(score)
                        steps = run_result["steps"].get(t, "?")
                        print(f"  {t:8s}  score={score:.4f}  steps={steps}")
            except Exception as exc:
                print(f"  ERROR: {exc}")
            print()

        if not results:
            print("No results collected.")
            return

        print("=== BENCHMARK RESULTS ===")
        all_scores: list[float] = []
        for t in tasks:
            if t not in results:
                continue
            s_list = results[t]
            mean = sum(s_list) / len(s_list)
            std  = (sum((s - mean) ** 2 for s in s_list) / len(s_list)) ** 0.5
            print(f"  {t.capitalize():8s}  mean={mean:.4f}  std={std:.4f}  n={len(s_list)}")
            all_scores.extend(s_list)

        if all_scores:
            overall = sum(all_scores) / len(all_scores)
            print(f"  {'Overall':8s}  mean={overall:.4f}")

        bad = [s for s in all_scores if not (0.0 < s < 1.0)]
        if bad:
            print(f"\n⚠  {len(bad)} score(s) outside open interval (0,1): {bad}")
        else:
            print("\n✓  All scores confirmed in open interval (0, 1).")

    finally:
        server.terminate()


if __name__ == "__main__":
    main()
