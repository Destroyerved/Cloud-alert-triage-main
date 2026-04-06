# OPENENV IMPLEMENTATION MASTER PLAN

**Project**: OpenEnv Hackathon Submission
**Team**: Better Call Coders
**Date**: 2026-04-06
**Status**: PLANNING COMPLETE — READY FOR EXECUTION

---

## TABLE OF CONTENTS

1. Domain Selection
2. Final Environment Concept
3. Repository Structure
4. Implementation Strategy
5. File-by-File Specification
6. Data Models
7. Task Design
8. Reward Design
9. Grader Design
10. API Design
11. openenv.yaml Plan
12. inference.py Plan
13. Testing Strategy
14. Debugging Playbook
15. Human Input Checkpoints
16. Token-Efficiency and Multi-Agent Handoff Protocol
17. Implementation Execution Checklist
18. Risks and Mitigations
19. Final Recommended Build Prompts
20. Output Format Requirements

---

## 1. DOMAIN SELECTION

### Candidate Domains

#### A) IT Helpdesk Ticket Triage & Resolution
- **What**: Agent reads incoming IT support tickets, classifies priority, assigns to correct team, and suggests resolution actions.
- Real-world utility: **9/10** — Every company with >50 employees has this workflow.
- Creativity/novelty: **7/10** — Related to "customer support" but narrower and more technical.
- Ease of deterministic grading: **10/10** — Each ticket has a known correct priority, team, and resolution. Binary per-field, aggregated for partial credit.
- Reward shaping quality: **9/10** — Every action (classify, assign, resolve) gives incremental signal.
- Difficulty scaling: **9/10** — Easy: 5 obvious tickets. Medium: 15 tickets with ambiguity. Hard: 30+ tickets with cascading dependencies, SLA pressure, red herrings.
- Feasibility: **10/10** — Pure text, zero external dependencies, tiny memory footprint.
- Reliability: **9/10** — Deterministic seed-based generation, no randomness at grading time.

#### B) Data Cleaning Pipeline
- **What**: Agent receives messy tabular data and must identify and fix errors (missing values, type mismatches, duplicates, outliers).
- Real-world utility: **9/10** — Universal data task.
- Creativity/novelty: **6/10** — Common hackathon topic.
- Ease of deterministic grading: **8/10** — Can compare cleaned output to ground truth, but edge cases in "acceptable" fixes.
- Reward shaping quality: **7/10** — Per-row or per-cell fix rewards, but some fixes are all-or-nothing.
- Difficulty scaling: **7/10** — Hard to make "hard" task genuinely hard for frontier LLMs since patterns repeat.
- Feasibility: **9/10** — Lightweight, but need to generate realistic dirty data.
- Reliability: **7/10** — More edge cases in grading.

#### C) Cloud Infrastructure Alert Triage
- **What**: Agent monitors a stream of infrastructure alerts (CPU spikes, disk full, latency increase, OOM kills), identifies root causes, sets severity, and takes remediation actions.
- Real-world utility: **10/10** — SRE/DevOps teams do this 24/7.
- Creativity/novelty: **9/10** — Unlikely anyone else picks this. Very specific and technical.
- Ease of deterministic grading: **9/10** — Each alert has a known root cause and correct remediation.
- Reward shaping quality: **9/10** — Partial credit for correct severity even if wrong root cause; bonus for identifying cascading failures.
- Difficulty scaling: **9/10** — Easy: independent alerts. Medium: correlated alerts. Hard: cascading failures with noise alerts masking the real issue.
- Feasibility: **10/10** — Pure text, tiny footprint.
- Reliability: **9/10** — Deterministic generation and grading.

### Comparison Table

| Criterion (weight)         | IT Helpdesk (A) | Data Cleaning (B) | Cloud Alerts (C) |
|---------------------------|-----------------|-------------------|-------------------|
| Real-world utility (30%)  | 9               | 9                 | 10                |
| Task/grader quality (25%) | 10              | 8                 | 9                 |
| Environment design (20%)  | 9               | 7                 | 9                 |
| Code quality (15%)        | 10              | 9                 | 9                 |
| Creativity (10%)          | 7               | 6                 | 9                 |
| **Weighted total**        | **9.15**        | **7.95**          | **9.40**          |

### FINAL CHOICE: C — Cloud Infrastructure Alert Triage

**Justification**: Highest weighted score. Strongest on creativity (judges reward novelty). Extremely real-world — fills a genuine gap since no OpenEnv environment likely models SRE workflows. Grading is clean and deterministic. Difficulty scaling is natural: isolated alerts → correlated alerts → cascading failures with noise. Pure text, zero heavy dependencies, runs on 2 vCPU easily. The hard task is genuinely hard for LLMs because it requires multi-step causal reasoning across correlated events under noise.

---

## 2. FINAL ENVIRONMENT CONCEPT

### What It Simulates
An on-call SRE (Site Reliability Engineer) receives a batch of infrastructure monitoring alerts. The agent must triage each alert by:
1. Reading the alert details (service name, metric, severity hint, timestamp, message)
2. Classifying the **root cause category** (e.g., resource_exhaustion, network_failure, deployment_bug, config_error, dependency_outage, false_alarm)
3. Setting the **severity level** (critical, high, medium, low)
4. Choosing a **remediation action** (restart_service, scale_up, rollback_deploy, fix_config, escalate_to_team, acknowledge_and_monitor, dismiss)
5. Optionally **linking related alerts** (identifying that multiple alerts stem from the same incident)

### Who the Agent Is
The agent is an SRE on-call engineer at a mid-size tech company with ~20 microservices. They receive alerts from a monitoring system (like PagerDuty/Datadog).

### Entities in the World
- **Alerts**: Each has an ID, timestamp, source service, metric name, metric value, threshold, alert message, and optional context (recent deploy, dependency status).
- **Services**: A known set of microservices with dependency relationships (service A depends on service B).
- **Incidents**: Groups of related alerts that share a root cause (hidden from agent, used for grading).
- **Service Dependency Graph**: A simple DAG showing which services depend on which. Visible to the agent.

### Allowed Actions
Single action model with a `type` discriminator:

| Action type | Fields | Meaning |
|------------|--------|---------|
| `triage` | `alert_id`, `root_cause`, `severity`, `remediation` | Classify and act on one alert |
| `link_alerts` | `alert_ids`, `incident_label` | Group alerts as same incident |
| `skip` | `alert_id` | Explicitly skip an alert (acknowledge but take no action) |

### Observations Exposed to Agent
On `reset()` and after each `step()`:
- `alerts`: List of all alerts (full details). Already-triaged alerts are marked with the agent's previous decisions.
- `service_map`: The dependency graph (adjacency list).
- `pending_count`: Number of un-triaged alerts remaining.
- `step_number`: Current step.
- `max_steps`: Step budget for this episode.
- `feedback`: After each triage action, a short text hint (e.g., "Severity accepted" or "Consider checking upstream dependencies"). This gives learning signal without revealing ground truth.

### Hidden State (Not Shown to Agent)
- Ground truth root cause for each alert.
- Ground truth severity for each alert.
- Ground truth optimal remediation for each alert.
- Incident groupings (which alerts belong to the same incident).
- The seed used to generate the scenario.

### Episode Success
An episode ends when:
- All alerts have been triaged or skipped (natural completion), OR
- Step budget is exhausted.

Success is measured by the grader, which compares agent decisions to ground truth across all alerts.

### Incremental Reward Shaping
After every `triage` action:
- +0.3 if `root_cause` is correct
- +0.3 if `severity` is correct
- +0.2 if `remediation` is correct
- +0.1 if `severity` is within 1 level of correct (partial credit, not stacked with exact match)
- +0.1 bonus if alert is part of an incident AND agent has already correctly linked it

After `link_alerts`:
- +0.2 per correctly grouped pair of alerts
- -0.1 per incorrectly grouped pair

After `skip`:
- +0.2 if the alert is genuinely a `false_alarm`
- -0.3 if skipping a real critical/high alert

Penalties:
- -0.05 per step after 80% of step budget used (urgency pressure)
- -0.1 for invalid action format
- -0.2 for triaging an already-triaged alert
- No reward for no-op / doing nothing (budget drains)

### What Makes the Hard Task Hard
- 30+ alerts from 15+ services
- 4-5 distinct incidents where alerts cascade across dependent services
- 5-8 false alarm / noise alerts mixed in
- Ambiguous alerts where the metric alone doesn't distinguish root causes — agent must reason about the dependency graph and temporal correlation
- Some alerts have misleading severity hints (monitoring system says "critical" but it's actually a false alarm from a known flaky check)
- Tight step budget (can't brute-force by trying everything)

---

## 3. REPOSITORY STRUCTURE

```
cloud-alert-triage/
├── openenv.yaml                 # REQUIRED - OpenEnv metadata
├── inference.py                 # REQUIRED - Baseline agent (root level)
├── Dockerfile                   # REQUIRED - Container definition
├── README.md                    # REQUIRED - Documentation
├── requirements.txt             # Python dependencies
├── .env.example                 # Template for env vars
├── Makefile                     # Convenience commands
├── PROGRESS.md                  # Agent/human progress tracker
├── MAP.md                       # Repository map and architecture doc
│
├── server/
│   ├── __init__.py
│   ├── app.py                   # FastAPI application with endpoints
│   ├── config.py                # Configuration / constants
│   ├── models.py                # All Pydantic models
│   ├── environment.py           # Core environment logic (reset/step/state)
│   ├── scenario_generator.py    # Seed-based alert/incident generation
│   ├── rewards.py               # Reward calculation
│   ├── grading.py               # Task graders
│   └── service_graph.py         # Service dependency graph definition
│
├── tasks/
│   ├── task_easy.json           # Task config: easy scenario
│   ├── task_medium.json         # Task config: medium scenario
│   └── task_hard.json           # Task config: hard scenario
│
├── tests/
│   ├── test_models.py           # Pydantic model tests
│   ├── test_environment.py      # Core env logic tests
│   ├── test_rewards.py          # Reward calculation tests
│   ├── test_graders.py          # Grader determinism tests
│   ├── test_api.py              # API endpoint tests
│   └── test_scenario_gen.py     # Scenario generation tests
│
├── scripts/
│   ├── validate.sh              # Local pre-submission validation
│   └── smoke_test.py            # Quick end-to-end smoke test
│
└── docs/
    └── decision_log.md          # Design decisions and rationale
```

### PROGRESS.md Template

```markdown
# PROGRESS TRACKER

## Current Status: PHASE [X] — [Phase Name]

## Phase Completion

| Phase | Status | Last Updated | Agent/Human |
|-------|--------|-------------|-------------|
| 1. Bootstrap | NOT STARTED | — | — |
| 2. Models | NOT STARTED | — | — |
| 3. Scenario Gen | NOT STARTED | — | — |
| 4. Environment Core | NOT STARTED | — | — |
| 5. Rewards | NOT STARTED | — | — |
| 6. Graders | NOT STARTED | — | — |
| 7. API Server | NOT STARTED | — | — |
| 8. openenv.yaml | NOT STARTED | — | — |
| 9. inference.py | NOT STARTED | — | — |
| 10. Tests | NOT STARTED | — | — |
| 11. Docker | NOT STARTED | — | — |
| 12. Local Validation | NOT STARTED | — | — |
| 13. HF Deployment | NOT STARTED | — | — |
| 14. Final Polish | NOT STARTED | — | — |

## Change Log

### [Timestamp] — [Agent ID or Human]
- **Phase**: [N]
- **Files changed**: [list]
- **What was done**: [1-2 sentences]
- **What's next**: [1-2 sentences]
- **Blockers**: [none / description]
- **Assumptions made**: [list or none]
```

### MAP.md Template

```markdown
# REPOSITORY MAP

## Architecture Overview
Cloud Alert Triage OpenEnv environment. FastAPI server exposes /reset, /step, /state.

## File Purposes

| File | Purpose | Depends On |
|------|---------|-----------|
| server/app.py | FastAPI routes | models.py, environment.py |
| server/models.py | All Pydantic schemas | — |
| server/environment.py | Core env logic | models.py, scenario_generator.py, rewards.py |
| server/scenario_generator.py | Generates alerts from seed | models.py, service_graph.py |
| server/rewards.py | Computes per-step rewards | models.py |
| server/grading.py | Task graders (0-1 scores) | models.py |
| server/service_graph.py | Service dependency DAG | — |
| server/config.py | Constants and defaults | — |
| inference.py | Baseline LLM agent | OpenAI SDK, server API |
| openenv.yaml | OpenEnv metadata | — |
| Dockerfile | Container config | requirements.txt |

## Key Design Decisions
- Single Action model with `action_type` discriminator (not multiple models)
- Deterministic scenario generation via seed
- Rewards are per-step; grader score is end-of-episode
- One global environment instance (not per-session) for simplicity
- FastAPI with uvicorn

## Data Flow
reset(task_id, seed) → scenario_generator creates alerts → env stores state → returns observation
step(action) → env validates action → env updates state → rewards.py computes reward → returns (obs, reward, done, info)
state() → returns full internal state including hidden ground truth
```

---

## 4. IMPLEMENTATION STRATEGY

### Phase 1: Bootstrap (30 min)
- **Objective**: Create repo skeleton, all empty files, docs scaffolding.
- **Why first**: Everything else depends on file structure existing.
- **Files created**: All directories, empty `__init__.py` files, `PROGRESS.md`, `MAP.md`, `requirements.txt`, `.env.example`, `Makefile`.
- **Tasks**:
  1. Create directory tree exactly as specified in Section 3
  2. Write `PROGRESS.md` with template from above
  3. Write `MAP.md` with template from above
  4. Write `requirements.txt`: `fastapi`, `uvicorn[standard]`, `pydantic>=2.0`, `openai`, `httpx` (for testing)
  5. Write `.env.example`: `API_BASE_URL=`, `MODEL_NAME=`, `HF_TOKEN=`, `OPENAI_API_KEY=`
  6. Write basic `Makefile` with targets: `run`, `test`, `docker-build`, `docker-run`, `validate`
- **Validation**: `ls -R` shows correct tree. `pip install -r requirements.txt` succeeds.
- **Failure modes**: Wrong directory nesting. Missing `__init__.py`.
- **Fix**: Re-check tree against Section 3.

### Phase 2: Data Models (45 min)
- **Objective**: Implement all Pydantic models in `server/models.py`.
- **Why now**: Every other module imports from models.
- **Files created/edited**: `server/models.py`
- **Tasks**:
  1. Define all models from Section 6
  2. Add validators and field constraints
  3. Write 2-3 inline examples as model_config examples
- **Validation**: `python -c "from server.models import *; print('OK')"`. Run `tests/test_models.py`.
- **Failure modes**: Pydantic v1 vs v2 syntax confusion. Missing Optional fields.
- **Fix**: Use Pydantic v2 syntax exclusively. Test instantiation with minimal and maximal data.

### Phase 3: Service Graph + Scenario Generator (60 min)
- **Objective**: Build deterministic alert/scenario generation.
- **Why now**: Environment core needs scenarios to exist.
- **Files created/edited**: `server/service_graph.py`, `server/scenario_generator.py`, `server/config.py`, `tasks/task_easy.json`, `tasks/task_medium.json`, `tasks/task_hard.json`
- **Tasks**:
  1. Define service dependency graph (15-20 services with clear hierarchy)
  2. Implement `generate_scenario(task_id, seed)` that returns alerts and ground truth
  3. Implement difficulty-specific generators: easy (5 alerts, 0 incidents), medium (15 alerts, 2 incidents), hard (30 alerts, 5 incidents + noise)
  4. Write task config JSON files with seed, params, expected alert counts
- **Validation**: `python -c "from server.scenario_generator import generate_scenario; s = generate_scenario('easy', 42); print(len(s.alerts))"` → 5. Same seed → same output (run twice, compare).
- **Failure modes**: Non-determinism from dict ordering or timestamps. Unrealistic alert messages.
- **Fix**: Use `random.Random(seed)` instance (not global). Sort all outputs. Use fixed relative timestamps.

### Phase 4: Environment Core (60 min)
- **Objective**: Implement `AlertTriageEnv` class with `reset()`, `step()`, `state()`.
- **Why now**: This is the heart of the project.
- **Files created/edited**: `server/environment.py`
- **Tasks**:
  1. Implement `reset(task_id, seed)` → generates scenario, initializes state, returns observation
  2. Implement `step(action)` → validates action, updates state, computes reward, checks done
  3. Implement `state()` → returns full state including ground truth
  4. Handle edge cases: double-triage, invalid alert_id, episode already done
- **Validation**: Manual walkthrough: reset → step (valid triage) → check state updated → step until done. `tests/test_environment.py`.
- **Failure modes**: State mutation bugs. Forgetting to copy state. Off-by-one in step counting.
- **Fix**: Use immutable patterns or explicit deep copies. Add assertion checks in step().

### Phase 5: Reward System (30 min)
- **Objective**: Implement reward calculation.
- **Why now**: Depends on models + environment state.
- **Files created/edited**: `server/rewards.py`
- **Tasks**:
  1. Implement `compute_reward(action, ground_truth, env_state)` returning float
  2. Handle all action types (triage, link_alerts, skip)
  3. Add step-budget pressure penalty
  4. Add invalid action penalty
- **Validation**: Unit tests with known inputs → known rewards. Check reward examples from Section 8.
- **Failure modes**: Reward values not summing correctly. Penalties applied wrong.
- **Fix**: Test each reward component independently.

### Phase 6: Graders (30 min)
- **Objective**: Implement per-task grader functions.
- **Why now**: Depends on environment state at episode end.
- **Files created/edited**: `server/grading.py`
- **Tasks**:
  1. Implement `grade_episode(task_id, final_state)` → float in [0.0, 1.0]
  2. Implement per-field accuracy scoring
  3. Implement incident-linking bonus
  4. Clamp output to [0.0, 1.0]
- **Validation**: Run known-good trajectory → expected score. Run known-bad trajectory → low score. Same input twice → same output (determinism).
- **Failure modes**: Division by zero if no alerts. Score outside [0, 1].
- **Fix**: Guard all divisions. Clamp final output.

### Phase 7: API Server (45 min)
- **Objective**: FastAPI app exposing /reset, /step, /state endpoints.
- **Why now**: Depends on everything above.
- **Files created/edited**: `server/app.py`
- **Tasks**:
  1. Create FastAPI app
  2. POST `/reset` — accepts `{task_id, seed}`, calls `env.reset()`, returns observation
  3. POST `/step` — accepts action JSON, calls `env.step()`, returns `{observation, reward, done, info}`
  4. GET `/state` — returns current state
  5. GET `/health` — returns 200
  6. Add error handling (422 for bad input, 400 for invalid state transitions)
- **Validation**: `uvicorn server.app:app` then `curl -X POST localhost:8000/reset -d '{"task_id":"easy","seed":42}'` → 200 with observation JSON.
- **Failure modes**: CORS issues. Wrong content types. Pydantic serialization errors with nested models.
- **Fix**: Add CORS middleware. Use `.model_dump()` for responses. Test with curl.

### Phase 8: openenv.yaml (15 min)
- **Objective**: Write metadata file.
- **Why now**: Can be written once API shape is finalized.
- **Files created/edited**: `openenv.yaml`
- **Tasks**: Write all fields per Section 11.
- **Validation**: `openenv validate` passes (install openenv-core first).
- **Failure modes**: Wrong field names. Missing required fields. VERIFY MANUALLY against latest openenv-core docs.

### Phase 9: inference.py (60 min)
- **Objective**: Baseline agent using OpenAI SDK.
- **Why now**: Needs working API server.
- **Files created/edited**: `inference.py`
- **Tasks**:
  1. Read env vars (API_BASE_URL, MODEL_NAME, HF_TOKEN/API_KEY)
  2. Connect to environment via HTTP
  3. Run all 3 tasks sequentially
  4. For each task: reset → loop (observe → prompt LLM → parse action → step) → log
  5. Print exact [START], [STEP], [END] logs
  6. Handle LLM errors gracefully (retry once, then default action)
  7. Enforce 20-minute total timeout
- **Validation**: Run against local server with a real or mock LLM. Check log format matches spec exactly.
- **Failure modes**: LLM returns unparseable action. Timeout. Auth errors.
- **Fix**: Wrap LLM calls in try/except. Parse action with fallback to "skip". Add per-task timeout.

### Phase 10: Tests (30 min)
- **Objective**: Core tests pass.
- **Files created/edited**: All files in `tests/`
- **Tasks**: Write tests per Section 13.
- **Validation**: `pytest tests/ -v` all green.

### Phase 11: Docker (30 min)
- **Objective**: Working Dockerfile.
- **Files created/edited**: `Dockerfile`
- **Tasks**:
  1. Use `python:3.11-slim` base
  2. Copy requirements, install deps
  3. Copy source
  4. Expose port 7860 (HF Spaces default)
  5. CMD: `uvicorn server.app:app --host 0.0.0.0 --port 7860`
- **Validation**: `docker build -t alert-triage .` succeeds. `docker run -p 7860:7860 alert-triage` then curl /reset → 200.
- **Failure modes**: Missing system deps. Wrong port. File permissions.
- **Fix**: Add `--no-cache-dir` to pip. Use port 7860. Check COPY paths.

### Phase 12: Local Validation (15 min)
- **Objective**: All pre-submission checks pass locally.
- **Tasks**: Run `scripts/validate.sh` or equivalent manual checks.
- **Validation**: HF Space ping (simulated), docker build, openenv validate all pass.

### Phase 13: HF Deployment (30 min, NEEDS HUMAN)
- **Objective**: Deploy to Hugging Face Spaces.
- **Tasks**:
  1. Human creates HF Space (Docker type, tagged `openenv`)
  2. Push code to Space repo
  3. Wait for build
  4. Test /reset endpoint on live URL
- **Validation**: `curl -X POST https://<space-url>/reset` → 200.

### Phase 14: Final Polish (30 min)
- **Objective**: README, cleanup, final checks.
- **Tasks**:
  1. Write complete README per Section 5 spec
  2. Verify all baseline scores documented
  3. Run full inference.py against live Space
  4. Final code cleanup
- **Validation**: Full checklist from Section 17 all green.

---

## 5. FILE-BY-FILE SPECIFICATION

### `openenv.yaml`
- **Path**: `/openenv.yaml`
- **Purpose**: OpenEnv metadata. Required for `openenv validate`.
- **Required**: YES
- **Contents**: See Section 11.
- **Constraints**: Must pass `openenv validate`. Field names must match OpenEnv spec exactly.

### `inference.py`
- **Path**: `/inference.py`
- **Purpose**: Baseline agent. Must be in root directory.
- **Required**: YES
- **Contents**: See Section 12.
- **Interfaces**:
  - `main()` — async entry point
  - `get_model_action(client, observation, task_id, history)` — calls LLM, returns parsed action dict
  - `log_start(task, env, model)` — prints [START] log
  - `log_step(step, action, reward, done, error)` — prints [STEP] log
  - `log_end(success, steps, score, rewards)` — prints [END] log
- **Constraints**: Must use `from openai import OpenAI`. Must read env vars. Must handle all 3 tasks. Must finish under 20 minutes total.

### `Dockerfile`
- **Path**: `/Dockerfile`
- **Purpose**: Container definition for HF Spaces.
- **Required**: YES
- **Contents**:
  - FROM python:3.11-slim
  - WORKDIR /app
  - COPY requirements.txt .
  - RUN pip install --no-cache-dir -r requirements.txt
  - COPY . .
  - EXPOSE 7860
  - CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
- **Constraints**: Must build on 2 vCPU / 8GB. Port 7860 (HF default).

### `README.md`
- **Path**: `/README.md`
- **Purpose**: Judge-facing documentation.
- **Required**: YES
- **Contents**:
  - Project title and tagline
  - Problem statement: why SRE alert triage matters
  - Environment description (what the agent does)
  - Action space table (all action types with fields and types)
  - Observation space table (all fields with types)
  - Task descriptions table (easy/medium/hard with expected difficulty)
  - Reward design summary
  - Setup instructions (pip install, run locally, run with Docker)
  - Baseline scores table
  - Architecture diagram (text-based)
  - License
- **Constraints**: Must be comprehensive but scannable. Judges skim. Use tables heavily.

### `requirements.txt`
- **Path**: `/requirements.txt`
- **Purpose**: Python dependencies.
- **Required**: YES
- **Contents**:
  ```
  fastapi>=0.100.0
  uvicorn[standard]>=0.23.0
  pydantic>=2.0.0
  openai>=1.0.0
  httpx>=0.24.0
  ```
- **Constraints**: No stdlib modules listed. Pin minimum versions but allow patches.

### `.env.example`
- **Path**: `/.env.example`
- **Purpose**: Template showing required env vars.
- **Required**: Optional but strongly recommended.
- **Contents**:
  ```
  API_BASE_URL=https://api.openai.com/v1
  MODEL_NAME=gpt-4o-mini
  OPENAI_API_KEY=sk-...
  HF_TOKEN=hf_...
  ```

### `Makefile`
- **Path**: `/Makefile`
- **Purpose**: Convenience commands.
- **Required**: Optional but recommended.
- **Contents/targets**:
  - `run`: `uvicorn server.app:app --reload --port 7860`
  - `test`: `pytest tests/ -v`
  - `docker-build`: `docker build -t cloud-alert-triage .`
  - `docker-run`: `docker run -p 7860:7860 cloud-alert-triage`
  - `validate`: `openenv validate`
  - `smoke`: `python scripts/smoke_test.py`
  - `infer`: `python inference.py`

### `server/__init__.py`
- **Path**: `server/__init__.py`
- **Purpose**: Makes server a Python package.
- **Contents**: Empty file.

### `server/config.py`
- **Path**: `server/config.py`
- **Purpose**: All constants, enums, configuration.
- **Contents**:
  - `ROOT_CAUSE_CATEGORIES`: list of valid root cause strings
  - `SEVERITY_LEVELS`: list `["critical", "high", "medium", "low"]`
  - `REMEDIATION_ACTIONS`: list of valid remediation strings
  - `ACTION_TYPES`: list `["triage", "link_alerts", "skip"]`
  - `DEFAULT_PORT`: 7860
  - `MAX_STEPS_BY_TASK`: dict mapping task_id → max steps
  - `SEVERITY_ORDER`: dict mapping severity → numeric rank for proximity scoring

### `server/models.py`
- **Path**: `server/models.py`
- **Purpose**: All Pydantic models.
- **Required**: YES
- **Contents**: See Section 6 for complete model definitions.
- **Constraints**: Pydantic v2 syntax. All fields typed. Validators for enums.

### `server/service_graph.py`
- **Path**: `server/service_graph.py`
- **Purpose**: Defines the microservice dependency graph.
- **Contents**:
  - `SERVICE_GRAPH`: dict mapping service_name → list of dependencies
  - `get_service_names()` → list of all service names
  - `get_dependencies(service)` → list of services it depends on
  - `get_dependents(service)` → list of services that depend on it
  - `get_graph_as_adjacency_list()` → dict for observation serialization
- **Constraints**: 15-20 services. Clear hierarchy (frontend → backend → database/cache/queue pattern). Deterministic.

### `server/scenario_generator.py`
- **Path**: `server/scenario_generator.py`
- **Purpose**: Generates alert scenarios deterministically from seed.
- **Contents**:
  - `generate_scenario(task_id: str, seed: int) -> Scenario` — main entry point
  - `_generate_easy(rng: random.Random) -> Scenario`
  - `_generate_medium(rng: random.Random) -> Scenario`
  - `_generate_hard(rng: random.Random) -> Scenario`
  - `_create_alert(rng, service, root_cause, severity, ...) -> Alert`
  - `_create_incident(rng, root_service, cascade_depth, ...) -> list[Alert]`
  - `_create_noise_alerts(rng, count) -> list[Alert]`
- **Constraints**: Must use `random.Random(seed)` instance for all randomness. Same seed + same task_id → identical output every time.

### `server/environment.py`
- **Path**: `server/environment.py`
- **Purpose**: Core environment logic.
- **Contents**:
  - `class AlertTriageEnv`:
    - `reset(task_id: str, seed: int) -> Observation`
    - `step(action: Action) -> StepResult`
    - `state() -> EnvironmentState`
    - Internal state: `_alerts`, `_ground_truth`, `_agent_decisions`, `_step_count`, `_done`, `_incidents`, `_task_id`
- **Constraints**: Single instance. Not thread-safe (fine for hackathon). Must handle all edge cases (already triaged, invalid alert_id, already done).

### `server/rewards.py`
- **Path**: `server/rewards.py`
- **Purpose**: Reward calculation per step.
- **Contents**:
  - `compute_reward(action: Action, ground_truth: dict, env_state: dict) -> float`
  - `_reward_triage(action, truth) -> float`
  - `_reward_link(alert_ids, true_incidents) -> float`
  - `_reward_skip(alert_id, truth) -> float`
  - `_penalty_budget(step, max_steps) -> float`
- **Constraints**: Deterministic. Returns float. No side effects.

### `server/grading.py`
- **Path**: `server/grading.py`
- **Purpose**: End-of-episode scoring.
- **Contents**:
  - `grade_episode(task_id: str, final_state: EnvironmentState) -> float`
  - `_accuracy_score(decisions, ground_truth, field) -> float`
  - `_incident_linking_score(agent_links, true_incidents) -> float`
- **Constraints**: Output always in [0.0, 1.0]. Deterministic. No side effects.

### `server/app.py`
- **Path**: `server/app.py`
- **Purpose**: FastAPI HTTP server.
- **Contents**:
  - FastAPI app instance
  - `POST /reset` — body: `{task_id: str, seed: int}` — returns Observation
  - `POST /step` — body: Action — returns StepResult
  - `GET /state` — returns EnvironmentState
  - `GET /health` — returns `{"status": "ok"}`
  - CORS middleware
  - Exception handlers
- **Constraints**: Port configured via env var or default 7860.

### Task config files (`tasks/task_easy.json`, etc.)
- **Path**: `tasks/task_*.json`
- **Purpose**: Static task metadata.
- **Contents**:
  ```json
  {
    "task_id": "easy",
    "title": "Basic Alert Classification",
    "description": "Triage 5 independent alerts with clear root causes.",
    "difficulty": "easy",
    "default_seed": 42,
    "num_alerts": 5,
    "num_incidents": 0,
    "noise_alerts": 0,
    "max_steps": 10
  }
  ```
- **Constraints**: JSON. Consistent with scenario generator behavior.

### Test files
- See Section 13 for individual test file specs.

### `scripts/validate.sh`
- **Path**: `scripts/validate.sh`
- **Purpose**: Local pre-submission validation.
- **Contents**: Bash script that:
  1. Starts the server in background
  2. Curls /reset, checks 200
  3. Curls /step with a triage action, checks 200
  4. Kills server
  5. Runs `openenv validate`
  6. Runs `docker build .`
  7. Prints pass/fail summary

### `scripts/smoke_test.py`
- **Path**: `scripts/smoke_test.py`
- **Purpose**: Python-based end-to-end smoke test.
- **Contents**: Uses httpx to call /reset, /step through a full easy episode, asserts reward > 0 and done == True at end.

---

## 6. DATA MODELS

All models use Pydantic v2 syntax (`from pydantic import BaseModel, Field, field_validator`).

### Alert
```
class Alert(BaseModel):
    alert_id: str              # e.g., "alert-001"
    timestamp: str             # ISO format relative timestamp, e.g., "2024-01-15T10:23:00Z"
    service: str               # e.g., "api-gateway"
    metric: str                # e.g., "cpu_usage_percent"
    metric_value: float        # e.g., 94.5
    threshold: float           # e.g., 80.0
    message: str               # human-readable alert text
    context: str | None = None # optional extra context (recent deploy, dependency info)
    triaged: bool = False      # whether agent has already triaged this
    agent_decision: dict | None = None  # agent's triage result if triaged
```

### Observation
```
class Observation(BaseModel):
    alerts: list[Alert]
    service_map: dict[str, list[str]]   # adjacency list
    pending_count: int
    step_number: int
    max_steps: int
    feedback: str = ""                  # hint after last action
```

### Action
Single model with discriminator:
```
class Action(BaseModel):
    action_type: str   # "triage" | "link_alerts" | "skip"

    # For triage:
    alert_id: str | None = None
    root_cause: str | None = None    # from ROOT_CAUSE_CATEGORIES
    severity: str | None = None      # "critical" | "high" | "medium" | "low"
    remediation: str | None = None   # from REMEDIATION_ACTIONS

    # For link_alerts:
    alert_ids: list[str] | None = None
    incident_label: str | None = None

    # Validators ensure required fields present per action_type
```

**Why single model**: Simpler for LLM to output. Simpler JSON schema. Avoids discriminated union complexity in OpenEnv yaml. The validator approach catches missing fields per action_type at runtime.

### StepResult
```
class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = {}     # can include grader_score at episode end
```

### EnvironmentState
```
class EnvironmentState(BaseModel):
    task_id: str
    seed: int
    step_number: int
    max_steps: int
    done: bool
    alerts: list[Alert]
    ground_truth: list[dict]         # hidden truth per alert
    agent_decisions: list[dict]      # all decisions made
    incidents: list[dict]            # true incident groupings
    cumulative_reward: float
    grader_score: float | None = None  # populated when done
```

### ResetRequest
```
class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42
```

### TaskConfig
```
class TaskConfig(BaseModel):
    task_id: str
    title: str
    description: str
    difficulty: str
    default_seed: int
    num_alerts: int
    num_incidents: int
    noise_alerts: int
    max_steps: int
```

---

## 7. TASK DESIGN

### Task 1: "easy" — Basic Alert Classification
- **Task ID**: `easy`
- **Title**: Basic Alert Classification
- **Difficulty**: Easy
- **Scenario**: 5 independent alerts, each from a different service, with obvious root causes. No incidents (no correlated alerts). No noise.
- **Generation logic**: Pick 5 random services. For each, generate one alert with a clear and obvious root cause. E.g., high CPU on compute service → resource_exhaustion; failed health checks after deploy → deployment_bug.
- **Ambiguity**: None. Alert messages and metrics directly imply the root cause.
- **Max steps**: 10 (generous — only need 5 triage actions)
- **Ground truth**: Each alert has exactly one correct root_cause, severity, and remediation.
- **Grader**:
  - root_cause accuracy: correct / total alerts × 0.4
  - severity accuracy: correct / total × 0.3
  - remediation accuracy: correct / total × 0.3
  - Final = weighted sum, clamped to [0, 1]
- **Partial credit**: Each correct field on each alert contributes independently.
- **Failure cases**: Agent misclassifies obvious alerts, skips real alerts, tries to link non-incident alerts.
- **Expected score for strong LLM**: 0.85–1.0

### Task 2: "medium" — Correlated Incident Response
- **Task ID**: `medium`
- **Title**: Correlated Incident Response
- **Difficulty**: Medium
- **Scenario**: 15 alerts across 10 services. 2 distinct incidents (each incident = 3-4 correlated alerts cascading through dependency chain). 2-3 independent alerts. 1-2 noise/false alarms.
- **Generation logic**: Pick 2 root-cause services. For each, generate cascade: root service alert → dependent service alert → further dependent. Add independent alerts. Add false alarms (metric barely over threshold, known flaky check).
- **Ambiguity**: Moderate. Some cascaded alerts look like independent issues unless agent checks dependency graph. False alarms have borderline metrics.
- **Max steps**: 25
- **Ground truth**: Correct root_cause, severity, remediation per alert + correct incident groupings.
- **Grader**:
  - root_cause accuracy × 0.3
  - severity accuracy × 0.2
  - remediation accuracy × 0.2
  - incident linking F1 score × 0.2 (precision and recall of alert-pairs in same incident)
  - false alarm identification accuracy × 0.1
  - Final = weighted sum, clamped to [0, 1]
- **Partial credit**: Correctly triaging alerts within an incident gives partial even if linking is wrong. Getting severity within 1 level gives half credit.
- **Expected score for strong LLM**: 0.65–0.85

### Task 3: "hard" — Cascading Failure Under Noise
- **Task ID**: `hard`
- **Title**: Cascading Failure Under Noise
- **Difficulty**: Hard
- **Scenario**: 30 alerts across 15 services. 4-5 distinct incidents with deep cascades (3-5 hops). 5-8 noise alerts. Misleading severity hints (monitoring system marks a false alarm as "critical"). Temporal interleaving (alerts from different incidents arrive mixed). One "stealth" incident where the root cause service has a subtle alert (e.g., slow response time, not outright failure) but its dependents are screaming.
- **Generation logic**: Complex multi-incident generation with noise injection. Alerts shuffled by timestamp to break causal ordering.
- **Ambiguity**: High. Multiple alerts per service (some from different incidents). Root cause must be inferred from dependency graph + temporal correlation. Noise alerts distract.
- **Max steps**: 45 (tight — agent must be efficient)
- **Ground truth**: Full truth table.
- **Grader**: Same formula as medium but:
  - Incident linking is weighted more (0.25)
  - Stealth incident detection bonus: +0.05 if root cause of the stealth incident is correctly identified
  - Final = weighted sum, clamped to [0, 1]
- **What makes it genuinely hard**:
  - LLMs struggle with multi-hop causal reasoning across 15 services
  - Noise alerts consume steps and attention
  - Misleading severity requires ignoring the monitoring system's own classification
  - Temporal interleaving breaks the "just read alerts in order" heuristic
  - Tight step budget means brute-forcing costs too much
- **Expected score for strong LLM (GPT-4o)**: 0.40–0.65

---

## 8. REWARD DESIGN

### Per-Action Rewards

**For `triage` actions:**

| Component | Condition | Reward |
|-----------|-----------|--------|
| Root cause | Exact match | +0.30 |
| Root cause | Wrong | +0.00 |
| Severity | Exact match | +0.30 |
| Severity | Within 1 level | +0.10 |
| Severity | Wrong by 2+ | +0.00 |
| Remediation | Exact match | +0.20 |
| Remediation | Wrong | +0.00 |
| Incident link bonus | Alert is in an incident AND agent previously linked it correctly | +0.10 |

Max per triage action: +0.90 (0.80 base + 0.10 link bonus)

**For `link_alerts` actions:**

| Component | Condition | Reward |
|-----------|-----------|--------|
| Correct pair | Two alerts truly in same incident | +0.15 per pair |
| Incorrect pair | Two alerts NOT in same incident | -0.10 per pair |

**For `skip` actions:**

| Component | Condition | Reward |
|-----------|-----------|--------|
| True false alarm | Alert's ground truth root_cause == "false_alarm" | +0.20 |
| Wrong skip | Skipping a real alert | -0.30 |

### Penalties

| Penalty | Condition | Amount |
|---------|-----------|--------|
| Budget pressure | Step > 80% of max_steps | -0.05 per step |
| Invalid action | Malformed action or missing required fields | -0.10 |
| Double triage | Triaging an already-triaged alert | -0.15 |
| Already done | Stepping after episode is done | -0.00 (just return done=True) |

### How Total Reward Differs from Grader Score
- **Cumulative reward** = sum of all per-step rewards. Can exceed 1.0 or be negative.
- **Grader score** = accuracy-based, computed once at episode end, always in [0.0, 1.0].
- They correlate but are not identical. Reward guides the agent during the episode. Grader is the official evaluation metric.

### Example Trajectories

**Example 1: Perfect easy run (5 alerts)**
```
Step 1: triage alert-001, all correct → +0.80
Step 2: triage alert-002, all correct → +0.80
Step 3: triage alert-003, all correct → +0.80
Step 4: triage alert-004, all correct → +0.80
Step 5: triage alert-005, all correct → +0.80
Total reward: 4.00. Grader score: 1.00.
```

**Example 2: Partial medium run**
```
Step 1: triage alert-001, root_cause correct, severity wrong by 1, remediation correct → +0.30 + 0.10 + 0.20 = +0.60
Step 2: triage alert-002, all wrong → +0.00
Step 3: link_alerts [alert-003, alert-004] correctly → +0.15
Step 4: skip alert-010 (true false alarm) → +0.20
Step 5: triage alert-003, root_cause correct, severity correct, remediation wrong → +0.30 + 0.30 + 0.00 = +0.60
... (10 more steps)
Total reward: ~4.2. Grader score: ~0.55.
```

**Example 3: Bad run with penalties**
```
Step 1: invalid action (missing alert_id) → -0.10
Step 2: triage alert-001, all wrong → +0.00
Step 3: triage alert-001 again (double triage) → -0.15
Step 4: skip alert-002 (real critical alert) → -0.30
Step 5-10: (budget pressure kicks in at step 8/10) → -0.05 per step
Total reward: -0.85. Grader score: 0.05.
```

---

## 9. GRADER DESIGN

### Common Formula

```
score = (
    w_rc * root_cause_accuracy +
    w_sev * severity_accuracy +
    w_rem * remediation_accuracy +
    w_link * incident_link_f1 +
    w_fa * false_alarm_accuracy +
    bonus
)
score = clamp(score, 0.0, 1.0)
```

### Per-Task Weights

| Component | Easy | Medium | Hard |
|-----------|------|--------|------|
| root_cause_accuracy (w_rc) | 0.40 | 0.30 | 0.25 |
| severity_accuracy (w_sev) | 0.30 | 0.20 | 0.20 |
| remediation_accuracy (w_rem) | 0.30 | 0.20 | 0.15 |
| incident_link_f1 (w_link) | 0.00 | 0.20 | 0.25 |
| false_alarm_accuracy (w_fa) | 0.00 | 0.10 | 0.10 |
| stealth_bonus | 0.00 | 0.00 | 0.05 |

### Accuracy Calculations

**root_cause_accuracy**: (number of alerts with correct root_cause) / (total alerts). Un-triaged alerts count as wrong.

**severity_accuracy**: For each alert, +1.0 if exact match, +0.5 if within 1 level, +0.0 otherwise. Sum / total alerts.

**remediation_accuracy**: (number correct) / (total alerts).

**incident_link_f1**: Convert agent's incident groups and true incident groups into sets of alert-pairs. Compute precision and recall of pairs. F1 = 2 * P * R / (P + R). If no true incidents, this component = 1.0 (vacuously correct). If true incidents exist but agent made no links, F1 = 0.0.

**false_alarm_accuracy**: (correctly skipped false alarms + correctly triaged real alerts) / (total false alarms + total real alerts among those the agent interacted with). If no false alarms exist, component = 1.0.

**stealth_bonus** (hard only): +0.05 if the root cause service of the "stealth incident" was correctly identified in at least one triage action for that incident.

### Determinism Guarantees
- All inputs are from `EnvironmentState` which is deterministic given seed.
- All calculations use exact arithmetic (no floats that depend on ordering). Use `round(score, 6)` as final step.
- No randomness in grading.

### Anti-Cheating Considerations
- Agent cannot access ground truth via /state during inference (inference.py only calls /reset and /step).
- Grader runs on final state, not on agent-reported data.
- Skipping all alerts gives 0.0 (un-triaged = wrong).
- Spamming the same answer for everything: root_cause accuracy will be ~1/6 (random chance among 6 categories).
- Trying to triage the same alert multiple times: second attempt gets penalty and is ignored for grading.

---

## 10. API DESIGN

### Endpoints

#### `POST /reset`
- **Request body**:
  ```json
  {
    "task_id": "easy",
    "seed": 42
  }
  ```
- **Response** (200):
  ```json
  {
    "observation": {
      "alerts": [
        {
          "alert_id": "alert-001",
          "timestamp": "2024-01-15T10:23:00Z",
          "service": "api-gateway",
          "metric": "error_rate_percent",
          "metric_value": 15.2,
          "threshold": 5.0,
          "message": "Error rate spike on api-gateway: 15.2% (threshold: 5%)",
          "context": "Deploy v2.3.1 rolled out 10 minutes ago",
          "triaged": false,
          "agent_decision": null
        }
      ],
      "service_map": {
        "web-frontend": ["api-gateway"],
        "api-gateway": ["user-service", "order-service"],
        "user-service": ["postgres-primary"],
        "order-service": ["postgres-primary", "redis-cache"]
      },
      "pending_count": 5,
      "step_number": 0,
      "max_steps": 10,
      "feedback": ""
    }
  }
  ```
- **Errors**: 422 if invalid task_id.

#### `POST /step`
- **Request body**:
  ```json
  {
    "action_type": "triage",
    "alert_id": "alert-001",
    "root_cause": "deployment_bug",
    "severity": "high",
    "remediation": "rollback_deploy"
  }
  ```
- **Response** (200):
  ```json
  {
    "observation": { "..." },
    "reward": 0.80,
    "done": false,
    "info": {}
  }
  ```
- When `done` is true, `info` includes `{"grader_score": 0.85}`.
- **Errors**: 400 if no active episode. 422 if invalid action format.

#### `GET /state`
- **Response** (200): Full `EnvironmentState` JSON including ground truth.
- **Purpose**: Debugging and validation only.

#### `GET /health`
- **Response** (200): `{"status": "ok"}`

### Session Handling
- Single global environment instance.
- Calling `/reset` starts a new episode, discarding any in-progress episode.
- No authentication. No session tokens.
- This is appropriate for a hackathon. Production would need per-session instances.

---

## 11. OPENENV.YAML PLAN

**VERIFY MANUALLY**: The exact required fields may depend on the version of `openenv-core`. Install it and run `openenv validate` early. The following is the best-effort plan based on known OpenEnv conventions.

```yaml
name: cloud-alert-triage
version: "1.0.0"
description: >
  An SRE alert triage environment where an AI agent must classify,
  prioritize, and remediate cloud infrastructure monitoring alerts
  across a microservice dependency graph.
author: Better Call Coders
tags:
  - openenv
  - sre
  - devops
  - incident-response

tasks:
  - id: easy
    title: Basic Alert Classification
    description: Triage 5 independent alerts with clear root causes.
    difficulty: easy
  - id: medium
    title: Correlated Incident Response
    description: Triage 15 alerts including 2 correlated incidents and false alarms.
    difficulty: medium
  - id: hard
    title: Cascading Failure Under Noise
    description: Triage 30 alerts with 5 cascading incidents, noise, and misleading severity.
    difficulty: hard

action_space:
  type: object
  description: >
    JSON object with action_type (triage/link_alerts/skip) and
    corresponding fields. See README for full schema.

observation_space:
  type: object
  description: >
    JSON object containing alerts list, service dependency map,
    pending count, step number, max steps, and feedback string.

endpoints:
  reset: /reset
  step: /step
  state: /state
```

**Common validation mistakes to avoid**:
- Missing `tasks` list
- Wrong endpoint paths
- YAML syntax errors (watch indentation)
- Using `tab` characters (YAML requires spaces)

---

## 12. INFERENCE.PY PLAN

### Runtime Flow
```
1. Read env vars: API_BASE_URL, MODEL_NAME, OPENAI_API_KEY (or HF_TOKEN)
2. Create OpenAI client with base_url=API_BASE_URL, api_key=OPENAI_API_KEY
3. Set ENV_URL from env var or default to HF Space URL
4. For each task_id in ["easy", "medium", "hard"]:
   a. POST /reset with {task_id, seed: 42}
   b. Parse observation
   c. log_start(task_id, ...)
   d. Loop up to max_steps:
      i.   Format observation into compact prompt
      ii.  Call LLM with prompt → get text response
      iii. Parse text into Action JSON (with fallback)
      iv.  POST /step with action
      v.   Parse response (observation, reward, done, info)
      vi.  log_step(...)
      vii. If done: break
   e. Compute final score
   f. log_end(...)
5. Exit
```

### Prompt Strategy

System prompt (sent once per task):
```
You are an SRE triaging infrastructure alerts. For each alert, respond with a JSON action.
Valid action_types: triage, link_alerts, skip.
For triage: include alert_id, root_cause (one of: resource_exhaustion, network_failure, deployment_bug, config_error, dependency_outage, false_alarm), severity (critical/high/medium/low), remediation (restart_service/scale_up/rollback_deploy/fix_config/escalate_to_team/acknowledge_and_monitor/dismiss).
For link_alerts: include alert_ids (list) and incident_label (string).
For skip: include alert_id.
Respond ONLY with valid JSON. No explanation.
```

User prompt (per step):
```
Pending alerts: {pending_count}
Step {step_number}/{max_steps}
Last feedback: {feedback}

Untriaged alerts:
{compact alert summaries — only untriaged ones, max ~200 tokens each}

Service dependencies:
{compact adjacency list}

Respond with ONE action as JSON.
```

### Token Efficiency
- Only include untriaged alerts in prompt (not already-triaged ones)
- Truncate alert context if over 100 chars
- Omit service_map after first prompt if unchanged (or always include — it's small)
- Use the shortest possible field names in examples

### Action Parsing
```python
def parse_action(text: str) -> dict:
    # Strip markdown code fences if present
    text = text.strip().strip("```json").strip("```").strip()
    try:
        action = json.loads(text)
        if "action_type" in action:
            return action
    except json.JSONDecodeError:
        pass
    # Fallback: skip the first untriaged alert
    return {"action_type": "skip", "alert_id": first_untriaged_id}
```

### Timeout Management
- Total budget: 20 minutes
- Per-task budget: 6 minutes (18 min for 3 tasks, 2 min buffer)
- Per-step LLM call timeout: 30 seconds
- If per-task budget exceeded, break and move to next task

### Exact Logging Schema

```
[START] {"task": "easy", "env": "cloud-alert-triage", "model": "gpt-4o-mini"}
[STEP] {"step": 1, "action": {"action_type": "triage", "alert_id": "alert-001", ...}, "reward": 0.80, "done": false, "error": null}
[STEP] {"step": 2, "action": {"action_type": "triage", "alert_id": "alert-002", ...}, "reward": 0.60, "done": false, "error": null}
...
[END] {"success": true, "steps": 5, "score": 0.92, "rewards": [0.80, 0.60, 0.80, 0.80, 0.60]}
```

All log lines are valid JSON after the tag prefix. `score` is clamped to [0.0, 1.0].

### Reproducibility
- Use `seed=42` for all tasks in the default run.
- LLM temperature=0 (or as low as possible) for reproducibility.
- Document in README: "Baseline scores recorded with MODEL_NAME=gpt-4o-mini, seed=42, temperature=0".

---

## 13. TESTING STRATEGY

### test_models.py
- **Purpose**: Validate Pydantic models accept valid data and reject invalid data.
- **Tests**:
  - Create Alert with all fields → succeeds
  - Create Alert with missing required field → ValidationError
  - Create Action with action_type="triage" but missing alert_id → ValidationError (if validator enforces)
  - Create Action with invalid root_cause → ValidationError
  - Serialize Observation to JSON and back → equal
- **Bugs caught**: Typos in field names, wrong types, missing validators.

### test_scenario_gen.py
- **Purpose**: Verify deterministic scenario generation.
- **Tests**:
  - generate_scenario("easy", 42) twice → identical output
  - generate_scenario("easy", 42) vs generate_scenario("easy", 99) → different output
  - Easy scenario has exactly 5 alerts
  - Medium scenario has 15 alerts
  - Hard scenario has 30 alerts
  - All alerts have valid services (exist in service graph)
  - All ground truth root_causes are valid enum values
- **Bugs caught**: Non-determinism, wrong alert counts, invalid data generation.

### test_environment.py
- **Purpose**: Core logic integration.
- **Tests**:
  - reset() → observation with correct pending_count
  - step(valid triage) → reward > 0, pending_count decreases
  - step(triage already-triaged alert) → penalty
  - step after done → done remains True
  - Full episode (all alerts triaged) → done == True and info contains grader_score
  - state() returns EnvironmentState with ground_truth populated
- **Bugs caught**: State mutation bugs, wrong done condition, missing grader call.

### test_rewards.py
- **Purpose**: Reward calculations are correct.
- **Tests**:
  - Perfect triage → 0.80 reward
  - Severity off by 1 → 0.10 instead of 0.30
  - Skip a false alarm → +0.20
  - Skip a real alert → -0.30
  - Step > 80% budget → penalty applied
  - Invalid action → -0.10
- **Bugs caught**: Wrong reward values, missing penalties, off-by-one in budget.

### test_graders.py
- **Purpose**: Grader determinism and correctness.
- **Tests**:
  - Perfect run → 1.0
  - All wrong → close to 0.0
  - Same input twice → same output
  - Partial run (some triaged, some not) → score between 0 and 1
  - Score is always in [0.0, 1.0]
- **Bugs caught**: Non-determinism, division by zero, score outside range.

### test_api.py
- **Purpose**: HTTP endpoint integration.
- **Setup**: Use FastAPI TestClient (httpx-based, no actual server needed).
- **Tests**:
  - POST /reset → 200, returns valid Observation JSON
  - POST /step with valid action → 200, returns StepResult JSON
  - POST /step without prior reset → 400
  - POST /step with invalid action → 422
  - GET /state → 200
  - GET /health → 200
- **Bugs caught**: Routing errors, serialization bugs, missing error handlers.

### Docker smoke test (manual)
```bash
docker build -t test-env .
docker run -d -p 7860:7860 --name test-env test-env
sleep 3
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy","seed":42}'
# Must print 200
docker stop test-env && docker rm test-env
```

---

## 14. DEBUGGING PLAYBOOK

### Bug: `/reset` returns 500
- **Symptom**: Server starts but /reset crashes.
- **Look at**: Server logs for traceback. Usually a Pydantic serialization error.
- **Common cause**: Model field has a type that's not JSON-serializable (e.g., `set` instead of `list`).
- **Fix**: Use `model_dump(mode="json")` or ensure all fields are JSON-compatible types.

### Bug: Non-deterministic scenario generation
- **Symptom**: Same seed produces different alerts.
- **Look at**: `scenario_generator.py`. Check if global `random` is used anywhere instead of the seeded `rng` instance.
- **Fix**: Pass `rng` instance to every helper function. Never use `random.random()` directly.

### Bug: Reward doesn't accumulate correctly
- **Symptom**: Total reward at end doesn't match sum of per-step rewards.
- **Look at**: `environment.py` — where is cumulative_reward updated?
- **Fix**: Ensure `self._cumulative_reward += reward` happens in `step()` after reward is computed.

### Bug: `openenv validate` fails
- **Symptom**: Validation script reports errors.
- **Look at**: `openenv.yaml` for typos, wrong indentation, missing fields.
- **Fix**: Install `openenv-core`, run `openenv validate`, read the error message carefully. **VERIFY MANUALLY** what fields the current version requires.

### Bug: Docker build fails
- **Symptom**: `docker build .` errors out.
- **Common causes**: Missing system dependency, wrong Python version, `requirements.txt` has broken package name.
- **Fix**: Check error log. Use `python:3.11-slim` base. Verify all package names with `pip install`.

### Bug: inference.py can't connect to environment
- **Symptom**: Connection refused or timeout.
- **Look at**: Is the server running? Is the port correct? Is `ENV_URL` env var set?
- **Fix**: Start server first. Use `http://localhost:7860` for local testing. Use HF Space URL for remote.

### Bug: LLM returns unparseable action
- **Symptom**: JSON decode error in inference.py.
- **Look at**: Raw LLM output. Often contains markdown fences or explanation text.
- **Fix**: Strip markdown fences. Use regex to extract JSON object. Fall back to skip action.

### Bug: Grader score outside [0, 1]
- **Symptom**: Score is negative or > 1.
- **Look at**: `grading.py`. Check that weights sum to ≤ 1.0. Check clamping logic.
- **Fix**: Add `score = max(0.0, min(1.0, score))` as final line.

### Bug: HF Space won't start
- **Symptom**: Space shows "Runtime error" or never finishes building.
- **Look at**: HF build logs.
- **Common causes**: Port not 7860, CMD wrong, dependency install fails.
- **Fix**: Match Dockerfile port to 7860. Test locally with Docker first.

---

## 15. HUMAN INPUT CHECKPOINTS

### Checkpoint 1: HF Token and Space Setup
- **What**: Create Hugging Face account, create a Docker-type Space, get HF_TOKEN.
- **When**: Before Phase 13 (HF Deployment). Can be done in parallel any time.
- **Format**: Provide `HF_TOKEN=hf_xxxxx` and Space URL `https://huggingface.co/spaces/USERNAME/cloud-alert-triage`.
- **If not provided**: Agent can complete all phases 1-12 without this. Block only at Phase 13.

### Checkpoint 2: LLM API Key and Endpoint
- **What**: API key for whatever LLM service will be used (OpenAI, or whatever the hackathon provides).
- **When**: Before Phase 9 (inference.py testing). Can use mock/stub until then.
- **Format**: `API_BASE_URL=https://...` and `OPENAI_API_KEY=sk-...` and `MODEL_NAME=gpt-4o-mini`.
- **If not provided**: Agent can write inference.py but can't test it end-to-end. Can still test with a mock server.

### Checkpoint 3: Repository Push
- **What**: Human must push code to HF Space git repo (or set up CI).
- **When**: Phase 13.
- **Format**: `git remote add space https://huggingface.co/spaces/USERNAME/cloud-alert-triage` then `git push space main`.
- **If not provided**: Agent prepares all code locally. Human does the push.

### Checkpoint 4: Final Verification URL
- **What**: Human confirms the live Space URL and that it responds.
- **When**: After Phase 13.
- **Format**: "Space is live at https://USERNAME-cloud-alert-triage.hf.space"

### Checkpoint 5: Naming/Branding (Optional)
- **What**: Team name, project name, any custom branding for README.
- **When**: Phase 14 (Final Polish). Non-blocking.
- **Default if not provided**: Use "cloud-alert-triage" and "Better Call Coders".

---

## 16. TOKEN-EFFICIENCY AND MULTI-AGENT HANDOFF PROTOCOL

### Protocol Rules

1. **FIRST ACTION for any new agent**: Read `MAP.md` then `PROGRESS.md`. Do not read any source files until you know what exists and what phase the project is in.

2. **Before editing any file**: Read only that specific file. Do not read the whole repo.

3. **After completing any phase**:
   - Update `PROGRESS.md` with the standard change log entry (see template below)
   - Update `MAP.md` only if new files were created or architecture changed

4. **If unsure about a design decision**: Check `docs/decision_log.md` first. If not covered, make the simplest safe choice, document it in `decision_log.md`, and mark it in `PROGRESS.md` as an assumption.

5. **Do not rewrite files unnecessarily**. Use targeted edits (str_replace) instead of recreating entire files.

6. **Do not re-read files you just wrote**. Trust your own output within the same session.

7. **Request human input** by writing a clearly marked block in `PROGRESS.md`:
   ```
   ### HUMAN INPUT NEEDED
   - **What**: [specific request]
   - **Blocking**: [phase N] / [non-blocking]
   - **Default if not provided**: [fallback]
   ```

### Standard PROGRESS.md Update Template
```
### [YYYY-MM-DD HH:MM] — [Agent/Human ID]
- **Phase**: [N — Phase Name]
- **Status**: COMPLETE / IN PROGRESS / BLOCKED
- **Files changed**: server/models.py, server/config.py
- **What was done**: Implemented all Pydantic models per Section 6 of master plan. Added field validators for enum types.
- **Tests passing**: test_models.py — 8/8 green
- **What's next**: Phase 3 — Scenario Generator
- **Blockers**: None
- **Assumptions**: Used Pydantic v2 syntax. Assumed openenv-core supports Pydantic v2.
```

### Recommended Agent Summary Format (for chat output)
```
✅ Phase N complete.
Files: [list]
Tests: [X/Y passing]
Next: Phase [N+1] — [Name]
Blockers: [None / description]
```

---

## 17. IMPLEMENTATION EXECUTION CHECKLIST

Copy this into `PROGRESS.md`:

```markdown
## MASTER CHECKLIST

### Architecture & Planning
- [ ] Master plan reviewed and understood
- [ ] Domain confirmed: Cloud Alert Triage
- [ ] MAP.md created
- [ ] PROGRESS.md created

### Phase 1: Bootstrap
- [ ] Directory structure created
- [ ] requirements.txt written
- [ ] .env.example written
- [ ] Makefile written
- [ ] All __init__.py files created

### Phase 2: Data Models
- [ ] server/models.py implemented
- [ ] server/config.py implemented
- [ ] test_models.py passes

### Phase 3: Scenario Generator
- [ ] server/service_graph.py implemented
- [ ] server/scenario_generator.py implemented
- [ ] Task config JSONs written
- [ ] test_scenario_gen.py passes
- [ ] Determinism verified (same seed → same output)

### Phase 4: Environment Core
- [ ] server/environment.py implemented
- [ ] reset() works
- [ ] step() works
- [ ] state() works
- [ ] test_environment.py passes

### Phase 5: Rewards
- [ ] server/rewards.py implemented
- [ ] test_rewards.py passes
- [ ] Reward examples from plan verified

### Phase 6: Graders
- [ ] server/grading.py implemented
- [ ] test_graders.py passes
- [ ] Scores always in [0.0, 1.0]
- [ ] Determinism verified

### Phase 7: API Server
- [ ] server/app.py implemented
- [ ] /reset endpoint works
- [ ] /step endpoint works
- [ ] /state endpoint works
- [ ] /health endpoint works
- [ ] test_api.py passes

### Phase 8: OpenEnv Metadata
- [ ] openenv.yaml written
- [ ] openenv validate passes (VERIFY MANUALLY)

### Phase 9: Inference Script
- [ ] inference.py implemented
- [ ] Reads env vars correctly
- [ ] Runs all 3 tasks
- [ ] [START]/[STEP]/[END] logs correct
- [ ] Completes under 20 minutes
- [ ] Baseline scores recorded

### Phase 10: Tests
- [ ] All test files written
- [ ] pytest tests/ -v all green

### Phase 11: Docker
- [ ] Dockerfile written
- [ ] docker build succeeds
- [ ] docker run starts server
- [ ] /reset returns 200 from Docker container

### Phase 12: Local Validation
- [ ] scripts/validate.sh passes
- [ ] scripts/smoke_test.py passes

### Phase 13: HF Deployment
- [ ] HF Space created (HUMAN)
- [ ] Code pushed to Space
- [ ] Space builds successfully
- [ ] /reset returns 200 on live URL

### Phase 14: Final Polish
- [ ] README.md complete with all required sections
- [ ] Baseline scores in README
- [ ] Code cleaned up
- [ ] decision_log.md updated
- [ ] Final inference.py run against live Space
- [ ] All checklist items green
```

---

## 18. RISKS AND MITIGATIONS

| # | Risk | Severity | Likelihood | Mitigation |
|---|------|----------|-----------|------------|
| 1 | `openenv validate` fails due to unknown schema requirements | HIGH | MEDIUM | Install openenv-core early. Run validate after Phase 8 immediately. Read source code of openenv-core if docs are lacking. Adapt yaml. |
| 2 | HF Space doesn't build or boot | HIGH | MEDIUM | Test Docker locally first (Phase 11). Use `python:3.11-slim`. Keep dependencies minimal. Check HF build logs. |
| 3 | inference.py exceeds 20-minute timeout | MEDIUM | MEDIUM | Set per-task timeout of 6 min. Use fast model (gpt-4o-mini). Limit max_steps. Skip remaining tasks if time is low. |
| 4 | LLM returns unparseable actions frequently | MEDIUM | HIGH | Robust parsing with fallback. Strip markdown fences. Use structured prompts. Fall back to skip action. |
| 5 | Non-deterministic grading | HIGH | LOW | Use seeded RNG everywhere. Sort outputs. Test determinism explicitly. Round scores. |
| 6 | Reward hacking (agent finds shortcut) | LOW | LOW | Anti-cheating grader design. Penalize repeated actions. Grade on accuracy not reward total. |
| 7 | Judges can't reproduce baseline scores | MEDIUM | MEDIUM | Document exact env vars, model, seed, temperature. Use deterministic seed=42. |
| 8 | Missing or wrong openenv.yaml fields | HIGH | MEDIUM | Verify manually. Run validate. Check openenv-core source if needed. |
| 9 | Deadline missed due to HF deployment issues | HIGH | LOW | Start HF Space creation early (parallel to coding). Have human on standby for Phase 13. |
| 10 | Environment too simple / boring for judges | MEDIUM | LOW | Hard task is genuinely hard. Incident cascading is interesting. README sells the value. |

### Minimum Viable Submission Path
If time is critically short, prioritize:
1. Phases 1-4 (bootstrap, models, scenario gen, env core) — functional /reset and /step
2. Phase 7 (API server) — endpoints work
3. Phase 8 (openenv.yaml) — validate passes
4. Phase 11 (Docker) — builds
5. Phase 13 (HF deploy) — live

Skip if needed: sophisticated rewards (use simple per-field accuracy as reward), complex hard task (use medium with more alerts), inference.py can use hardcoded actions as fallback.

### Best Possible Polished Submission Path
All 14 phases complete. Rich hard task with cascading incidents. Polished README with architecture diagram. All tests green. Baseline scores documented. Clean code with comments.

---

## 19. FINAL RECOMMENDED BUILD PROMPTS

### Prompt 1: Bootstrap

```
TASK: Create the repository skeleton for the cloud-alert-triage OpenEnv project.

Read the master plan's Section 3 (Repository Structure).

Create ALL directories and files listed. Files should be empty or minimal stubs except:
- PROGRESS.md: use the exact template from the master plan
- MAP.md: use the exact template from the master plan
- requirements.txt: fastapi>=0.100.0, uvicorn[standard]>=0.23.0, pydantic>=2.0.0, openai>=1.0.0, httpx>=0.24.0
- .env.example: API_BASE_URL=, MODEL_NAME=, OPENAI_API_KEY=, HF_TOKEN=, ENV_URL=http://localhost:7860
- Makefile: targets for run, test, docker-build, docker-run, validate, smoke, infer
- All __init__.py files: empty
- server/config.py: Define ROOT_CAUSE_CATEGORIES = ["resource_exhaustion", "network_failure", "deployment_bug", "config_error", "dependency_outage", "false_alarm"], SEVERITY_LEVELS = ["critical", "high", "medium", "low"], REMEDIATION_ACTIONS = ["restart_service", "scale_up", "rollback_deploy", "fix_config", "escalate_to_team", "acknowledge_and_monitor", "dismiss"], ACTION_TYPES = ["triage", "link_alerts", "skip"], MAX_STEPS_BY_TASK = {"easy": 10, "medium": 25, "hard": 45}, SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

After creating all files, update PROGRESS.md to mark Phase 1 COMPLETE.
Verify: `ls -R` shows correct structure.
```

### Prompt 2: Core Models

```
TASK: Implement all Pydantic models in server/models.py.

Read MAP.md and PROGRESS.md first.
Read the master plan's Section 6 (Data Models).

Implement these models using Pydantic v2:
- Alert, Observation, Action (with validators per action_type), StepResult, EnvironmentState, ResetRequest, TaskConfig

Import enums/constants from server/config.py.

Action validators:
- If action_type == "triage": alert_id, root_cause, severity, remediation must all be non-None
- If action_type == "link_alerts": alert_ids must be non-None and len >= 2, incident_label must be non-None
- If action_type == "skip": alert_id must be non-None
- root_cause must be in ROOT_CAUSE_CATEGORIES if provided
- severity must be in SEVERITY_LEVELS if provided
- remediation must be in REMEDIATION_ACTIONS if provided

Write tests/test_models.py: test valid creation, test invalid data raises ValidationError, test JSON round-trip.

Update PROGRESS.md after completion.
```

### Prompt 3: Scenario Generator

```
TASK: Implement the service graph and deterministic scenario generator.

Read MAP.md and PROGRESS.md first.
Read the master plan's Sections 2, 7 for environment concept and task design.

server/service_graph.py:
- Define SERVICE_GRAPH dict with 15-20 microservices in a realistic dependency hierarchy:
  web-frontend → api-gateway → [user-service, order-service, search-service, notification-service]
  user-service → [postgres-primary, redis-cache]
  order-service → [postgres-primary, payment-gateway, inventory-service]
  ... etc. Make it realistic.
- Functions: get_service_names(), get_dependencies(service), get_dependents(service), get_graph_as_adjacency_list()

server/scenario_generator.py:
- generate_scenario(task_id, seed) → returns a dict with keys: alerts (list), ground_truth (list), incidents (list)
- Use random.Random(seed) for ALL randomness. Never use global random.
- Easy: 5 independent alerts, obvious root causes, no incidents
- Medium: 15 alerts, 2 incidents (3-4 alerts each cascading through deps), 2 independent, 1-2 false alarms
- Hard: 30 alerts, 4-5 incidents (some deep cascades), 5-8 noise/false alarms, one stealth incident, shuffled timestamps

Each alert must have realistic message text that hints at root cause.
Each ground_truth entry must have: alert_id, true_root_cause, true_severity, true_remediation, incident_id (or null).

Write tasks/task_easy.json, task_medium.json, task_hard.json with metadata.
Write tests/test_scenario_gen.py: determinism test, correct counts, valid fields.

Update PROGRESS.md after completion.
```

### Prompt 4: Environment Core + Rewards + Graders

```
TASK: Implement the core environment, reward system, and graders.

Read MAP.md and PROGRESS.md first.
Read the master plan's Sections 4, 5, 6, 8, 9.

server/environment.py:
- class AlertTriageEnv with reset(task_id, seed), step(action), state()
- reset: call scenario_generator, store alerts + ground_truth + incidents, reset step counter, return Observation
- step: validate action, apply action to state, compute reward via rewards.py, check done condition, return StepResult. When done, include grader_score in info.
- state: return full EnvironmentState

server/rewards.py:
- compute_reward(action_dict, ground_truth_list, env_state_dict) → float
- Implement exact reward table from master plan Section 8
- Handle all action types and penalties

server/grading.py:
- grade_episode(task_id, final_state_dict) → float in [0.0, 1.0]
- Use per-task weights from master plan Section 9
- Implement accuracy calculations, incident F1, false alarm accuracy
- Always clamp to [0.0, 1.0]

Write tests for all three: test_environment.py, test_rewards.py, test_graders.py.

Update PROGRESS.md after completion.
```

### Prompt 5: API Server

```
TASK: Implement the FastAPI HTTP server.

Read MAP.md and PROGRESS.md first.
Read the master plan's Section 10 (API Design).

server/app.py:
- Create FastAPI app
- Instantiate one global AlertTriageEnv
- POST /reset: accept ResetRequest body, call env.reset(), return {"observation": ...}
- POST /step: accept Action body, call env.step(), return StepResult as JSON
- GET /state: call env.state(), return EnvironmentState as JSON
- GET /health: return {"status": "ok"}
- Add CORSMiddleware(allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
- Add exception handlers for 400 (no active episode) and 422 (validation)

Write tests/test_api.py using FastAPI TestClient.

Test manually: start server with `uvicorn server.app:app --port 7860`, curl /reset, /step, /state.

Update PROGRESS.md after completion.
```

### Prompt 6: Inference Script

```
TASK: Implement inference.py baseline agent.

Read MAP.md and PROGRESS.md first.
Read the master plan's Section 12 (inference.py Plan).

inference.py in project root:
- Use `from openai import OpenAI`
- Read env vars: API_BASE_URL, MODEL_NAME, OPENAI_API_KEY (fall back to HF_TOKEN)
- Read ENV_URL env var (default http://localhost:7860)
- For each task in ["easy", "medium", "hard"]:
  - POST to {ENV_URL}/reset
  - Loop: format observation → call LLM → parse action → POST to {ENV_URL}/step
  - Print [START], [STEP], [END] logs in EXACT format from master plan
- Robust action parsing with fallback to skip
- Per-task timeout of 6 minutes
- LLM temperature=0
- Total timeout: 20 minutes

The [START], [STEP], [END] log format must be: tag followed by space followed by JSON on one line.
Example: [START] {"task": "easy", "env": "cloud-alert-triage", "model": "gpt-4o-mini"}

Update PROGRESS.md after completion.
```

### Prompt 7: Docker + OpenEnv YAML

```
TASK: Create Dockerfile and openenv.yaml.

Read MAP.md and PROGRESS.md first.
Read the master plan's Sections 4 (Phase 11) and 11.

Dockerfile:
- FROM python:3.11-slim
- WORKDIR /app
- COPY requirements.txt .
- RUN pip install --no-cache-dir -r requirements.txt
- COPY . .
- EXPOSE 7860
- CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

openenv.yaml:
- Use exact content from master plan Section 11
- Run `pip install openenv-core` and `openenv validate` to check

Test: docker build -t cloud-alert-triage . && docker run -p 7860:7860 cloud-alert-triage
Then curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"easy","seed":42}'
Must return 200.

Update PROGRESS.md after completion.
```

### Prompt 8: Validation + Bugfix

```
TASK: Run full validation suite and fix any issues.

Read MAP.md and PROGRESS.md first.

Run in order:
1. pytest tests/ -v → fix any failures
2. openenv validate → fix any yaml issues
3. docker build . → fix any build issues
4. docker run + curl /reset → fix any runtime issues
5. python inference.py (with local server running) → fix any inference issues
6. Check all [START]/[STEP]/[END] logs are correct format

For each failure: identify root cause, fix, re-run, confirm fix.

Update PROGRESS.md with all fixes applied.
```

### Prompt 9: Final Polish

```
TASK: Write README.md and do final cleanup.

Read MAP.md and PROGRESS.md first.
Read the master plan's Section 5 (README spec).

README.md must include:
- Title: Cloud Alert Triage — OpenEnv Environment
- Badge: openenv compatible
- One-paragraph description and motivation (SRE alert triage is critical, costs millions in downtime, etc.)
- Architecture section with text diagram
- Action Space table (action_type, fields, types, meaning)
- Observation Space table (field, type, meaning)
- Task Descriptions table (ID, title, difficulty, description, expected score)
- Reward Design summary
- Setup: pip install, run locally, Docker, HF Space URL
- Baseline Scores table (task, model, score, steps)
- Team credit

Make README judge-friendly: scannable, uses tables, has clear structure, sells the project.

Final cleanup:
- Remove any debug prints
- Ensure all files have docstrings
- Update PROGRESS.md: mark all phases COMPLETE
- Verify master checklist all green

Update PROGRESS.md with final status.
```

---

## 20. OUTPUT FORMAT REQUIREMENTS

This document follows all requested formatting:
- Clear numbered sections with headings
- Tables where data is tabular
- Bullet points for lists
- Numbered steps for sequences
- Explicit decisions made (not "it depends")
- Concrete examples for abstract concepts
- Uncertainties marked with **VERIFY MANUALLY**
- Designed for downstream AI agent execution with minimal ambiguity

**END OF MASTER PLAN**