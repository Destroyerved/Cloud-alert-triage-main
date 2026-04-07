# REPOSITORY MAP

## Architecture Overview
Cloud Alert Triage OpenEnv environment. FastAPI server exposes /reset, /step, /state.
Agent (inference.py) connects over HTTP and calls an OpenAI-compatible LLM to triage alerts.

## File Purposes

| File | Purpose | Depends On |
|------|---------|-----------|
| server/app.py | FastAPI routes | models.py, environment.py |
| server/models.py | All Pydantic schemas (v2) | config.py |
| server/environment.py | Core env logic (reset/step/state) | models.py, scenario_generator.py, rewards.py, grading.py |
| server/scenario_generator.py | Deterministic alert/incident generation; generate_scenario(task_id, seed) → dict | models.py, service_graph.py, config.py |
| server/rewards.py | Computes per-step rewards | models.py, config.py |
| server/grading.py | End-of-episode task graders (0-1 scores) | models.py, config.py |
| server/service_graph.py | Service dependency DAG (17 microservices, 5 tiers); helpers: get_service_names, get_dependencies, get_dependents, get_graph_as_adjacency_list | — |
| server/config.py | Constants, enums, task step limits | — |
| inference.py | Baseline LLM agent (root level) | openai SDK, httpx, ENV_URL |
| openenv.yaml | OpenEnv metadata and endpoint declarations | — |
| Dockerfile | Container config for HF Spaces (port 7860) | requirements.txt |
| tasks/task_easy.json | Easy task metadata + grader weights (5 alerts, 0 incidents) | — |
| tasks/task_medium.json | Medium task metadata + grader weights (15 alerts, 2 incidents) | — |
| tasks/task_hard.json | Hard task metadata + grader weights (30 alerts, 5 incidents) | — |
| tests/test_models.py | Pydantic model validation tests | server/models.py |
| tests/test_environment.py | Core env logic integration tests | server/environment.py |
| tests/test_rewards.py | Reward calculation unit tests | server/rewards.py |
| tests/test_graders.py | Grader determinism and correctness tests | server/grading.py |
| tests/test_api.py | FastAPI endpoint integration tests (TestClient) | server/app.py |
| tests/test_scenario_gen.py | Scenario generation determinism tests (25 tests, now active) | server/scenario_generator.py |
| scripts/validate.sh | Bash pre-submission validation script | server running, openenv CLI, docker |
| scripts/smoke_test.py | Python end-to-end smoke test (httpx) | server running |
| docs/decision_log.md | Design decisions and rationale log | — |

## Key Design Decisions
- Single `Action` model with `action_type` discriminator (not multiple models)
- Deterministic scenario generation via `random.Random(seed)` — never global random
- All rng.sample/shuffle calls receive pre-sorted list inputs for cross-version stability
- Alert IDs assigned via sequential counter (not rng) for guaranteed uniqueness
- Rewards are per-step floats; grader score is end-of-episode float in [0, 1]
- One global `AlertTriageEnv` instance (not per-session) for simplicity
- FastAPI with uvicorn; port 7860 (HF Spaces default)
- `grader_score` injected into `info` dict only when `done == True`
- service_graph.py uses a pre-computed reverse index for O(1) get_dependents() lookups
- All graph outputs sorted alphabetically for determinism across Python versions
- Hard scenario: alerts list is shuffled by a deterministic rng permutation; ground_truth is NOT shuffled
- **Dynamic cascade**: after step 5, untriaged critical/high original alerts spawn one
  deterministic dependent alert on a downstream service.  Dynamic alerts (prefix `dyn-`)
  participate in per-step rewards but are **excluded** from the final grader score.
  Tracking state: `_dynamic_alert_ids`, `_spawned_from`, `_original_alert_ids` in env.

## Scenario Generator Design (scenario_generator.py)

### Output structure
```
{
  "alerts":       [alert dicts],      # matches Alert Pydantic model
  "ground_truth": [gt dicts],         # one per alert; not shown to agent
  "incidents":    [incident dicts],   # 0/2/5 entries depending on task
}
```

### Alert counts
| Task   | Incident alerts | Independent | False alarms | Total |
|--------|----------------|-------------|--------------|-------|
| easy   | 0              | 5           | 0            | 5     |
| medium | 4+4=8          | 5           | 2            | 15    |
| hard   | 4+4+4+3+3=18   | 6           | 6            | 30    |

### Medium incidents
- INC-001: redis-cache (resource_exhaustion) → auth-service, recommendation-engine, user-service
- INC-002: object-storage (network_failure) → email-worker, notification-service, api-gateway

### Hard incidents
- INC-001: postgres-primary (resource_exhaustion) → auth-service, inventory-service, order-service
- INC-002: elasticsearch (network_failure) → recommendation-engine, search-service, api-gateway
- INC-003: kafka-broker (config_error) → notification-service, order-service, api-gateway
- INC-004: object-storage (deployment_bug) → email-worker, notification-service
- INC-005: redis-cache (stealth, resource_exhaustion subtle) → auth-service, recommendation-engine

## Service Graph (17 nodes, 5 tiers)
```
Tier 1  web-frontend
Tier 2  api-gateway
Tier 3  auth-service, user-service, order-service, search-service, notification-service
Tier 4  payment-gateway, inventory-service, recommendation-engine, email-worker, sms-worker
Tier 5  postgres-primary, redis-cache, kafka-broker, elasticsearch, object-storage
```

## Data Flow
```
reset(task_id, seed)
  → scenario_generator.generate_scenario(task_id, seed)
  → env stores alerts, ground_truth, incidents
  → env records _original_alert_ids (for cascade eligibility)
  → returns Observation

step(action)
  → env validates action (Pydantic)
  → env updates state (marks alert triaged)
  → rewards.compute_reward(action, ground_truth, state) → float
  → _maybe_spawn_cascade_alerts() — spawns dynamic alerts after step 5
  → checks done condition
  → if done: grading.grade_episode(task_id, state) → grader_score
      (grader filters out dynamic_alert_ids from ground_truth)
  → returns StepResult(observation, reward, done, info)

state()
  → returns full EnvironmentState (includes ground_truth — for /state debugging only)
```

## Current Phase
Phase 1 (Bootstrap) COMPLETE.
Phase 2 (Data Models) COMPLETE.
Phase 3 (Scenario Gen) ✅ VERIFIED COMPLETE.
Phase 4 (Environment Core) ✅ COMPLETE.
Phase 5 (Rewards) ✅ COMPLETE. rewards.py fully implemented (42 tests).
Phase 6 (Graders) ✅ COMPLETE. grading.py fully implemented.
Phase 7 (API Server) ✅ COMPLETE. app.py with /reset, /step, /state, /health; CORS; error handlers; 47 tests.
Phase A (Bug Fixes) ✅ COMPLETE. stealth key, severity partial credit, link bonus.
Phase B (Dynamic Cascade) ✅ COMPLETE. environment.py cascade mechanic; grading.py excludes dynamic alerts.

**Next phase: Phase C — inference.py cleanup**
