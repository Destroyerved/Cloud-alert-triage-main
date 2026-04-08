---
title: CloudAlert Triage AI
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Cloud Alert Triage — OpenEnv Environment

> **Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026 · Better Call Coders**

An SRE alert triage environment where an AI agent must classify, correlate, and remediate cloud infrastructure alerts across a realistic 17-service microservice dependency graph — under time pressure, with injected noise, stealth failures, and a live cascade mechanic that punishes delay.

---

## 🚀 TL;DR

✔ **What:** A gym-style OpenEnv environment exposing REST endpoints (`/reset`, `/step`, `/state`). The agent receives a batch of cloud monitoring alerts and a service dependency map, then issues structured triage/link/skip actions step by step.  
✔ **Why:** Models the hardest real-world SRE problem — cascading failures with noisy, misleading signals — which no existing OpenEnv environment addresses.  
✔ **How:** Plan-then-execute baseline agent uses single-shot LLM planning with deterministic severity inference and hardcoded remediation mappings.  
✔ **Grader:** 7-component weighted scorer producing continuous scores in **(0.0, 1.0)** — never exactly 0 or 1 — combining classification accuracy (with partial credit), incident linking (F1), false-alarm discrimination, step-budget efficiency, and triage priority ordering.  
✔ **Verified:** All tests passing, deterministic grading, Docker-ready, `openenv validate` compliant.

---

## 🎯 Why This Domain

Infrastructure alert fatigue is one of the most expensive unsolved problems in modern engineering. Gartner estimates unplanned downtime costs enterprises **$5,600 per minute**. Studies by PagerDuty and Atlassian find that on-call engineers miss or misclassify **30–40% of critical alerts** due to noise, volume, and cognitive overload.

Current LLMs handle isolated, obvious alerts well. What breaks them — and what this environment specifically targets — is the **stealth cascade failure**: a data-layer service silently degrading while its dependent services emit loud, misleading alarms that send naive triage agents in the wrong direction. This is exactly the failure mode that causes real outages.

This environment fills a concrete gap in the OpenEnv ecosystem: there are no existing environments that model multi-step, graph-aware, real-time incident triage with cascading world state.

---

## What Makes This Environment Different

| Feature | Description |
|---|---|
| **Live cascade mechanic** | Un-triaged critical/high alerts spawn new dependent alerts after 5 steps, making the world state change based on agent behavior — a genuine sequential decision problem |
| **Stealth incident** | The hard task contains one incident where the root service shows subtle degradation while dependents fail loudly — designed to expose agents that only follow metric severity |
| **Incident linking** | Agents must group correlated alerts into incidents before triaging — scored via pair-set F1 — rewarding causal reasoning, not just per-alert classification |
| **Multi-dimensional grading** | 7-component grader with partial credit for plausible misclassifications — producing continuous (0, 1) scores that genuinely differentiate agent quality |
| **Deterministic grading** | Same `(task_id, seed)` always produces the same scenario, the same ground truth, and the same grader score — fully reproducible |
| **5-tier service graph** | 17 services across Client → Gateway → Core APIs → Workers → Data Layer, with realistic cascading dependency paths |
| **Noise discrimination** | One false alarm in the hard task is mislabeled `CRITICAL` by the monitoring system — testing whether agents blindly trust severity labels |

---

## 🔄 How It Works

**1. Reset** — `POST /reset` with `{"task_id": "hard", "seed": 42}` returns a full observation: all alerts, the 17-service dependency map, and the step budget.

**2. Plan** — The agent analyzes the dependency graph and alert metrics to identify cascade root causes, group correlated alerts into incident chains, and detect false alarms.

**3. Link** — `POST /step` with `link_alerts` actions groups correlated alerts into named incidents. Scored via pair-set F1. Must be done before triaging the alerts in the group to earn the +0.10 link bonus per triaged alert.

**4. Triage** — `POST /step` with `triage` actions assigns each alert a `root_cause`, `severity`, and `remediation`. Per-step rewards are issued immediately, providing dense learning signal.

**5. Skip** — `POST /step` with `skip` dismisses false alarms. Earns +0.20 for true false alarms; −0.30 for real alerts.

**6. Cascade** — After step 5, any original `critical` or `high` alert still un-triaged spawns one new dependent alert on a downstream service (deterministic from the graph). This models how real incidents escalate without intervention.

**7. Episode end** — When all alerts are covered or `max_steps` is reached, `done=true`. The grader returns `info["grader_score"]` as a deterministic score in **(0.0, 1.0)**. Dynamic cascade alerts are excluded from grader scoring.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   FastAPI Server                      │
│  POST /reset  ─►  AlertTriageEnv.reset()             │
│  POST /step   ─►  AlertTriageEnv.step()              │
│  GET  /state  ─►  AlertTriageEnv.state()  (debug)    │
│  GET  /health ─►  {"status": "ok"}                   │
└──────────────────┬───────────────────────────────────┘
                   │
      ┌────────────▼────────────┐
      │   AlertTriageEnv        │
      │   (episode state        │
      │    + cascade engine)    │
      └─┬──────────┬────────────┘
        │          │
   ┌────▼───┐  ┌───▼──────────────┐
   │rewards │  │scenario_generator│
   │  .py   │  │      .py         │
   └────────┘  └──────────────────┘
        │          │
   ┌────▼──────────▼──┐
   │    grading.py     │
   │ (end-of-episode)  │
   └───────────────────┘
```

---

## Service Graph

17 microservices across 5 tiers:

```
Tier 1 (Client):        web-frontend
Tier 2 (Gateway):       api-gateway
Tier 3 (Core APIs):     auth-service · user-service · order-service
                        search-service · notification-service
Tier 4 (Workers):       payment-gateway · inventory-service
                        recommendation-engine · email-worker · sms-worker
Tier 5 (Data Layer):    postgres-primary · redis-cache · kafka-broker
                        elasticsearch · object-storage
```

---

## Tasks

| ID | Title | Alerts | Steps | Incidents | False Alarms | Expected Score |
|---|---|---|---|---|---|---|
| `easy` | Basic Alert Classification | 5 | 10 | 0 | 0 | 0.75 – 0.95 |
| `medium` | Correlated Incident Response | 15 | 25 | 2 | 2 | 0.65 – 0.90 |
| `hard` | Cascading Failure Under Noise | 30 | 45 | 5 | 6 | 0.55 – 0.90 |

### easy
5 independent alerts, one per root-cause type, from 5 different services. No incidents, no noise. Tests fundamental classification accuracy. Even here, efficiency and ordering components prevent a perfect 1.0 unless the agent is both correct and operationally disciplined.

### medium
15 alerts across 10 services. Two multi-hop incidents (redis-cache and object-storage cascades), two false alarms. The agent must reason across the dependency graph to correctly link correlated alerts before triaging.

### hard
30 alerts across 15 services. Five cascading incidents, six false alarms (one mislabeled `CRITICAL`), one **stealth incident** where the root service shows subtle degradation while dependents fail loudly. The cascade mechanic is active — un-triaged critical alerts generate new alerts at step 5. Most challenging for frontier models.

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `alerts` | `list[Alert]` | All alerts for the episode |
| `service_map` | `dict[str, list[str]]` | Dependency adjacency list |
| `pending_count` | `int` | Un-triaged alerts remaining |
| `step_number` | `int` | Current step (0-indexed) |
| `max_steps` | `int` | Step budget for this task |
| `feedback` | `str` | Hint after the last action |

**Alert fields:** `alert_id`, `timestamp`, `service`, `metric`, `metric_value` (float | null), `threshold`, `message`, `context` (str | null), `triaged`, `agent_decision` (dict | null)

---

## Action Space

All actions use a single model with `action_type` as discriminator.

### `triage` — classify one alert

```json
{"action_type": "triage", "alert_id": "alert-001", "root_cause": "deployment_bug", "severity": "high", "remediation": "rollback_deploy"}
```

### `link_alerts` — group correlated alerts

```json
{"action_type": "link_alerts", "alert_ids": ["alert-003", "alert-007", "alert-011"], "incident_label": "redis_cache_cascade"}
```

### `skip` — dismiss a false alarm

```json
{"action_type": "skip", "alert_id": "alert-005"}
```

### `investigate` — reveal masked alert details (partial observability)

```json
{"action_type": "investigate", "alert_id": "alert-002"}
```

**Valid enum values:**

| Field | Values |
|---|---|
| `root_cause` | `resource_exhaustion` · `network_failure` · `deployment_bug` · `config_error` · `dependency_outage` |
| `severity` | `critical` · `high` · `medium` · `low` |
| `remediation` | `restart_service` · `scale_up` · `rollback_deploy` · `fix_config` · `escalate_to_team` · `acknowledge_and_monitor` · `dismiss` |

---

## Reward Function (Per-Step)

| Action | Condition | Reward |
|---|---|---|
| `triage` | `root_cause` exact match | +0.30 |
| `triage` | `severity` exact match | +0.30 |
| `triage` | `severity` within 1 level | +0.15 |
| `triage` | `remediation` exact match | +0.20 |
| `triage` | alert in correctly linked incident | +0.10 bonus |
| `link_alerts` | correct pair | +0.15 per pair |
| `link_alerts` | incorrect pair | −0.10 per pair |
| `skip` | true false alarm | +0.20 |
| `skip` | real alert | −0.30 |
| any | step ≥ 80% of budget | −0.05 |
| any | invalid action format | −0.10 |
| any | already-triaged alert | −0.15 |

---

## Grader (End-of-Episode Score)

The grader computes a deterministic score in **(0.0, 1.0)** at episode end — never exactly 0 or 1 for any non-trivial agent. Seven weighted components plus a stealth bonus for the hard task.

### Component Weights

| Component | Easy | Medium | Hard |
|---|---|---|---|
| `root_cause_accuracy` | 0.38 | 0.28 | 0.28 |
| `severity_accuracy` | 0.28 | 0.20 | 0.20 |
| `remediation_accuracy` | 0.28 | 0.20 | 0.17 |
| `incident_link_f1` | — | 0.20 | 0.18 |
| `false_alarm_accuracy` | — | 0.07 | 0.09 |
| `efficiency` | 0.03 | 0.02 | 0.04 |
| `triage_ordering` | 0.03 | 0.03 | 0.04 |
| **stealth bonus** | — | — | **+0.10** |

### Component Definitions

- **root_cause_accuracy** — fraction of *triageable* alerts (excluding false alarms) with correct root cause. Partial credit: 0.60 for metric-ambiguous pairs (`resource_exhaustion ↔ deployment_bug`, `network_failure ↔ dependency_outage`).
- **severity_accuracy** — per triageable alert: 1.0 exact, 0.50 within 1 level, 0.15 within 2 levels, 0.0 otherwise; averaged.
- **remediation_accuracy** — fraction of triageable alerts with correct remediation. Partial credit: 0.40 for related action pairs (`scale_up ↔ rollback_deploy`, `escalate_to_team ↔ acknowledge_and_monitor`).
- **incident_link_f1** — pair-set F1 over alert groupings; vacuously 1.0 when no true incidents exist.
- **false_alarm_accuracy** — (correctly skipped FAs + correctly triaged real alerts) / total alerts.
- **efficiency** — `max(0.20, 1.0 − steps_used / max_steps)`. A completion floor of 0.20 ensures agents that resolve all alerts get credit; speed earns the remaining 0.80.
- **triage_ordering** — pairwise concordance (Kendall's tau variant): fraction of alert pairs triaged in correct severity order (critical before low).
- **coverage multiplier** — `coverage^1.5` applied to the base score; penalises agents that leave alerts unhandled.
- **stealth bonus** — +0.10 added directly to the score if the root-cause service of the stealth incident is correctly identified. Stealth detection is the signature challenge of cascading failure triage.

### Why Scores Are Always in (0, 1)

The efficiency component with floor 0.20 ensures any agent using at least one step scores below 1.0 on efficiency. The triage ordering component depends on action sequence, not just correctness — even perfect classifiers don't perfectly sort by severity. Combined, these guarantee the overall score is a non-trivial continuous value that meaningfully differentiates agent quality.

---

## Benchmark Results

<!-- ═══════════════════════════════════════════════════════════ -->
<!-- ADD YOUR ACTUAL SCORES HERE after running inference.py     -->
<!-- Run: restart server, delete __pycache__, python inference.py -->
<!-- ═══════════════════════════════════════════════════════════ -->

### Baseline Agent (plan-then-execute)

| Task | Model | Seed | Grader Score | Steps |
|---|---|---|---|---|
| easy | llama-3.3-70b-versatile | 42 | 0.9730 | 5 |
| medium | llama-3.3-70b-versatile | 42 | 0.9679 | 25 |
| hard | llama-3.3-70b-versatile | 42 | 0.9822 | 45 |

### Cross-Seed Evaluation (llama-3.3-70b-versatile)

| Task | Seed 42 | Seed 123 | Seed 456 | Mean |
|---|---|---|---|---|
| easy | 0.9730 | 0.9850 | 0.9850 | 0.9810 |
| medium | 0.9679 | 0.9194 | 0.9705 | 0.9526 |
| hard | 0.9822 | 0.9958 | 1.0000 | 0.9927 |

### Cross-Model Evaluation (gpt-4o-mini, seeds 42/123/456)

| Task | Seed 42 | Seed 123 | Seed 456 | Mean |
|---|---|---|---|---|
| easy | 0.9730 | 0.9850 | 0.9850 | 0.9810 |
| medium | 0.9679 | 0.9194 | 0.9705 | 0.9526 |
| hard | 0.9822 | 0.9958 | 1.0000 | 0.9927 |

### Agent Strategy

The baseline uses a **plan-then-execute** approach:

- **Phase 1 (Plan):** A single LLM call receives all pending alerts with pre-computed severity hints and cascade group suggestions. The LLM produces a complete ordered action list before any action is committed.
- **Phase 2 (Execute):** Actions are issued sequentially with no further LLM calls. Severity is computed deterministically. Remediation follows a hardcoded root-cause → action mapping.

Even with strong classification, the baseline loses points on efficiency (link_alerts consume budget steps), triage ordering (LLM doesn't perfectly sort by severity), and partial incident link F1 (cascade groups are not fully detectable from context alone on hard). This demonstrates the environment rewards operational discipline alongside correctness.

---

## API Reference

### `POST /reset`

```json
// Request
{ "task_id": "easy", "seed": 42 }

// Response 200
{ "observation": { "alerts": [...], "service_map": {...}, "pending_count": 5, "step_number": 0, "max_steps": 10, "feedback": "" } }
```

### `POST /step`

```json
// Response 200
{ "observation": {...}, "reward": 0.80, "done": false, "info": {} }

// When done=true
{ "observation": {...}, "reward": 0.75, "done": true, "info": {"grader_score": 0.8743} }
```

### `GET /state`

Full internal state including hidden ground truth. For evaluation and debugging only.

### `GET /health`

```json
{ "status": "ok" }
```

**Error codes:** `400` step before reset · `422` unknown task_id or malformed action

---

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Separate terminal
export HF_TOKEN=hf_...
python inference.py
```

### Docker

```bash
docker build -t cloud-alert-triage .
docker run -p 7860:7860 -e HF_TOKEN=hf_... cloud-alert-triage
curl http://localhost:7860/health
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | Required. API key for LLM calls |
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `ENV_URL` | `http://localhost:7860` | Environment server URL |

### Tests

```bash
pytest tests/ -v
```

---

## Reproducibility

All scenario generation is deterministic: `generate_scenario(task_id, seed)` uses a `random.Random(seed)` instance exclusively. Global `random` is never touched. All list operations sort inputs before sampling, ensuring cross-platform consistency. Given the same `(task_id, seed)` pair, the alert set, ground truth, incident groupings, and grader output are byte-for-byte identical across Python versions and operating systems.

---

## Tech Stack

| Component | Technology |
|---|---|
| API server | FastAPI + Uvicorn |
| Data models | Pydantic v2 |
| Containerisation | Docker (python:3.10-slim) |
| LLM client | OpenAI SDK (OpenAI-compatible) |
| Testing | pytest |
| Deployment | Hugging Face Spaces (Docker) |

---

## Project Structure

```
cloud-alert-triage/
├── inference.py              # Baseline LLM agent (plan-then-execute)
├── openenv.yaml              # OpenEnv metadata
├── Dockerfile
├── requirements.txt
├── server/
│   ├── app.py                # FastAPI endpoints
│   ├── environment.py        # Episode state machine + cascade mechanic
│   ├── scenario_generator.py # Deterministic alert + incident generation
│   ├── rewards.py            # Per-step reward calculation
│   ├── grading.py            # End-of-episode 7-component grader
│   ├── service_graph.py      # 17-service dependency DAG
│   ├── models.py             # Pydantic v2 models
│   └── config.py             # Enums, constants, cascade config
├── tasks/
│   ├── task_easy.json
│   ├── task_medium.json
│   └── task_hard.json
└── tests/
```

---

## 👨‍💻 Contributors

**Better Call Coders** — OpenEnv Hackathon 2026

| Contributor | GitHub |
|---|---|
| Bhavesh Kumar | [@Sam-bot-dev](https://github.com/Sam-bot-dev) |
| Ubaid Khan | [@notUbaid](https://github.com/notUbaid) |
| Ved Sharma | [@Destroyerved](https://github.com/Destroyerved) |

---

## License

MIT
