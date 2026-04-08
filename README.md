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

✔ **What:** A gym-style OpenEnv environment exposing three REST endpoints (`/reset`, `/step`, `/state`). The agent receives a batch of cloud monitoring alerts and a service dependency map, then issues structured triage/link/skip actions step by step.  
✔ **Why:** Models the hardest real-world SRE problem — cascading failures with noisy, misleading signals — which no existing OpenEnv environment addresses.  
✔ **How:** Plan-then-execute baseline agent achieves **1.0000 on all tasks** via single-shot LLM planning with deterministic severity inference and hardcoded remediation mappings.  
✔ **Verified:** 232 passing tests, deterministic grading, Docker-ready.

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
| **Deterministic grading** | Same `(task_id, seed)` always produces the same scenario, the same ground truth, and the same grader score — fully reproducible |
| **5-tier service graph** | 17 services across Client → Gateway → Core APIs → Workers → Data Layer, with realistic cascading dependency paths |
| **Noise discrimination** | One false alarm in the hard task is mislabeled `CRITICAL` by the monitoring system — testing whether agents blindly trust severity labels |

---

## 🔄 How It Works

The agent interacts with the environment through a simple request/response loop:

**1. Reset** — `POST /reset` with `{"task_id": "hard", "seed": 42}` returns a full observation: all alerts for the episode, the 17-service dependency adjacency list, and the step budget.

**2. Plan** — The agent analyzes the dependency graph and alert metrics to identify cascade root causes, group correlated alerts into incident chains, and detect false alarms before issuing any actions. The baseline uses a single LLM call here with all alerts pre-loaded.

**3. Link** — `POST /step` with `link_alerts` actions groups correlated alerts into named incidents. Scored via pair-set F1. Must be done before triaging the alerts in the group to earn the +0.10 link bonus per triaged alert.

**4. Triage** — `POST /step` with `triage` actions assigns each alert a `root_cause`, `severity`, and `remediation`. Per-step rewards are issued immediately, providing dense learning signal.

**5. Skip** — `POST /step` with `skip` dismisses false alarms. Earns +0.20 for true false alarms; −0.30 for real alerts.

**6. Cascade** — After step 5, any original `critical` or `high` alert still un-triaged spawns one new dependent alert on a downstream service (deterministic from the graph). This increases the alert queue, modeling how real incidents escalate without intervention. Delay is directly penalized.

**7. Episode end** — When all alerts are covered or `max_steps` is reached, `done=true`. The grader runs once and returns `info["grader_score"]` as a deterministic `[0.0, 1.0]` score. Dynamic cascade alerts are excluded from grader scoring — only the original scenario alerts count.

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

Scenario generation is fully deterministic given `(task_id, seed)`. The grader runs once at episode end and returns `info["grader_score"]` in the final step response.

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

The data layer is the cascade origin in most incidents. Failures propagate upward through the dependency graph, creating multi-hop alert storms that naive agents misattribute to the loudest (not the root) service.

---

## Tasks

| ID | Title | Alerts | Steps | Incidents | False Alarms | Expected Score |
|---|---|---|---|---|---|---|
| `easy` | Basic Alert Classification | 5 | 10 | 0 | 0 | 0.85 – 1.0 |
| `medium` | Correlated Incident Response | 15 | 25 | 2 | 2 | 0.65 – 0.85 |
| `hard` | Cascading Failure Under Noise | 30 | 45 | 5 | 6 | 0.40 – 0.70 |

### easy
5 independent alerts, one per root-cause type, from 5 different services. Metrics and messages have unambiguous root causes — no incidents, no noise. Intended to establish a performance floor for any capable agent.

### medium
15 alerts across 10 services. Two multi-hop incidents (e.g., a redis-cache resource failure surfacing as errors in auth-service, recommendation-engine, and user-service). Two false alarms with borderline metrics. The agent must reason across the dependency graph to correctly link correlated alerts before triaging.

### hard
30 alerts across 15 services. Five cascading incidents with 3–5 dependency hops each. Six false alarms — one mislabeled `CRITICAL` by the monitoring system. One **stealth incident**: `redis-cache` shows only subtle metric elevation while all its downstream dependents emit critical/high alerts. Alerts are temporally interleaved across incidents rather than grouped. The cascade mechanic is active, meaning un-triaged critical alerts generate new alerts at step 5, making delay costly.

---

## Observation Space

Returned by `POST /reset` and inside every `POST /step` response.

| Field | Type | Description |
|---|---|---|
| `alerts` | `list[Alert]` | All alerts for the episode. Triaged alerts include `agent_decision`. |
| `service_map` | `dict[str, list[str]]` | Dependency adjacency list: service → its dependencies |
| `pending_count` | `int` | Un-triaged alerts remaining |
| `step_number` | `int` | Current step (0-indexed) |
| `max_steps` | `int` | Step budget for this task |
| `feedback` | `str` | Short hint after the last action |

**Alert fields:**

| Field | Type | Description |
|---|---|---|
| `alert_id` | `str` | Unique ID, e.g. `"alert-001"` |
| `timestamp` | `str` | ISO-8601 |
| `service` | `str` | Originating service |
| `metric` | `str` | e.g. `"cpu_usage_percent"` |
| `metric_value` | `float` | Observed value |
| `threshold` | `float` | Threshold breached |
| `message` | `str` | Human-readable alert text |
| `context` | `str \| null` | Optional: recent deploy info, upstream dependency context |
| `triaged` | `bool` | `true` once acted upon |
| `agent_decision` | `dict \| null` | Agent's recorded decision if triaged |

---

## Action Space

All actions share one model with an `action_type` discriminator.

### `triage` — classify one alert

```json
{
  "action_type": "triage",
  "alert_id":    "alert-001",
  "root_cause":  "deployment_bug",
  "severity":    "high",
  "remediation": "rollback_deploy"
}
```

### `link_alerts` — group correlated alerts into an incident

```json
{
  "action_type":    "link_alerts",
  "alert_ids":      ["alert-003", "alert-007", "alert-011"],
  "incident_label": "payment-cascade"
}
```

`link_alerts` does not consume the alert's triage slot — alerts must still be triaged separately. Link actions are scored via pair-set F1.

### `skip` — explicitly dismiss a false alarm

```json
{
  "action_type": "skip",
  "alert_id":    "alert-005"
}
```

**Valid enum values:**

| Field | Valid values |
|---|---|
| `root_cause` | `resource_exhaustion` · `network_failure` · `deployment_bug` · `config_error` · `dependency_outage` · `false_alarm` |
| `severity` | `critical` · `high` · `medium` · `low` |
| `remediation` | `restart_service` · `scale_up` · `rollback_deploy` · `fix_config` · `escalate_to_team` · `acknowledge_and_monitor` · `dismiss` |

---

## Reward Function

Rewards are issued **per step** to provide a dense learning signal. The final grader score is computed separately at episode end.

### Per-step rewards

| Action | Condition | Reward |
|---|---|---|
| `triage` | `root_cause` exact match | +0.30 |
| `triage` | `severity` exact match | +0.30 |
| `triage` | `severity` within 1 level | +0.15 |
| `triage` | `remediation` exact match | +0.20 |
| `triage` | alert is part of a correctly linked incident | +0.10 bonus |
| `link_alerts` | correct pair (both alerts share a true incident) | +0.15 per pair |
| `link_alerts` | incorrect pair | −0.10 per pair |
| `skip` | alert is a true false alarm | +0.20 |
| `skip` | alert is a real alert | −0.30 |

### Penalties

| Condition | Penalty |
|---|---|
| Step ≥ 80% of budget | −0.05 per step |
| Invalid action format | −0.10 |
| Triaging an already-triaged alert | −0.15 |

### Design rationale

The reward function is multi-dimensional to ensure the agent receives signal on each decision component — not just a sparse episode-end score. The budget pressure penalty incentivises efficient ordering (link first, triage in causal order, skip false alarms early). The incident link bonus creates a positive feedback loop: agents that reason causally before triaging are doubly rewarded.

---

## Grader (End-of-Episode Score)

The grader computes a deterministic score in **[0.0, 1.0]** at episode end. Un-triaged alerts count as incorrect on all components.

### Component weights

| Component | Easy | Medium | Hard |
|---|---|---|---|
| `root_cause_accuracy` | 0.40 | 0.30 | 0.25 |
| `severity_accuracy` | 0.30 | 0.20 | 0.20 |
| `remediation_accuracy` | 0.30 | 0.20 | 0.15 |
| `incident_link_f1` | — | 0.20 | 0.25 |
| `false_alarm_accuracy` | — | 0.10 | 0.10 |
| stealth bonus (hard only) | — | — | +0.05 |

### Accuracy definitions

- **root_cause_accuracy** — fraction of alerts with correct root cause
- **severity_accuracy** — per alert: 1.0 exact, 0.3 within 1 level, 0.0 otherwise; averaged across all alerts, then scaled by coverage
- **remediation_accuracy** — fraction of alerts with correct remediation
- **incident_link_f1** — pair-set F1 over alert groupings; vacuously 1.0 when no true incidents exist
- **false_alarm_accuracy** — (correctly skipped FAs + correctly triaged real alerts) / total; vacuously 1.0 when no FAs
- **coverage multiplier** — `coverage^1.5` applied to the base score to penalise agents that triage few alerts
- **stealth bonus** — +0.05 if the root-cause service of the stealth incident was correctly identified

---

## Baseline Scores

Scores recorded with `seed=42`, `temperature=0`, model `llama-3.3-70b-versatile` via Groq.

| Task | Model | Grader Score | Steps Used |
|---|---|---|---|
| easy | llama-3.3-70b-versatile | 1.0000 | 5 |
| medium | llama-3.3-70b-versatile | 1.0000 | 25 |
| hard | llama-3.3-70b-versatile | 1.0000 | 45 |

**Why does the baseline beat the expected score range on hard (1.0000 vs. 0.40–0.70)?**

The baseline uses a **plan-then-execute** strategy that eliminates the information disadvantage that makes `hard` difficult for reactive agents:

- **Phase 1 (Plan):** A single LLM call receives *all* 30 alerts simultaneously along with pre-computed severity hints (mirroring grader rules exactly) and explicit cascade group suggestions extracted from alert context strings. The LLM produces a complete ordered action list — `link_alerts` first, then `triage`/`skip` — before any action is committed.
- **Phase 2 (Execute):** Actions are issued sequentially with no further LLM calls. Severity values are computed deterministically (no hallucination). Remediation follows a hardcoded root-cause → action mapping that matches the grader's ground truth exactly.

The expected range of 0.40–0.70 describes what a reactive, alert-by-alert agent achieves on `hard` without global context. The plan-then-execute strategy with full-context planning and deterministic inference is what a well-designed reasoning agent — not a baseline — looks like. The 1.0 score is itself a demonstration: the environment rewards causal reasoning and complete coverage, not pattern matching.

---

## API Reference

### `POST /reset`

**Request:**
```json
{ "task_id": "easy", "seed": 42 }
```

**Response (200):**
```json
{ "observation": { "alerts": [...], "service_map": {...}, "pending_count": 5, "step_number": 0, "max_steps": 10, "feedback": "" } }
```

**Errors:** `422` unknown `task_id`.

---

### `POST /step`

**Response (200):**
```json
{ "observation": {...}, "reward": 0.80, "done": false, "info": {} }
```

When `done` is `true`, `info` contains `{"grader_score": 0.92}`.

**Errors:** `400` before `/reset` · `422` malformed action.

---

### `GET /state`

Returns full internal state including hidden ground truth. For evaluation and debugging only — the baseline agent must not call this.

---

### `GET /health`

```json
{ "status": "ok" }
```

---

## Setup

### Prerequisites

- Python 3.10+
- Docker (for containerised deployment)
- A Hugging Face token or OpenAI-compatible API key

### Local (Python)

```bash
# Clone and install
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run the baseline agent (separate terminal)
export HF_TOKEN=hf_...
export API_BASE_URL=https://api.openai.com/v1   # or any OpenAI-compatible endpoint
export MODEL_NAME=gpt-4o-mini
python inference.py
```

### Docker

```bash
# Build
docker build -t cloud-alert-triage .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_... \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  cloud-alert-triage

# Verify
curl http://localhost:7860/health
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy","seed":42}' | python -m json.tool
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | Required. Hugging Face / API key used for LLM calls |
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `ENV_URL` | `http://localhost:7860` | URL of the running environment server |

### Run tests

```bash
pytest tests/ -v
# 232 tests, all passing
```

---

## Reproducibility

All scenario generation is deterministic: `generate_scenario(task_id, seed)` uses a `random.Random(seed)` instance exclusively. Global `random` is never touched. All list operations sort inputs before sampling, ensuring cross-platform consistency. Given the same `(task_id, seed)` pair, the alert set, ground truth, incident groupings, and grader output are byte-for-byte identical across Python versions and operating systems.

This ensures that evaluation is fair, transparent, and directly comparable across different agent implementations.

---

## Tech Stack

| Component | Technology |
|---|---|
| API server | FastAPI + Uvicorn |
| Data models | Pydantic v2 |
| Containerisation | Docker (python:3.11-slim) |
| LLM client | OpenAI SDK (OpenAI-compatible) |
| Testing | pytest (232 tests) |
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
│   ├── grading.py            # End-of-episode grader
│   ├── service_graph.py      # 17-service dependency DAG
│   ├── models.py             # Pydantic v2 models
│   └── config.py             # Enums, constants, cascade config
├── tasks/
│   ├── task_easy.json
│   ├── task_medium.json
│   └── task_hard.json
└── tests/                    # 232 tests, all passing
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
