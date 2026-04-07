# Cloud Alert Triage — OpenEnv Environment

> **OpenEnv hackathon submission · Better Call Coders**

An SRE alert triage environment where an AI agent must classify, prioritise, and remediate cloud infrastructure monitoring alerts across a microservice dependency graph.

---

## Why This Matters

On-call SRE teams triage hundreds of alerts per day. Missing a cascading failure costs millions in downtime; over-reacting to false alarms burns out engineers. This environment models that exact workflow — giving AI agents a realistic, graded triage challenge across three difficulty levels.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Server                     │
│  POST /reset ──► AlertTriageEnv.reset()             │
│  POST /step  ──► AlertTriageEnv.step()              │
│  GET  /state ──► AlertTriageEnv.state()  (debug)    │
│  GET  /health──► {"status": "ok"}                   │
└──────────┬──────────────────────────────────────────┘
           │
    ┌──────▼──────┐   ┌───────────────┐   ┌──────────────┐
    │  Environment │──►│scenario_genera│   │  grading.py  │
    │    Core      │   │     tor.py    │   │ (end-of-ep.) │
    └──────┬───────┘   └───────────────┘   └──────────────┘
           │
    ┌──────▼───────┐
    │  rewards.py  │
    │ (per-step)   │
    └──────────────┘
```

Scenario generation is fully deterministic given `(task_id, seed)`.
The grader score is computed once at episode end and returned in `info["grader_score"]`.

---

## Observation Space

Returned by `POST /reset` (wrapped in `{"observation": …}`) and inside every `POST /step` response.

| Field | Type | Description |
|---|---|---|
| `alerts` | `list[Alert]` | All alerts for this episode. Already-triaged alerts include `agent_decision`. |
| `service_map` | `dict[str, list[str]]` | Adjacency list of the microservice dependency graph. |
| `pending_count` | `int` | Number of un-triaged alerts remaining. |
| `step_number` | `int` | Current step (0-indexed). |
| `max_steps` | `int` | Step budget for this task. |
| `feedback` | `str` | Short hint after the last action (e.g. "Root cause accepted."). |

**Alert fields:**

| Field | Type | Description |
|---|---|---|
| `alert_id` | `str` | Unique ID, e.g. `"alert-001"` |
| `timestamp` | `str` | ISO-8601 timestamp |
| `service` | `str` | Originating microservice |
| `metric` | `str` | Metric name, e.g. `"cpu_usage_percent"` |
| `metric_value` | `float` | Observed value |
| `threshold` | `float` | Threshold that was breached |
| `message` | `str` | Human-readable alert text |
| `context` | `str \| null` | Optional context (recent deploy, dependency info) |
| `triaged` | `bool` | `true` once the agent has acted on this alert |
| `agent_decision` | `dict \| null` | Agent's recorded decision if triaged |

---

## Action Space

All actions share a single model with an `action_type` discriminator.

### `triage` — classify and act on one alert

```json
{
  "action_type": "triage",
  "alert_id":    "alert-001",
  "root_cause":  "deployment_bug",
  "severity":    "high",
  "remediation": "rollback_deploy"
}
```

### `link_alerts` — group alerts that share a root cause

```json
{
  "action_type":    "link_alerts",
  "alert_ids":      ["alert-003", "alert-007", "alert-011"],
  "incident_label": "payment-cascade"
}
```

### `skip` — explicitly skip a false alarm

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

## Tasks

| ID | Title | Alerts | Steps | Incidents | Difficulty |
|---|---|---|---|---|---|
| `easy` | Basic Alert Classification | 5 | 10 | 0 | Easy |
| `medium` | Correlated Incident Response | 15 | 25 | 2 | Medium |
| `hard` | Cascading Failure Under Noise | 30 | 45 | 5 | Hard |

### easy
5 independent alerts, each from a different service, with obvious root causes derived directly from metric names and alert messages. No incidents, no noise. Generous step budget (10 steps for 5 alerts).

### medium
15 alerts across 10 services. Two distinct incidents where alerts cascade through the service dependency graph (e.g., a database failure surfaces as errors in three dependent services). 1–2 false alarms with borderline metrics. Agent must reason across the dependency graph to identify correlated alerts.

### hard
30 alerts across 15 services. Five cascading incidents with 3–5 hop cascades. Six false alarms — one misleadingly marked `critical` by the monitoring system. One "stealth" incident where the root service shows only subtle degradation while dependents fail loudly. Alerts are temporally interleaved (not in causal order). Tight step budget (45 steps for 30 alerts + linking).

---

## Reward Design

Rewards are issued **per step** to guide the agent. The final grader score is computed separately at episode end.

### Per-step rewards

| Action | Condition | Reward |
|---|---|---|
| `triage` | `root_cause` exact match | +0.30 |
| `triage` | `severity` exact match | +0.30 |
| `triage` | `severity` within 1 level | +0.10 |
| `triage` | `remediation` exact match | +0.20 |
| `triage` | alert is in an incident the agent already linked correctly | +0.10 bonus |
| `link_alerts` | correct pair (both alerts in same true incident) | +0.15 per pair |
| `link_alerts` | incorrect pair | −0.10 per pair |
| `skip` | alert is a true false alarm | +0.20 |
| `skip` | alert is a real alert | −0.30 |

### Penalties

| Condition | Penalty |
|---|---|
| Step ≥ 80% of step budget | −0.05 per step |
| Invalid action format | −0.10 |
| Triaging an already-triaged alert | −0.15 |

---

## Grader (End-of-Episode Score)

The grader computes a deterministic score in **[0.0, 1.0]** from the final episode state. Un-triaged alerts count as wrong on all components.

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
- **severity_accuracy** — per alert: +1.0 exact, +0.5 within 1 level, +0.0 otherwise; averaged
- **remediation_accuracy** — fraction of alerts with correct remediation
- **incident_link_f1** — F1 over alert-pair sets; 1.0 if no true incidents (vacuously correct)
- **false_alarm_accuracy** — (correctly skipped FAs + correctly triaged real alerts) / total; 1.0 if no FAs
- **stealth bonus** — +0.05 if the root cause service of the stealth incident was correctly identified

---

## API Reference

### `POST /reset`

Start a new episode.

**Request:**
```json
{ "task_id": "easy", "seed": 42 }
```

**Response (200):**
```json
{ "observation": { "alerts": [...], "service_map": {...}, "pending_count": 5, "step_number": 0, "max_steps": 10, "feedback": "" } }
```

**Errors:** `422` for unknown `task_id`.

---

### `POST /step`

Apply one action.

**Response (200):**
```json
{ "observation": {...}, "reward": 0.80, "done": false, "info": {} }
```

When `done` is `true`, `info` contains `{"grader_score": 0.92}`.

**Errors:** `400` if called before `/reset` · `422` for malformed action.

---

### `GET /state`

Return full internal state including hidden ground truth. For evaluation/debugging only — the baseline agent must not call this.

---

### `GET /health`

```json
{ "status": "ok" }
```

---

## Setup

### Local (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run the baseline agent (separate terminal)
export OPENAI_API_KEY=sk-...
export MODEL_NAME=gpt-4o-mini        # or any OpenAI-compatible model
python inference.py
```

**Environment variables for `inference.py`:**

| Variable | Default | Description |
|---|---|---|
| `ENV_URL` | `http://localhost:7860` | URL of the running environment server |
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL |
| `MODEL_NAME` | `gpt-4o-mini` | Model name |
| `OPENAI_API_KEY` | — | API key (falls back to `HF_TOKEN`) |
| `HF_TOKEN` | — | Hugging Face token (fallback auth) |

### Docker

```bash
# Build
docker build -t cloud-alert-triage .

# Run
docker run -p 7860:7860 cloud-alert-triage

# Verify
curl http://localhost:7860/health
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy","seed":42}' | python -m json.tool
```

### Run tests

```bash
pytest tests/ -v
```

---

## Baseline Scores

Scores recorded with `seed=42`, `temperature=0`.

| Task | Model | Grader Score | Steps Used |
|---|---|---|---|
| easy | _pending live run_ | — | — |
| medium | _pending live run_ | — | — |
| hard | _pending live run_ | — | — |

> **Expected ranges** (strong frontier LLM): easy 0.85–1.0 · medium 0.65–0.85 · hard 0.40–0.65

---

## Project Structure

```
cloud-alert-triage/
├── inference.py              # Baseline LLM agent
├── openenv.yaml              # OpenEnv metadata
├── Dockerfile                # Container definition (port 7860)
├── requirements.txt
├── server/
│   ├── app.py                # FastAPI endpoints
│   ├── environment.py        # Episode state machine
│   ├── scenario_generator.py # Deterministic alert generation
│   ├── rewards.py            # Per-step reward calculation
│   ├── grading.py            # End-of-episode grader
│   ├── service_graph.py      # 17-service dependency DAG
│   ├── models.py             # Pydantic v2 models
│   └── config.py             # Enums and constants
├── tasks/
│   ├── task_easy.json
│   ├── task_medium.json
│   └── task_hard.json
└── tests/                    # 232 tests, all passing
```

---

## Team

**Better Call Coders** — OpenEnv Hackathon 2026

---

## License

MIT
