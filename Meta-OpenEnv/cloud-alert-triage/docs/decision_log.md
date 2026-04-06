# Decision Log

Record of non-obvious design decisions made during implementation.
Format: timestamp, author, decision, alternatives considered, rationale.

---

## 2026-04-06 — Agent (Phase 1 Bootstrap)

### Decision: Single `Action` model with `action_type` discriminator
- **Alternatives**: Separate `TriageAction`, `LinkAlertsAction`, `SkipAction` Pydantic models with a Union discriminator.
- **Chosen**: Single `Action(BaseModel)` with optional fields and a `model_validator` enforcing required fields per `action_type`.
- **Rationale**: Simpler JSON schema for LLM to produce. Avoids Pydantic v2 discriminated union complexity. Easier to serialize in `openenv.yaml` action_space description. The validator approach catches runtime errors clearly. Master plan explicitly prescribes this approach (Section 6).

### Decision: One global `AlertTriageEnv` instance (not per-session)
- **Alternatives**: Session-token-based instances (one env per caller).
- **Chosen**: Single global instance. `/reset` overwrites previous episode.
- **Rationale**: Sufficient for hackathon. Avoids session management complexity. Simplifies the API — no session tokens needed in `openenv.yaml`. Master plan explicitly documents this (Section 5, `server/environment.py`).

### Decision: Port 7860 as default
- **Alternatives**: 8000 (FastAPI default), 5000.
- **Chosen**: 7860.
- **Rationale**: Hugging Face Spaces uses 7860 as the expected port for Docker-based apps. Using it locally avoids a port-mismatch surprise when deploying. Master plan prescribes this (Section 4, Phase 11).

### Decision: `random.Random(seed)` instance (not global `random` module)
- **Alternatives**: Setting `random.seed()` globally, using `numpy.random`.
- **Chosen**: `random.Random(seed)` instance passed to all helper functions.
- **Rationale**: Global `random.seed()` creates cross-contamination if any other code uses `random`. `numpy.random` adds a dependency. A seeded `Random` instance is thread-safe and fully encapsulated. Master plan mandates this (Section 5, `server/scenario_generator.py`).

### Decision: Grader score in `info` dict at episode end
- **Alternatives**: Separate `/grade` endpoint, always-present `grader_score` field.
- **Chosen**: `grader_score` added to `info` only when `done == True`.
- **Rationale**: Matches OpenEnv conventions (step returns `(obs, reward, done, info)`). Keeps the `info` dict lightweight during active episodes. Grader is expensive to call repeatedly. Master plan specifies this (Section 4, Phase 4; Section 10).

### Decision: Severity proximity scoring (+0.10 for within-1-level)
- **Alternatives**: Binary exact-match only, linear distance scoring.
- **Chosen**: +0.30 exact, +0.10 within-1-level, +0.00 otherwise.
- **Rationale**: Provides gradient signal for almost-correct severity judgments. LLMs that confuse "high" with "critical" are closer to correct than those that guess "low". Binary grading would zero out meaningful partial progress. Master plan specifies these exact values (Section 8).

### Decision: Incident link F1 (not precision or recall alone)
- **Alternatives**: Precision-only, recall-only, Jaccard similarity.
- **Chosen**: F1 over alert-pairs (convert groups to pairs, compute P and R, then F1).
- **Rationale**: F1 balances over-linking (high false-positive rate, low precision) and under-linking (missing real incidents, low recall). Using pairs avoids the "one big group" hack (linking all alerts together would give 100% recall but low precision). Master plan specifies F1 (Section 9).

### Decision: `tasks/*.json` carry grader weights
- **Alternatives**: Hardcode weights in `grading.py`, derive from task difficulty string.
- **Chosen**: Weights stored in `task_*.json` and loaded by `grading.py`.
- **Rationale**: Makes per-task tuning visible and auditable without code changes. JSON is easy for humans to review. Future tasks can be added by adding a JSON file.

---

## 2026-04-07 — Agent (Phase 6 Graders + Final Polish)

### Decision: Coverage penalty (`coverage ^ 1.5`) in grader
- **Alternatives**: No coverage penalty; linear penalty; hard zero for under-50% coverage.
- **Chosen**: Multiply base score by `(triaged_count / total_alerts) ^ 1.5`.
- **Rationale**: Prevents agents from gaming the grader by triaging only the easiest alerts and skipping hard ones. The 1.5 exponent is steeper than linear — an agent that triages 50% of alerts takes a ~65% penalty, not 50%. Clamped to [0.0, 1.0] after application.

### Decision: Two-layer scoring (per-step reward + end-of-episode grader)
- **Alternatives**: Only a grader score; only per-step rewards; per-step rewards that are the grader score delta.
- **Chosen**: Independent per-step rewards (from `rewards.py`) AND a single grader score at episode end (from `grading.py`).
- **Rationale**: Per-step rewards give dense training signal for RL agents. The grader score gives a clean, interpretable submission metric. Decoupling them avoids reward hacking — a grader-delta reward would incentivise agents to hold off on triaging until they are certain, which is not realistic SRE behaviour.

### Decision: `link_alerts` does not mark alerts as triaged; `triage` and `skip` both do
- **Alternatives**: All three action types mark alerts as triaged; `skip` is triage-exempt.
- **Chosen**: `triage` and `skip` both set `alert.triaged = True` (reducing `pending_count`). `link_alerts` is grouping-only and does not.
- **Rationale**: `skip` is a deliberate decision ("this is a false alarm") and should consume the alert slot just like `triage`. `link_alerts` is metadata — it expresses correlation without resolving the alert. Agents must still `triage` or `skip` every alert to reach `done=True`.

### Decision: LLM history window capped at 6 turns in `inference.py`
- **Alternatives**: Full episode history; no history (stateless); 3 turns.
- **Chosen**: Last 6 assistant turns.
- **Rationale**: 6 turns gives the model context on recent decisions and feedback without exceeding typical context windows or dramatically increasing per-call token cost. Hard episodes have 30 alerts — feeding all prior actions would push 8k+ tokens per call.

### Decision: `python -m uvicorn` instead of bare `uvicorn` in all scripts
- **Alternatives**: Bare `uvicorn` command, `uvicorn` entry-point in PATH.
- **Chosen**: `python -m uvicorn` in Makefile, scripts/validate.sh, and README.
- **Rationale**: On Windows, pip-installed entry-point scripts are not always on PATH in every shell context (e.g. Git Bash, VS Code terminal). `python -m uvicorn` always works as long as the correct Python is active, regardless of shell PATH configuration.

---

## Template for future decisions

```
## YYYY-MM-DD — Agent/Human (Phase N)

### Decision: [Short title]
- **Alternatives**: [what else was considered]
- **Chosen**: [what was done]
- **Rationale**: [why]
```
