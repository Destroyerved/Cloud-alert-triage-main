#!/usr/bin/env bash
# scripts/validate.sh
# -------------------
# Local pre-submission validation script.
# Run from the cloud-alert-triage/ project root.
# Usage: bash scripts/validate.sh
#
# Checks performed:
#   1. Server starts on port 7860
#   2. GET /health returns 200
#   3. POST /reset (easy, seed=42) returns 200
#   4. POST /step (valid triage) returns 200
#   5. Server is stopped cleanly
#   6. openenv validate passes (if openenv-core is installed)
#   7. docker build succeeds (if Docker is available)

set -euo pipefail

PORT=7860
BASE="http://localhost:${PORT}"
PASS=0
FAIL=0
SERVER_PID=""

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
RESET="\033[0m"

pass() { echo -e "${GREEN}[PASS]${RESET} $1"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}[FAIL]${RESET} $1"; FAIL=$((FAIL + 1)); }
warn() { echo -e "${YELLOW}[WARN]${RESET} $1"; }
info() { echo -e "       $1"; }

cleanup() {
    if [ -n "${SERVER_PID}" ]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        info "Server (PID ${SERVER_PID}) stopped."
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 1. Start server
# ---------------------------------------------------------------------------
echo ""
echo "=== Cloud Alert Triage — Pre-Submission Validation ==="
echo ""

info "Starting uvicorn on port ${PORT}..."
python -m uvicorn server.app:app --host 0.0.0.0 --port "${PORT}" --log-level warning &
SERVER_PID=$!
sleep 3   # give server time to boot

# Check it's alive
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    fail "Server failed to start"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. GET /health
# ---------------------------------------------------------------------------
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${BASE}/health") || STATUS="000"
if [ "${STATUS}" = "200" ]; then
    pass "GET /health → 200"
else
    fail "GET /health → ${STATUS} (expected 200)"
fi

# ---------------------------------------------------------------------------
# 3. POST /reset
# ---------------------------------------------------------------------------
RESET_BODY='{"task_id":"easy","seed":42}'
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "${BASE}/reset" \
    -H "Content-Type: application/json" \
    -d "${RESET_BODY}") || STATUS="000"
if [ "${STATUS}" = "200" ]; then
    pass "POST /reset (easy, seed=42) → 200"
else
    fail "POST /reset → ${STATUS} (expected 200)"
fi

# ---------------------------------------------------------------------------
# 4. POST /step (valid triage — we grab the first alert_id dynamically)
# ---------------------------------------------------------------------------
RESET_RESP=$(curl -s -X POST "${BASE}/reset" \
    -H "Content-Type: application/json" \
    -d '{"task_id":"easy","seed":42}') || RESET_RESP="{}"

# Extract first alert_id using basic string parsing (no jq dependency)
ALERT_ID=$(echo "${RESET_RESP}" | grep -o '"alert_id":"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -z "${ALERT_ID}" ]; then
    ALERT_ID="alert-001"   # fallback
fi

STEP_BODY="{\"action_type\":\"triage\",\"alert_id\":\"${ALERT_ID}\",\"root_cause\":\"resource_exhaustion\",\"severity\":\"high\",\"remediation\":\"scale_up\"}"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "${BASE}/step" \
    -H "Content-Type: application/json" \
    -d "${STEP_BODY}") || STATUS="000"
if [ "${STATUS}" = "200" ]; then
    pass "POST /step (valid triage) → 200"
else
    fail "POST /step → ${STATUS} (expected 200)"
fi

# ---------------------------------------------------------------------------
# 5. Stop server
# ---------------------------------------------------------------------------
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
SERVER_PID=""
info "Server stopped."

# ---------------------------------------------------------------------------
# 6. openenv validate
# ---------------------------------------------------------------------------
if command -v openenv &>/dev/null; then
    if openenv validate 2>&1; then
        pass "openenv validate"
    else
        fail "openenv validate — check openenv.yaml"
    fi
else
    warn "openenv CLI not found — skipping 'openenv validate'"
    info "Install with: pip install openenv-core"
fi

# ---------------------------------------------------------------------------
# 7. Docker build
# ---------------------------------------------------------------------------
if command -v docker &>/dev/null; then
    info "Running docker build (this may take a minute)..."
    if docker build -t cloud-alert-triage . -q; then
        pass "docker build"
    else
        fail "docker build failed — check Dockerfile and requirements.txt"
    fi
else
    warn "Docker not found — skipping docker build check"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Summary ==="
echo -e "${GREEN}Passed: ${PASS}${RESET}   ${RED}Failed: ${FAIL}${RESET}"
echo ""
if [ "${FAIL}" -eq 0 ]; then
    echo -e "${GREEN}✅  All checks passed. Ready for submission.${RESET}"
    exit 0
else
    echo -e "${RED}❌  ${FAIL} check(s) failed. Fix before submitting.${RESET}"
    exit 1
fi
