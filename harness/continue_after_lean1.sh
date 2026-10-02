#!/usr/bin/env bash
# Wait for orphaned lean1 measure, then continue lean2-3 + strict + cold4.
set -euo pipefail
ROOT="."
cd "$ROOT"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

LOG="$ROOT/logs/lean_recover_then_strict.nohup"
echo "=== continue_after_lean1 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

(
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/batch_heartbeat"
    sleep 60
  done
) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

LEAN1_DIR=logs/mi355x1/8B/resume-lean-scratch-1-20261001141701
LEAN1_DID="<droplet-id>"

for i in $(seq 1 240); do
  date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/batch_heartbeat"
  if [[ -f "$LEAN1_DIR/resume_result.json" ]]; then
    if ! pgrep -f 'measure_resume_existing.sh.*<droplet-id>' >/dev/null 2>&1; then
      echo "lean1 result ready iter=$i" | tee -a "$LOG"
      break
    fi
  fi
  if ! pgrep -f 'measure_resume_existing.sh.*<droplet-id>' >/dev/null 2>&1; then
    if [[ -f "$LEAN1_DIR/resume_result.json" ]]; then
      break
    fi
    echo "ANOMALY_STOP: lean1 measure exited without result" | tee -a "$LOG"
    exit 3
  fi
  sleep 15
done
[[ -f "$LEAN1_DIR/resume_result.json" ]] || { echo ANOMALY_STOP: lean1 timeout; exit 3; }

# Reuse check/delete/lean2-3/cold4 from main script by invoking a trimmed runner
# shellcheck disable=SC1091
# Inline continue: source functions by running the main script from a marker is hard;
# call the main script which skips lean1 if result exists.
exec bash "$ROOT/scripts/run_lean_recover_then_strict.sh"
