#!/usr/bin/env bash
# Cold resume chain for MI355X 1x: 3 simulated-reclaim runs using clean 8B checkpoint.
# Deletes each replacement Droplet by ID after the timed measurement.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

RUN_META="${1:-$ROOT/logs/mi355x1/8B/clean-mi355x1-8B-20260930212438/run_meta.json}"
[[ -f "$RUN_META" ]] || { echo "missing run_meta: $RUN_META"; exit 1; }
RUN_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$RUN_META")
MODEL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$RUN_META")
ENDPOINT="https://${SPACES_REGION_MEM1}.digitaloceanspaces.com"
# Resume from last clean checkpoint (step 100)
SPACES_URI="s3://${SPACES_BUCKET_MEM1}/${RUN_ID}/ckpt-step-100"
SIZE="gpu-mi355x1-288gb-spot"
REGION="mem1"
IMAGE="gpu-amd-base"
GPU="mi355x"
LOG="$ROOT/logs/full_mi355x1_cold_resume.log"
mkdir -p "$(dirname "$LOG")"
echo "=== cold resume chain start $(date -u +%Y-%m-%dT%H:%M:%SZ) model=$MODEL ckpt=$SPACES_URI ===" | tee "$LOG"

for i in 1 2 3; do
  TS=$(date -u +%Y%m%d%H%M%S)
  NAME="spot-bench-resume-cold-mi355x1-${MODEL}-${i}-${TS}"
  EVENTS_DIR="$ROOT/logs/mi355x1/${MODEL}/resume-cold-${i}-${TS}"
  mkdir -p "$EVENTS_DIR"
  python3 - <<PY
import json, pathlib
meta=json.loads(open("$RUN_META").read())
meta.update({
  "variant": "cold",
  "resume_attempt": $i,
  "source_run_id": "$RUN_ID",
  "spaces_uri": "$SPACES_URI",
  "exclude_from_final_results": False,
})
pathlib.Path("$EVENTS_DIR/run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
PY
  echo "=== RESUME cold $i/3 name=$NAME ===" | tee -a "$LOG"
  set +e
  bash "$ROOT/scripts/measure_resume.sh" \
    --name "$NAME" \
    --size "$SIZE" \
    --region "$REGION" \
    --image "$IMAGE" \
    --gpu "$GPU" \
    --spaces-uri "$SPACES_URI" \
    --spaces-endpoint "$ENDPOINT" \
    --events-dir "$EVENTS_DIR" \
    --model "$MODEL" \
    --run-meta "$RUN_META" \
    2>&1 | tee -a "$LOG" | tee "$EVENTS_DIR/driver.log"
  RC=${PIPESTATUS[0]}
  set -e
  DID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("droplet_id",""))' "$EVENTS_DIR/resume_result.json" 2>/dev/null || true)
  if [[ -z "$DID" ]]; then
    DID=$(grep -E '^RESUME_DROPLET_ID=' "$EVENTS_DIR/driver.log" 2>/dev/null | tail -1 | cut -d= -f2 || true)
  fi
  if [[ -n "$DID" ]]; then
    echo "Deleting resume Droplet id=$DID" | tee -a "$LOG"
    doctl compute droplet delete "$DID" --force || true
  fi
  if [[ $RC -ne 0 ]]; then
    echo "RESUME_ATTEMPT_${i}_FAILED rc=$RC" | tee -a "$LOG"
    # continue remaining attempts unless create failed with no capacity forever
  else
    echo "RESUME_ATTEMPT_${i}_OK" | tee -a "$LOG"
  fi
done

python3 "$ROOT/scripts/analyze.py" --root "$ROOT" | tee -a "$LOG"
echo COLD_RESUME_CHAIN_DONE | tee -a "$LOG"
