#!/usr/bin/env bash
# One cold resume create+measure to /scratch. Ask operator yes before each invoke.
# Usage: run_one_cold_resume_scratch.sh <attempt 1|2|3>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

ATTEMPT="${1:?usage: $0 <1|2|3>}"
RUN_META="$ROOT/logs/mi355x1/8B/clean-scratch-mi355x1-8B-20261001081617/run_meta.json"
[[ -f "$RUN_META" ]] || { echo "missing $RUN_META"; exit 1; }
RUN_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$RUN_META")
MODEL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$RUN_META")
ENDPOINT="https://${SPACES_REGION_MEM1}.digitaloceanspaces.com"
SPACES_URI="s3://${SPACES_BUCKET_MEM1}/${RUN_ID}/ckpt-step-100"
SIZE="gpu-mi355x1-288gb-spot"
REGION="mem1"
IMAGE="gpu-amd-base"
GPU="mi355x"
TS=$(date -u +%Y%m%d%H%M%S)
NAME="spot-bench-resume-cold-scratch-mi355x1-${MODEL}-${ATTEMPT}-${TS}"
EVENTS_DIR="$ROOT/logs/mi355x1/${MODEL}/resume-cold-scratch-${ATTEMPT}-${TS}"
mkdir -p "$EVENTS_DIR"
python3 - <<PY
import json, pathlib
meta=json.loads(open("$RUN_META").read())
meta.update({
  "variant": "cold",
  "resume_attempt": int("$ATTEMPT"),
  "source_run_id": "$RUN_ID",
  "spaces_uri": "$SPACES_URI",
  "storage": "scratch",
  "exclude_from_final_results": False,
  "exclude_reason": "",
  "note": "cold resume download+ckpt on /scratch DOSCRATCH; source clean-scratch ckpt-100",
})
# drop boot-disk-only fields that would confuse analyze
meta.pop("steady_step_time_s", None)
pathlib.Path("$EVENTS_DIR/run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
print("wrote", "$EVENTS_DIR/run_meta.json")
PY

echo "=== COLD SCRATCH RESUME $ATTEMPT name=$NAME ckpt=$SPACES_URI ==="
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
  2>&1 | tee "$EVENTS_DIR/driver.log"
RC=${PIPESTATUS[0]}
set -e

# Enrich run_meta from resume_result
python3 - <<PY
import json
from pathlib import Path
d=Path("$EVENTS_DIR")
meta=json.loads((d/"run_meta.json").read_text())
rp=d/"resume_result.json"
if rp.exists():
  r=json.loads(rp.read_text())
  for k in ("image_size_bytes","steps_lost","step_time_s","R_s","loss_match","loss_match_strict","storage"):
    if k in r and r[k] not in ("", None):
      meta[k]=r[k]
  meta["storage"]="scratch"
  meta["exclude_from_final_results"]=False
(d/"run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
PY

DID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("droplet_id",""))' "$EVENTS_DIR/resume_result.json" 2>/dev/null || true)
if [[ -z "$DID" ]]; then
  DID=$(grep -E '^RESUME_DROPLET_ID=' "$EVENTS_DIR/driver.log" 2>/dev/null | tail -1 | cut -d= -f2 || true)
fi
if [[ -n "$DID" ]]; then
  echo "Deleting resume Droplet id=$DID"
  doctl compute droplet delete "$DID" --force || true
fi
python3 "$ROOT/scripts/analyze.py" --root "$ROOT"
echo "COLD_SCRATCH_RESUME_${ATTEMPT}_RC=$RC dir=$EVENTS_DIR"
exit "$RC"
