#!/usr/bin/env bash
# Like-for-like B300 deterministic resume test.
# 1) Train 8B from scratch in deterministic mode to step 100, saving ckpt-80.
# 2) Resume from that ckpt-80 in deterministic mode to step 100.
# 3) Steps 81-100 must match bit-exactly (no tolerance).
# Usage: bash scripts/run_b300_det_resume_bitexact.sh [--droplet-id ID --ip IP]
set -euo pipefail
ROOT="."
cd "$ROOT"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes)
SCP=(scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes)
SIZE="gpu-b300x1-288gb-lc-spot"
REGION="mkc1"
IMAGE="gpu-h100x1-base"
OUT="$ROOT/logs/b300x1/8B/det-resume-bitexact-$(date -u +%Y%m%d%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/driver.log"
BATCH_SIZE=4
GRAD_ACCUM=1

DID=""
IP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --droplet-id) DID="$2"; shift 2 ;;
    --ip) IP="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

echo "=== det resume bitexact out=$OUT ===" | tee "$LOG"
# Heartbeat so watchdog knows this run is active
(
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/batch_heartbeat"
    echo "det_bitexact droplet=${DID:-pending}" > "$ROOT/logs/active_run.json"
    sleep 30
  done
) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true; rm -f "$ROOT/logs/active_run.json"' EXIT

if [[ -z "$DID" ]]; then
  NAME="spot-bench-b300-det-bitexact-$(date -u +%Y%m%d%H%M%S)"
  attempt=0; max_attempts=40; sleep_s=90
  while [[ $attempt -lt $max_attempts ]]; do
    attempt=$((attempt+1))
    try_name="$NAME"
    [[ $attempt -gt 1 ]] && try_name="${NAME}-r${attempt}-$(date -u +%H%M%S)"
    echo "CREATE_ATTEMPT $attempt/$max_attempts name=$try_name" | tee -a "$LOG"
    bash "$ROOT/scripts/provision.sh" create \
      --name "$try_name" --size "$SIZE" --region "$REGION" --image "$IMAGE" \
      > "$OUT/create.out" 2>&1 || true
    cat "$OUT/create.out" >> "$LOG"
    DID=$(awk -F= '/^CREATED_ID=/{print $2}' "$OUT/create.out" | tail -1)
    if [[ -n "$DID" ]]; then
      echo "CREATED_OK id=$DID" | tee -a "$LOG"
      break
    fi
    if grep -qiE 'not available in this region|DropletLimit|capacity|currently unavailable' "$OUT/create.out"; then
      echo "CAPACITY_WAIT sleep ${sleep_s}s" | tee -a "$LOG"
      sleep "$sleep_s"
      continue
    fi
    echo "CREATE_FAILED"; exit 3
  done
  [[ -n "$DID" ]] || { echo "CREATE_CAPACITY_EXHAUSTED"; exit 3; }
  IP=$(bash "$ROOT/scripts/provision.sh" wait-ssh --id "$DID" --gpu-tool nvidia-smi 2>>"$LOG" \
    | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/{ip=$0} END{print ip}')
  [[ -n "$IP" ]] || { echo "NO_IP"; exit 4; }
else
  [[ -n "$IP" ]] || IP=$(doctl compute droplet get "$DID" --format PublicIPv4 --no-header)
fi
echo "$DID" > "$OUT/droplet_id.txt"
echo "$IP" > "$OUT/ip.txt"
echo "REUSE/READY id=$DID ip=$IP" | tee -a "$LOG"
echo "det_bitexact droplet=$DID" > "$ROOT/logs/active_run.json"

"${SSH[@]}" "root@$IP" 'mkdir -p /opt/bench-scripts /opt/bench/snapshot'
"${SCP[@]}" "$ROOT/scripts/"*.py "$ROOT/scripts/"*.sh "root@$IP:/opt/bench-scripts/"
"${SSH[@]}" "root@$IP" bash -s <<'REMOTE' 2>&1 | tee -a "$LOG"
set -euo pipefail
if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sh; fi
if ! command -v s5cmd >/dev/null; then
  curl -sL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin s5cmd
fi
bash /opt/bench-scripts/setup_node.sh --gpu b300 --snapshot-dir /opt/bench/snapshot
REMOTE
CONTAINER=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/container.ref)
SCRATCH=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/scratch.path)

# --- Phase 1: deterministic from-scratch to step 100, save ckpts at 60/80/100 ---
ORIG_EVENTS=/opt/bench/det_bitexact_orig_events.jsonl
ORIG_CKPT="$SCRATCH/bench/det_bitexact_orig"
echo "=== PHASE1 original deterministic train ===" | tee -a "$LOG"
"${SSH[@]}" "root@$IP" \
  CONTAINER="$CONTAINER" SCRATCH="$SCRATCH" ORIG_EVENTS="$ORIG_EVENTS" ORIG_CKPT="$ORIG_CKPT" \
  BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
  bash -s <<'REMOTE' | tee -a "$LOG"
set -euo pipefail
rm -rf "$ORIG_CKPT" "$ORIG_EVENTS"
mkdir -p "$ORIG_CKPT"
docker run --rm --network=host --ipc=host --gpus all \
  -v /opt/bench-scripts:/scripts \
  -v "$SCRATCH":"$SCRATCH" -v /opt/bench:/opt/bench \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$CONTAINER" bash -lc "pip install -q 'transformers>=4.51.0' && \
    export CUBLAS_WORKSPACE_CONFIG=:4096:8 && \
    torchrun --standalone --nproc_per_node=1 /scripts/train_bench.py \
      --model 8B --master-dtype fp32 --seed 42 --deterministic \
      --batch-size ${BATCH_SIZE} --grad-accum ${GRAD_ACCUM} \
      --events ${ORIG_EVENTS} \
      --ckpt-dir ${ORIG_CKPT} \
      --max-steps 101 --num-ckpts 3 \
      --warmup-steps 10 --steady-steps 50 --ckpt-every 20 \
      --log-every 1"
echo PHASE1_DONE
test -d "${ORIG_CKPT}/ckpt-step-80"
test -d "${ORIG_CKPT}/ckpt-step-100"
REMOTE
"${SCP[@]}" "root@$IP:$ORIG_EVENTS" "$OUT/orig_events.jsonl"
"${SCP[@]}" "root@$IP:$ORIG_CKPT/ckpt-step-80/meta.json" "$OUT/ckpt80_meta.json"

# Snapshot live-comparable state dump at ckpt-80 for diagnostics
"${SSH[@]}" "root@$IP" "cp -a $ORIG_CKPT/ckpt-step-80 $SCRATCH/bench/det_bitexact_ckpt80"

# --- Phase 2: deterministic resume from that ckpt-80 ---
RES_EVENTS=/opt/bench/det_bitexact_resume_events.jsonl
RES_CKPT="$SCRATCH/bench/det_bitexact_resume"
echo "=== PHASE2 deterministic resume from ckpt-80 ===" | tee -a "$LOG"
"${SSH[@]}" "root@$IP" \
  CONTAINER="$CONTAINER" SCRATCH="$SCRATCH" RES_EVENTS="$RES_EVENTS" RES_CKPT="$RES_CKPT" \
  BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
  bash -s <<'REMOTE' | tee -a "$LOG"
set -euo pipefail
rm -rf "$RES_CKPT" "$RES_EVENTS"
mkdir -p "$RES_CKPT"
docker run --rm --network=host --ipc=host --gpus all \
  -v /opt/bench-scripts:/scripts \
  -v "$SCRATCH":"$SCRATCH" -v /opt/bench:/opt/bench \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$CONTAINER" bash -lc "pip install -q 'transformers>=4.51.0' && \
    export CUBLAS_WORKSPACE_CONFIG=:4096:8 && \
    torchrun --standalone --nproc_per_node=1 /scripts/train_bench.py \
      --model 8B --master-dtype fp32 --seed 42 --deterministic \
      --batch-size ${BATCH_SIZE} --grad-accum ${GRAD_ACCUM} \
      --events ${RES_EVENTS} \
      --ckpt-dir ${RES_CKPT} \
      --resume-from ${SCRATCH}/spot-bench/det_bitexact_ckpt80 \
      --max-steps 21 --num-ckpts 0 --log-every 1 \
      --warmup-steps 0 --steady-steps 0"
echo PHASE2_DONE
REMOTE
"${SCP[@]}" "root@$IP:$RES_EVENTS" "$OUT/resume_events.jsonl"

# --- Compare steps 81-100 bit-exactly ---
python3 - "$OUT" <<'PY' | tee -a "$LOG" | tee "$OUT/compare.json"
import json, sys
from pathlib import Path
out=Path(sys.argv[1])

def losses(path):
  by={}
  for line in path.read_text().splitlines():
    if not line.strip(): continue
    e=json.loads(line)
    if e.get("event")=="step":
      by[int(e["step"])]=float(e["loss"])
  return by

orig=losses(out/"orig_events.jsonl")
res=losses(out/"resume_events.jsonl")
meta=json.loads((out/"ckpt80_meta.json").read_text())

mismatches=[]
for step in range(81, 101):
  o=orig.get(step); r=res.get(step)
  if o is None or r is None:
    mismatches.append({"step": step, "orig": o, "resume": r, "reason": "missing"})
  elif o != r:  # bit-exact float equality
    mismatches.append({"step": step, "orig": o, "resume": r, "delta": abs(o-r), "reason": "value"})

result={
  "method": "like_for_like_deterministic_train_ckpt80_resume",
  "deterministic": True,
  "cublas_workspace_config": ":4096:8",
  "seed": 42,
  "batch_size": 4,
  "ckpt80_meta_step": meta.get("step"),
  "ckpt80_meta_loss": meta.get("loss"),
  "orig_steps": sorted(orig),
  "resume_steps": sorted(res),
  "compare_range": [81, 100],
  "bit_exact": len(mismatches)==0,
  "first_diverge": mismatches[0] if mismatches else None,
  "mismatch_count": len(mismatches),
  "mismatches_head": mismatches[:5],
}
print(json.dumps(result, indent=2))
(out/"compare.json").write_text(json.dumps(result, indent=2)+"\n")
if mismatches:
  print("BITEXACT_FAIL")
  raise SystemExit(10)
print("BITEXACT_PASS")
PY

echo "DET_BITEXACT_DONE out=$OUT droplet=$DID" | tee -a "$LOG"
if [[ "${DELETE_AFTER:-1}" == "1" ]]; then
  bash "$ROOT/scripts/provision.sh" delete --id "$DID" || true
fi
