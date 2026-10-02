#!/usr/bin/env bash
# Ablation: like-for-like deterministic train, then resume after stripping the
# post-995a62d meta fields (python/numpy RNG + optimizer_state_steps sample).
# Confirms whether old checkpoint format still bit-exact matches steps 81-100.
# Usage: bash scripts/run_b300_det_old_meta_ablation.sh [--droplet-id ID --ip IP]
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
OUT="$ROOT/logs/b300x1/8B/det-old-meta-ablation-$(date -u +%Y%m%d%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/driver.log"
BATCH_SIZE=4
GRAD_ACCUM=1
STRIP_KEYS='python_random_state python_random_version numpy_rng_bit_generator_state numpy_rng_pos optimizer_state_steps'

DID=""
IP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --droplet-id) DID="$2"; shift 2 ;;
    --ip) IP="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

echo "=== det old-meta ablation out=$OUT ===" | tee "$LOG"
(
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/batch_heartbeat"
    echo "det_old_meta_ablation droplet=${DID:-pending}" > "$ROOT/logs/active_run.json"
    sleep 30
  done
) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true; rm -f "$ROOT/logs/active_run.json"' EXIT

if [[ -z "$DID" ]]; then
  NAME="spot-bench-b300-oldmeta-$(date -u +%Y%m%d%H%M%S)"
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
echo "READY id=$DID ip=$IP" | tee -a "$LOG"
echo "det_old_meta_ablation droplet=$DID" > "$ROOT/logs/active_run.json"

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

ORIG_EVENTS=/opt/bench/det_oldmeta_orig_events.jsonl
ORIG_CKPT="$SCRATCH/bench/det_oldmeta_orig"
echo "=== PHASE1 deterministic train (new meta format) ===" | tee -a "$LOG"
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
test -d "${ORIG_CKPT}/ckpt-step-80"
REMOTE

"${SCP[@]}" "root@$IP:$ORIG_CKPT/ckpt-step-80/meta.json" "$OUT/ckpt80_meta_new.json"
"${SSH[@]}" "root@$IP" "cp -a $ORIG_CKPT/ckpt-step-80 $SCRATCH/bench/det_oldmeta_ckpt80_new"
"${SSH[@]}" "root@$IP" "cp -a $ORIG_CKPT/ckpt-step-80 $SCRATCH/bench/det_oldmeta_ckpt80_oldfmt"

# Strip new fields to simulate pre-995a62d checkpoint meta
echo "=== STRIP meta to old format keys ===" | tee -a "$LOG"
"${SSH[@]}" "root@$IP" bash -s <<REMOTE | tee -a "$LOG"
set -euo pipefail
python3 - <<'PY'
import json
from pathlib import Path
p = Path("$SCRATCH/bench/det_oldmeta_ckpt80_oldfmt/meta.json")
meta = json.loads(p.read_text())
strip = [s for s in "$STRIP_KEYS".split() if s]
removed = {k: meta.pop(k) for k in list(meta) if k in strip}
kept = sorted(meta.keys())
p.write_text(json.dumps(meta, indent=2))
print("stripped:", sorted(removed.keys()))
print("kept:", kept)
print("meta_bytes_after=", p.stat().st_size)
PY
# also record sizes of new vs old meta and full ckpt dirs
du -sb "$SCRATCH/bench/det_oldmeta_ckpt80_new" "$SCRATCH/bench/det_oldmeta_ckpt80_oldfmt"
wc -c "$SCRATCH/bench/det_oldmeta_ckpt80_new/meta.json" "$SCRATCH/bench/det_oldmeta_ckpt80_oldfmt/meta.json"
REMOTE
"${SCP[@]}" "root@$IP:$SCRATCH/bench/det_oldmeta_ckpt80_oldfmt/meta.json" "$OUT/ckpt80_meta_oldfmt.json"

RES_EVENTS=/opt/bench/det_oldmeta_resume_events.jsonl
RES_CKPT="$SCRATCH/bench/det_oldmeta_resume"
echo "=== PHASE2 resume from OLD-FORMAT meta (torch+cuda RNG only) ===" | tee -a "$LOG"
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
      --resume-from ${SCRATCH}/spot-bench/det_oldmeta_ckpt80_oldfmt \
      --max-steps 21 --num-ckpts 0 --log-every 1 \
      --warmup-steps 0 --steady-steps 20 --ckpt-every 9999"
REMOTE

"${SCP[@]}" "root@$IP:$ORIG_EVENTS" "$OUT/orig_events.jsonl"
"${SCP[@]}" "root@$IP:$RES_EVENTS" "$OUT/resume_events.jsonl"

python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path
out = Path("$OUT")

def losses(path):
    d = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("event") == "step" and "loss" in ev and "step" in ev:
            d[int(ev["step"])] = float(ev["loss"])
    return d

orig = losses(out / "orig_events.jsonl")
res = losses(out / "resume_events.jsonl")
mismatches = []
for s in range(81, 101):
    if s not in orig or s not in res:
        mismatches.append({"step": s, "reason": "missing", "orig": orig.get(s), "resume": res.get(s)})
        continue
    if orig[s] != res[s]:
        mismatches.append({"step": s, "orig": orig[s], "resume": res[s], "delta": res[s] - orig[s]})

meta_new = json.loads((out / "ckpt80_meta_new.json").read_text())
meta_old = json.loads((out / "ckpt80_meta_oldfmt.json").read_text())
new_fields = sorted(set(meta_new) - set(meta_old))
result = {
    "method": "old_meta_ablation_strip_python_numpy_optimizer_sample",
    "stripped_fields": new_fields,
    "kept_rng_fields": [k for k in ("torch_rng_hex", "cuda_rng_hex") if k in meta_old],
    "compare_range": [81, 100],
    "bit_exact": len(mismatches) == 0,
    "first_diverge": mismatches[0]["step"] if mismatches else None,
    "mismatch_count": len(mismatches),
    "mismatches_head": mismatches[:5],
    "meta_new_bytes": (out / "ckpt80_meta_new.json").stat().st_size,
    "meta_oldfmt_bytes": (out / "ckpt80_meta_oldfmt.json").stat().st_size,
    "meta_delta_bytes": (out / "ckpt80_meta_new.json").stat().st_size - (out / "ckpt80_meta_oldfmt.json").stat().st_size,
}
(out / "compare.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if result["bit_exact"]:
    print("OLD_META_BITEXACT_PASS")
else:
    print("OLD_META_BITEXACT_FAIL")
PY

# Delete droplet (watchdog backup)
echo "DELETING droplet=$DID" | tee -a "$LOG"
bash "$ROOT/scripts/provision.sh" delete --id "$DID" >>"$LOG" 2>&1 || \
  doctl compute droplet delete "$DID" --force >>"$LOG" 2>&1 || true
rm -f "$ROOT/logs/active_run.json"
echo "OLD_META_ABLATION_DONE out=$OUT droplet=$DID" | tee -a "$LOG"
