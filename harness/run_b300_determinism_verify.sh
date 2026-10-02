#!/usr/bin/env bash
# B300 8B from-scratch determinism check on one Droplet.
# Runs: default A1, default A2, deterministic B1, deterministic B2 to step 100.
# Then optional ckpt-80 continuation under deterministic mode if B matches.
# Usage: bash scripts/run_b300_determinism_verify.sh [--droplet-id ID --ip IP]
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
OUT="$ROOT/logs/b300x1/8B/determinism-verify-$(date -u +%Y%m%d%H%M%S)"
mkdir -p "$OUT"
LOG="$OUT/driver.log"
CLEAN_META="$ROOT/logs/b300x1/8B/clean-scratch-b300x1-8B-20261001195841/run_meta.json"
CLEAN_RUN_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$CLEAN_META")
BATCH_SIZE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["batch_size"])' "$CLEAN_META")
GRAD_ACCUM=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("grad_accum",1))' "$CLEAN_META")
ENDPOINT="https://${SPACES_REGION_MKC1}.digitaloceanspaces.com"
CKPT80="s3://${SPACES_BUCKET_MKC1}/${CLEAN_RUN_ID}/ckpt-step-80"

DID=""
IP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --droplet-id) DID="$2"; shift 2 ;;
    --ip) IP="$2"; shift 2 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done

echo "=== determinism verify out=$OUT ===" | tee "$LOG"

if [[ -z "$DID" ]]; then
  NAME="spot-bench-b300-determinism-$(date -u +%Y%m%d%H%M%S)"
  echo "CREATE $NAME" | tee -a "$LOG"
  CREATE_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  attempt=0
  max_attempts=40
  sleep_s=90
  while [[ $attempt -lt $max_attempts ]]; do
    attempt=$((attempt+1))
    try_name="$NAME"
    if [[ $attempt -gt 1 ]]; then
      try_name="${NAME}-r${attempt}-$(date -u +%H%M%S)"
    fi
    echo "CREATE_ATTEMPT $attempt/$max_attempts name=$try_name" | tee -a "$LOG"
    bash "$ROOT/scripts/provision.sh" create \
      --name "$try_name" --size "$SIZE" --region "$REGION" --image "$IMAGE" \
      > "$OUT/create.out" 2>&1 || true
    cat "$OUT/create.out" >> "$LOG"
    DID=$(awk -F= '/^CREATED_ID=/{print $2}' "$OUT/create.out" | tail -1)
    if [[ -n "$DID" ]]; then
      echo "CREATED_OK id=$DID attempt=$attempt" | tee -a "$LOG"
      break
    fi
    if grep -qiE 'not available in this region|DropletLimit|capacity|currently unavailable' "$OUT/create.out"; then
      echo "CAPACITY_WAIT sleep ${sleep_s}s (attempt $attempt)" | tee -a "$LOG"
      sleep "$sleep_s"
      continue
    fi
    echo "CREATE_FAILED non-capacity" | tee -a "$LOG"
    exit 3
  done
  [[ -n "$DID" ]] || { echo "CREATE_CAPACITY_EXHAUSTED"; exit 3; }
  IP=$(bash "$ROOT/scripts/provision.sh" wait-ssh --id "$DID" --gpu-tool nvidia-smi 2>>"$LOG" \
    | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/{ip=$0} END{print ip}')
  [[ -n "$IP" ]] || { echo "NO_IP"; exit 4; }
  echo "CREATED_OK id=$DID ip=$IP create_start=$CREATE_START" | tee -a "$LOG"
else
  [[ -n "$IP" ]] || IP=$(doctl compute droplet get "$DID" --format PublicIPv4 --no-header)
  echo "REUSE id=$DID ip=$IP" | tee -a "$LOG"
fi

echo "$DID" > "$OUT/droplet_id.txt"
echo "$IP" > "$OUT/ip.txt"

# Push scripts; ensure docker/s5cmd; run setup_node on the Droplet (remote-only).
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

# Scratch verification: lsblk/df/dd/fio while we have the Droplet.
"${SSH[@]}" "root@$IP" bash -s <<'REMOTE' | tee "$OUT/scratch_probe.txt" | tee -a "$LOG"
set -euo pipefail
echo '=== df -h ==='
df -h
echo '=== lsblk ==='
lsblk
echo '=== findmnt /scratch ==='
findmnt /scratch || true
echo '=== dd write /scratch ==='
dd if=/dev/zero of=/scratch/bench_dd_probe.bin bs=1M count=256 oflag=direct 2>&1
rm -f /scratch/bench_dd_probe.bin
echo '=== fio if available ==='
if ! command -v fio >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq fio >/dev/null
fi
fio --name=scratch_probe --directory=/scratch --rw=write --bs=1M --size=1G --iodepth=32 --direct=1 --numjobs=1 --group_reporting
rm -f /scratch/scratch_probe.*
echo '=== nproc / mem ==='
nproc
free -h
REMOTE

run_train() {
  local label="$1" det_flag="$2"
  local remote_events="/opt/bench/det_${label}_events.jsonl"
  local ckpt_dir="$SCRATCH/bench/det_${label}"
  echo "=== TRAIN $label deterministic=$det_flag ===" | tee -a "$LOG"
  "${SSH[@]}" "root@$IP" \
    CONTAINER="$CONTAINER" SCRATCH="$SCRATCH" LABEL="$label" DET_FLAG="$det_flag" \
    BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" REMOTE_EVENTS="$remote_events" \
    CKPT_DIR="$ckpt_dir" \
    bash -s <<'REMOTE' | tee -a "$LOG"
set -euo pipefail
rm -rf "$CKPT_DIR" "$REMOTE_EVENTS"
mkdir -p "$CKPT_DIR"
if [[ "$DET_FLAG" == "1" ]]; then
  DET_ARGS="--deterministic"
  EXTRA_ENV=(-e CUBLAS_WORKSPACE_CONFIG=:4096:8)
  INNER_EXPORT="export CUBLAS_WORKSPACE_CONFIG=:4096:8;"
else
  DET_ARGS=""
  EXTRA_ENV=()
  INNER_EXPORT=""
fi
# shellcheck disable=SC2086
docker run --rm --network=host --ipc=host --gpus all \
  -v /opt/bench-scripts:/scripts \
  -v "$SCRATCH":"$SCRATCH" -v /opt/bench:/opt/bench \
  "${EXTRA_ENV[@]}" \
  "$CONTAINER" bash -lc "pip install -q 'transformers>=4.51.0' && \
    ${INNER_EXPORT} \
    torchrun --standalone --nproc_per_node=1 /scripts/train_bench.py \
      --model 8B --master-dtype fp32 --seed 42 \
      --batch-size ${BATCH_SIZE} --grad-accum ${GRAD_ACCUM} \
      --events ${REMOTE_EVENTS} \
      --ckpt-dir ${CKPT_DIR} \
      --max-steps 100 --num-ckpts 0 \
      --warmup-steps 10 --steady-steps 90 \
      --log-every 1 \
      ${DET_ARGS}"
echo "TRAIN_DONE label=$LABEL"
REMOTE
  "${SCP[@]}" "root@$IP:$remote_events" "$OUT/${label}_events.jsonl"
  python3 - <<PY
import json
from pathlib import Path
ev=[json.loads(l) for l in Path("$OUT/${label}_events.jsonl").read_text().splitlines() if l.strip()]
by={e["step"]: e["loss"] for e in ev if e.get("event")=="step"}
print("LOSS_${label}_STEP100", by.get(100))
Path("$OUT/${label}_loss100.txt").write_text(str(by.get(100))+"\n")
meta={"label":"$label","deterministic": bool(int("$det_flag")),
      "loss_step_100": by.get(100), "steps": sorted(by)}
Path("$OUT/${label}_summary.json").write_text(json.dumps(meta, indent=2)+"\n")
PY
}

run_train A1 0
run_train A2 0
run_train B1 1
run_train B2 1

python3 - "$OUT" <<'PY' | tee -a "$LOG"
import json, sys
from pathlib import Path
out=Path(sys.argv[1])
def loss(label):
  p=out/f"{label}_summary.json"
  return json.loads(p.read_text())["loss_step_100"]
a1,a2,b1,b2=loss("A1"),loss("A2"),loss("B1"),loss("B2")
def within(a,b,tol=1e-3):
  if a is None or b is None: return False
  return abs(float(a)-float(b)) <= max(tol, 1e-5*abs(float(b)))
result={
  "A1_loss_100": a1, "A2_loss_100": a2,
  "B1_loss_100": b1, "B2_loss_100": b2,
  "A_run_to_run_delta": None if None in (a1,a2) else abs(float(a1)-float(a2)),
  "B_run_to_run_delta": None if None in (b1,b2) else abs(float(b1)-float(b2)),
  "A_run_to_run_match_1e3": within(a1,a2),
  "B_bit_exact": (b1 is not None and b2 is not None and float(b1)==float(b2)),
}
a_delta = result["A_run_to_run_delta"]
if a_delta is not None and a_delta > 1e-3 and result["B_bit_exact"]:
  result["verdict"] = "KERNEL_NONDETERMINISM"
elif a_delta is not None and a_delta > 1e-3 and not result["B_bit_exact"]:
  result["verdict"] = "INCONCLUSIVE_OR_RESUME_BUG_RISK"
elif a_delta is not None and a_delta <= 1e-3:
  result["verdict"] = "DEFAULT_ALREADY_STABLE_CHECK_RESUME"
else:
  result["verdict"] = "MISSING_LOSSES"
print(json.dumps(result, indent=2))
(out/"compare.json").write_text(json.dumps(result, indent=2)+"\n")
PY

VERDICT=$(python3 -c 'import json; print(json.load(open("'"$OUT"'/compare.json"))["verdict"])')
echo "VERDICT=$VERDICT" | tee -a "$LOG"

if [[ "$VERDICT" == "KERNEL_NONDETERMINISM" ]]; then
  echo "=== deterministic ckpt-80 continuation ===" | tee -a "$LOG"
  # shellcheck source=/dev/null
  source "$ROOT/scripts/spaces_env_remote.sh"
  spaces_env_push "$IP"
  CKPT_LOCAL="$SCRATCH/bench/ref-ckpt-80-det"
  "${SSH[@]}" "root@$IP" bash -s <<REMOTE
set -euo pipefail
set -a; source /opt/bench/.spaces.env; set +a
rm -rf '$CKPT_LOCAL'; mkdir -p '$CKPT_LOCAL'
s5cmd --endpoint-url '$ENDPOINT' cp '${CKPT80%/}/*' '$CKPT_LOCAL/'
REMOTE
  REMOTE_EVENTS=/opt/bench/strict_det_events.jsonl
  "${SSH[@]}" "root@$IP" \
    CONTAINER="$CONTAINER" SCRATCH="$SCRATCH" CKPT_LOCAL="$CKPT_LOCAL" \
    BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" REMOTE_EVENTS="$REMOTE_EVENTS" \
    bash -s <<'REMOTE' | tee -a "$LOG"
set -euo pipefail
rm -f "$REMOTE_EVENTS"
docker run --rm --network=host --ipc=host --gpus all \
  -v /opt/bench-scripts:/scripts \
  -v /usr/local/bin/s5cmd:/host-bin/s5cmd:ro \
  -v "$SCRATCH":"$SCRATCH" -v /opt/bench:/opt/bench \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$CONTAINER" bash -lc "pip install -q 'transformers>=4.51.0' && \
    export CUBLAS_WORKSPACE_CONFIG=:4096:8 && \
    torchrun --standalone --nproc_per_node=1 /scripts/train_bench.py \
      --model 8B --master-dtype fp32 --seed 42 --deterministic \
      --batch-size ${BATCH_SIZE} --grad-accum ${GRAD_ACCUM} \
      --events ${REMOTE_EVENTS} \
      --ckpt-dir ${SCRATCH}/bench/ckpts_ref_det \
      --resume-from ${CKPT_LOCAL} \
      --max-steps 21 --num-ckpts 0 --log-every 1 \
      --warmup-steps 0 --steady-steps 0"
echo REF_DET_DONE
REMOTE
  spaces_env_clear "$IP"
  "${SCP[@]}" "root@$IP:$REMOTE_EVENTS" "$OUT/strict_det_events.jsonl"
  python3 - <<PY | tee -a "$LOG" | tee "$OUT/strict_det_result.json"
import json
from pathlib import Path
out=Path("$OUT")
root=Path("$ROOT")
ev=[json.loads(l) for l in (out/"strict_det_events.jsonl").read_text().splitlines() if l.strip()]
by={e["step"]: e["loss"] for e in ev if e.get("event")=="step"}
orig={}
for line in (root/"logs/b300x1/8B/clean-scratch-b300x1-8B-20261001195841/events.jsonl").read_text().splitlines():
  e=json.loads(line)
  if e.get("event")=="step":
    orig[int(e["step"])]=float(e["loss"])
tol=1e-3
def within(a,b):
  return abs(float(a)-float(b)) <= max(tol, 1e-5*abs(float(b)))
ref100=by.get(100); orig100=orig.get(100)
ok = ref100 is not None and orig100 is not None and within(ref100, orig100)
result={"reference_loss_step_100": ref100, "original_loss_step_100": orig100,
        "anchor_match_step_100": ok, "delta": None if None in (ref100,orig100) else abs(float(ref100)-float(orig100)),
        "mode": "deterministic_ckpt80_continuation"}
print(json.dumps(result, indent=2))
(out/"strict_det_result.json").write_text(json.dumps(result, indent=2)+"\n")
if not ok:
  raise SystemExit("DETERMINISTIC_ANCHOR_STILL_FAILS_POSSIBLE_RESUME_BUG")
print("DETERMINISTIC_ANCHOR_OK")
PY
fi

echo "DETERMINISM_VERIFY_DONE out=$OUT droplet=$DID" | tee -a "$LOG"
# Keep droplet for operator unless DELETE=1
if [[ "${DELETE_AFTER:-0}" == "1" ]]; then
  bash "$ROOT/scripts/provision.sh" delete --id "$DID" || true
fi
