#!/usr/bin/env bash
# Snapshot resume run 1 extras: disk breakdown + strict-loss reference continuation.
# Assumes Droplet already has scripts synced and setup_node done (or can re-run setup).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
IP="${1:?ip}"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes)
SCP=(scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes)
ENDPOINT="https://${SPACES_REGION_MEM1}.digitaloceanspaces.com"
RUN_META="$ROOT/logs/mi355x1/8B/clean-mi355x1-8B-20260930212438/run_meta.json"
BATCH_SIZE=$(python3 -c 'import json; print(json.load(open("'"$RUN_META"'"))["batch_size"])')
GRAD_ACCUM=$(python3 -c 'import json; print(json.load(open("'"$RUN_META"'")).get("grad_accum",1))')
CKPT80="s3://${SPACES_BUCKET_MEM1}/clean-mi355x1-8B-20260930212438/ckpt-step-80"
OUT_DIR="$ROOT/logs/mi355x1/8B/strict-loss-ref-$(date -u +%Y%m%d%H%M%S)"
BREAKDOWN="$ROOT/snapshots/mi355x1/snapshot-breakdown.txt"
mkdir -p "$OUT_DIR" "$(dirname "$BREAKDOWN")"

echo "=== disk breakdown on $IP ===" | tee "$OUT_DIR/driver.log"
"${SSH[@]}" "root@$IP" bash -s <<'REMOTE' | tee "$BREAKDOWN"
set -euo pipefail
echo "# snapshot disk breakdown"
echo "hostname=$(hostname)"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "## df -h"
df -h
echo
echo "## docker system df"
docker system df || true
echo
echo "## docker images"
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.Size}}' || true
echo
echo "## top-level du -sh (may take a few minutes)"
du -sh /* 2>/dev/null | sort -h || true
echo
echo "## key paths"
for p in /var/lib/docker /var/lib/containerd /root /home /tmp /opt /mnt /usr /var/cache /var/tmp /var/log; do
  if [ -e "$p" ]; then du -sh "$p" 2>/dev/null || true; fi
done
echo
echo "## largest under /var/lib/docker (depth 2)"
du -sh /var/lib/docker/* 2>/dev/null | sort -h | tail -30 || true
echo
echo "## largest under /root (depth 2)"
du -sh /root/* /root/.[!.]* 2>/dev/null | sort -h | tail -30 || true
echo
echo "## huggingface / pip / caches if present"
du -sh /root/.cache /root/.cache/huggingface /root/.cache/pip /var/cache/apt 2>/dev/null || true
echo
echo "## spot-bench paths"
du -sh /opt/bench /mnt/spot-bench /opt/bench-scripts 2>/dev/null || true
REMOTE

echo "BREAKDOWN_WRITTEN=$BREAKDOWN"

# Parse non-image estimate: docker images ~50G; flag if boot-disk used - image >> 100G leftovers
python3 - <<PY | tee -a "$OUT_DIR/driver.log"
from pathlib import Path
import re
text=Path("$BREAKDOWN").read_text()
# df for /
used=None
for line in text.splitlines():
  if line.startswith('/dev/') and ('/' == line.split()[-1] or line.strip().endswith(' /')):
    parts=line.split()
    # Filesystem Size Used Avail Use% Mounted
    if len(parts)>=6:
      used=parts[2]
print('root_used_field', used)
# docker image line sizes
print('NOTE: review breakdown; non-image threshold ~100 GiB beyond ~50 GiB docker image')
PY

CONTAINER=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/container.ref)
SCRATCH=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/scratch.path)

echo "=== download ckpt-step-80 for reference continuation ===" | tee -a "$OUT_DIR/driver.log"
CKPT_LOCAL="$SCRATCH/bench/ref-ckpt-80"
# shellcheck source=/dev/null
source "$ROOT/scripts/spaces_env_remote.sh"
spaces_env_push "$IP"
"${SSH[@]}" "root@$IP" bash -s <<REMOTE
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a
rm -rf '$CKPT_LOCAL'; mkdir -p '$CKPT_LOCAL'
s5cmd --endpoint-url '$ENDPOINT' cp '${CKPT80%/}/*' '$CKPT_LOCAL/'
REMOTE

REMOTE_EVENTS=/opt/bench/strict_loss_ref_events.jsonl
# start_step after ckpt 80 = 81; need through step 101 inclusive => max_steps additional = 21
echo "=== reference train: resume ckpt-80, batch=$BATCH_SIZE, through step 101 (21 steps), log-every 1 ===" | tee -a "$OUT_DIR/driver.log"
"${SSH[@]}" "root@$IP" \
  CONTAINER="$CONTAINER" SCRATCH="$SCRATCH" CKPT_LOCAL="$CKPT_LOCAL" \
  BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" REMOTE_EVENTS="$REMOTE_EVENTS" \
  bash -s <<'REMOTE' | tee -a "$OUT_DIR/driver.log"
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a
rm -f "$REMOTE_EVENTS"
docker run --rm --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /opt/bench-scripts:/scripts \
  -v /usr/local/bin/s5cmd:/host-bin/s5cmd:ro \
  -v "$SCRATCH":"$SCRATCH" -v /opt/bench:/opt/bench \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  "$CONTAINER" bash -lc "pip install -q 'transformers>=4.51.0' && \
    torchrun --standalone --nproc_per_node=1 /scripts/train_bench.py \
      --model 8B --master-dtype fp32 \
      --batch-size ${BATCH_SIZE} --grad-accum ${GRAD_ACCUM} \
      --events ${REMOTE_EVENTS} \
      --ckpt-dir ${SCRATCH}/bench/ckpts_ref \
      --resume-from ${CKPT_LOCAL} \
      --max-steps 21 \
      --log-every 1 \
      --warmup-steps 0 --steady-steps 0 --num-ckpts 0"
echo REF_TRAIN_DONE
REMOTE
spaces_env_clear "$IP"

"${SCP[@]}" "root@$IP:$REMOTE_EVENTS" "$OUT_DIR/events.jsonl"

python3 - <<PY
import json
from pathlib import Path
out=Path("$OUT_DIR")
events=[json.loads(l) for l in (out/"events.jsonl").read_text().splitlines() if l.strip()]
by_step={e["step"]: e["loss"] for e in events if e.get("event")=="step"}
# original clean losses
orig={}
for line in Path("$ROOT/logs/mi355x1/8B/clean-mi355x1-8B-20260930212438/events.jsonl").read_text().splitlines():
  e=json.loads(line)
  if e.get("event")=="step":
    orig[int(e["step"])]=float(e["loss"])
tol_abs=1e-3
def within(a,b):
  return abs(float(a)-float(b)) <= max(tol_abs, 1e-5*abs(float(b)))

ref100=by_step.get(100)
ref101=by_step.get(101)
orig100=orig.get(100)
anchor_ok = (ref100 is not None and orig100 is not None and within(ref100, orig100))
result={
  "tolerance_abs": tol_abs,
  "tolerance_rel": 1e-5,
  "tolerance_rule": "abs(a-b) <= max(1e-3, 1e-5*abs(b))",
  "reference_loss_step_100": ref100,
  "original_loss_step_100": orig100,
  "anchor_match_step_100": anchor_ok,
  "reference_loss_step_101": ref101,
  "steps_logged": sorted(by_step),
  "batch_size": int("$BATCH_SIZE"),
  "grad_accum": int("$GRAD_ACCUM"),
  "resume_from": "ckpt-step-80",
}
(out/"strict_loss_reference.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
if not anchor_ok:
  raise SystemExit("ANCHOR_FAIL: step-100 reference loss does not match original")
print("ANCHOR_OK")
PY
