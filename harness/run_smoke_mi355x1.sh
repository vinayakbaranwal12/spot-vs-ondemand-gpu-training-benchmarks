#!/usr/bin/env bash
# MI355X 1x smoke test: 1.7B, one checkpoint, one upload, one simulated reclaim+resume.
# ONLY run after operator typed yes for this Droplet create.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
# shellcheck source=/dev/null
source "$ROOT/scripts/spaces_env_remote.sh"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

SIZE="gpu-mi355x1-288gb-spot"
REGION="mem1"
IMAGE="gpu-amd-base"
GPU="mi355x"
MODEL="1.7B"
NAME="spot-bench-smoke-mi355x1-$(date -u +%Y%m%d%H%M%S)"
RUN_ID="smoke-mi355x1-1p7b-$(date -u +%Y%m%d%H%M%S)"
LOG_DIR="$ROOT/logs/mi355x1/${MODEL}/${RUN_ID}"
SNAP_HOST="$ROOT/snapshots/mi355x1/${NAME}"
mkdir -p "$LOG_DIR" "$SNAP_HOST"

SPACES_BUCKET="$SPACES_BUCKET_MEM1"
SPACES_REGION="$SPACES_REGION_MEM1"
SPACES_ENDPOINT="https://${SPACES_REGION}.digitaloceanspaces.com"
SPACES_PREFIX="s3://${SPACES_BUCKET}/${RUN_ID}"

set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a

echo "=== SMOKE CREATE ==="
echo "size=$SIZE region=$REGION image=$IMAGE name=$NAME"
echo "spaces=${SPACES_PREFIX} endpoint=$SPACES_ENDPOINT"
echo "cross_region_disclaimer: droplet=$REGION spaces=$SPACES_REGION"

DROPLET_ID=$("$ROOT/scripts/provision.sh" create \
  --name "$NAME" --size "$SIZE" --region "$REGION" --image "$IMAGE" --wait | tee /tmp/bench_create.out | awk -F= '/^CREATED_ID=/{print $2}' | tail -1)

echo "DROPLET_ID=$DROPLET_ID"
doctl compute droplet get "$DROPLET_ID" -o json > "$SNAP_HOST/droplet.json"
IP=$("$ROOT/scripts/provision.sh" wait-ssh --id "$DROPLET_ID" --gpu-tool rocm-smi)
echo "IP=$IP"

# Persist run meta
python3 - <<PY
import json
meta={
  "run_id": "$RUN_ID",
  "gpu": "MI355X-1x",
  "model": "$MODEL",
  "size": "$SIZE",
  "region": "$REGION",
  "image": "$IMAGE",
  "droplet_id": int("$DROPLET_ID"),
  "spaces_bucket": "$SPACES_BUCKET",
  "spaces_region": "$SPACES_REGION",
  "cross_region": "$REGION" != "$SPACES_REGION",
  "smoke": True,
  "simulated_reclaim": True,
}
open("$LOG_DIR/run_meta.json","w").write(json.dumps(meta, indent=2)+"\n")
PY

# Tools + scripts on droplet
scp -o StrictHostKeyChecking=no -r "$ROOT/scripts" "root@$IP:/opt/bench-scripts"
ssh -o StrictHostKeyChecking=no "root@$IP" bash -s <<REMOTE
set -euo pipefail
if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sh; fi
if ! command -v s5cmd >/dev/null; then
  curl -sL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin s5cmd
fi
bash /opt/bench-scripts/setup_node.sh --gpu mi355x --snapshot-dir /opt/bench/snapshot --smoke
REMOTE

# Pull snapshot back
scp -o StrictHostKeyChecking=no -r "root@$IP:/opt/bench/snapshot/"* "$SNAP_HOST/" || true
SCRATCH=$(ssh -o StrictHostKeyChecking=no "root@$IP" cat /opt/bench/snapshot/scratch.path)
CONTAINER=$(ssh -o StrictHostKeyChecking=no "root@$IP" cat /opt/bench/snapshot/container.ref)

# Train smoke: warmup+steady, 1 ckpt, upload, then continue a few steps and kill
REMOTE_EVENTS="/opt/bench/train_events.jsonl"
CKPT_STEP_TARGET=""  # filled from events later
KILL_AT=""

ssh -o StrictHostKeyChecking=no "root@$IP" bash -s <<REMOTE
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a
CONTAINER='${CONTAINER}'
SCRATCH='${SCRATCH}'
SPACES_PREFIX='${SPACES_PREFIX}'
SPACES_ENDPOINT='${SPACES_ENDPOINT}'
docker run --rm --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /opt/bench-scripts:/scripts \
  -v "\$SCRATCH":"\$SCRATCH" \
  -v /opt/bench:/opt/bench \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  "\$CONTAINER" \
  bash -lc "pip install -q 'transformers>=4.51.0' && \
    torchrun --standalone --nproc_per_node=1 /scripts/train_bench.py \
      --model 1.7B --smoke \
      --events ${REMOTE_EVENTS} \
      --ckpt-dir \$SCRATCH/bench/ckpts \
      --upload-after-ckpt \
      --spaces-uri \$SPACES_PREFIX \
      --spaces-endpoint \$SPACES_ENDPOINT \
      --kill-at-step 9999 \
      --max-steps 50"
REMOTE

# Fetch train events
scp -o StrictHostKeyChecking=no "root@$IP:$REMOTE_EVENTS" "$LOG_DIR/events.jsonl"
scp -o StrictHostKeyChecking=no "root@$IP:/opt/bench/losses.json" "$LOG_DIR/losses.json" 2>/dev/null || true

# Determine ckpt step and a later kill step from events
CKPT_STEP=$(python3 - <<'PY'
import json
step=None
for line in open("'"$LOG_DIR"'/events.jsonl"):
    e=json.loads(line)
    if e.get("event")=="ckpt_save_end":
        step=e.get("step")
print(step if step is not None else "")
PY
)
if [[ -z "$CKPT_STEP" ]]; then
  echo "NO CHECKPOINT in events; aborting before reclaim" >&2
  "$ROOT/scripts/provision.sh" delete --id "$DROPLET_ID"
  exit 1
fi
# For smoke we already uploaded during train. Find uploaded ckpt name.
CKPT_URI=$(python3 - <<'PY'
import json
dest=""
for line in open("'"$LOG_DIR"'/events.jsonl"):
    e=json.loads(line)
    if e.get("event")=="upload_end":
        dest=e.get("dest","")
print(dest.rstrip("/"))
PY
)
echo "CKPT_STEP=$CKPT_STEP CKPT_URI=$CKPT_URI"

# Simulate reclaim: delete Droplet A by ID
echo "=== SIMULATED RECLAIM: deleting Droplet $DROPLET_ID ==="
"$ROOT/scripts/provision.sh" delete --id "$DROPLET_ID"

# Write marker that replacement needs a NEW yes
cat > "$LOG_DIR/NEEDS_REPLACEMENT_APPROVAL.txt" <<EOF
Simulated reclaim complete. Droplet $DROPLET_ID deleted.
Next: operator must type yes for replacement Droplet before measure_resume.sh starts.
Replacement plan:
  name: ${NAME}-resume
  size: $SIZE
  region: $REGION
  image: $IMAGE
  spaces ckpt: $CKPT_URI
EOF

echo "SMOKE_PHASE1_DONE"
echo "LOG_DIR=$LOG_DIR"
echo "CKPT_URI=$CKPT_URI"
echo "Awaiting yes for replacement Droplet before resume timing starts."
spaces_env_clear "$IP"
