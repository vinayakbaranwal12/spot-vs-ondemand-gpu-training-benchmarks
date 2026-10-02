#!/usr/bin/env bash
# Full MI355X 8x clean checkpoints on scratch: 1.7B, 8B, 32B.
# Usage: REGION=mem1 bash scripts/run_full_mi355x8.sh
# Requires region that accepts gpu-mi355x8-2304gb-spot (probe first).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
# shellcheck source=/dev/null
source "$ROOT/scripts/spaces_env_remote.sh"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes)
SCP=(scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes)

SIZE="gpu-mi355x8-2304gb-spot"
REGION="${REGION:?set REGION from probe_mi355x8_region.sh}"
IMAGE="gpu-amd-base"
GPU="mi355x"
NPROC=8
NAME="spot-bench-full-mi355x8-$(date -u +%Y%m%d%H%M%S)"
PROJECT_ID="a25347e6-2cdd-4b40-9d8b-def0047f3720"
SSH_KEY_ID="${SSH_KEY_ID:-58999301}"

SPACES_BUCKET="$SPACES_BUCKET_MEM1"
SPACES_REGION="$SPACES_REGION_MEM1"
SPACES_ENDPOINT="https://${SPACES_REGION}.digitaloceanspaces.com"

echo "=== FULL MI355X 8x CREATE ===" | tee "$ROOT/logs/full_mi355x8.nohup"
echo "size=$SIZE region=$REGION image=$IMAGE nproc=$NPROC price=\$23.76/hr name=$NAME" | tee -a "$ROOT/logs/full_mi355x8.nohup"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/batch_heartbeat"
echo "full_mi355x8 creating" > "$ROOT/logs/active_run.json"

CREATE_JSON=$(doctl compute droplet create "$NAME" \
  --size "$SIZE" --image "$IMAGE" --region "$REGION" \
  --ssh-keys "$SSH_KEY_ID" --tag-name spot-bench \
  --project-id "$PROJECT_ID" --enable-monitoring --wait -o json)
DROPLET_ID=$(python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print(d["id"])' <<<"$CREATE_JSON")
echo "DROPLET_ID=$DROPLET_ID" | tee -a "$ROOT/logs/full_mi355x8.nohup"
echo "full_mi355x8 droplet=$DROPLET_ID" > "$ROOT/logs/active_run.json"

python3 - <<PY
import json,datetime
path="$ROOT/manifest.json"
m=json.load(open(path))
m.setdefault("droplets",[]).append({
  "id": int("$DROPLET_ID"), "name": "$NAME", "size": "$SIZE", "region": "$REGION",
  "image": "$IMAGE", "status": "active",
  "created_at": datetime.datetime.utcnow().isoformat()+"Z", "deleted_at": None,
  "role": "full_mi355x8_checkpoint", "precision": "fp32 master + bf16 compute", "nproc": 8,
})
json.dump(m, open(path,"w"), indent=2)
print("manifest ok")
PY

IP=$(bash "$ROOT/scripts/provision.sh" wait-ssh --id "$DROPLET_ID" --gpu-tool rocm-smi \
  | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/{ip=$0} END{print ip}')
echo "IP=$IP" | tee -a "$ROOT/logs/full_mi355x8.nohup"
[[ -n "$IP" ]] || exit 1
echo "$DROPLET_ID" > /tmp/bench_full_mi355x8_id.txt
echo "$IP" > /tmp/bench_full_mi355x8_ip.txt

SNAP_HOST="$ROOT/snapshots/mi355x8/${NAME}"
mkdir -p "$SNAP_HOST"
doctl compute droplet get "$DROPLET_ID" -o json > "$SNAP_HOST/droplet.json"

"${SCP[@]}" "$ROOT/scripts/"*.py "$ROOT/scripts/"*.sh "root@$IP:/tmp/spot-bench-scripts-upload/"
"${SSH[@]}" "root@$IP" 'mkdir -p /opt/bench-scripts; cp -f /tmp/spot-bench-scripts-upload/* /opt/bench-scripts/'
"${SSH[@]}" "root@$IP" bash -s <<'REMOTE'
set -euo pipefail
if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sh; fi
if ! command -v s5cmd >/dev/null; then
  curl -sL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin s5cmd
fi
bash /opt/bench-scripts/setup_node.sh --gpu mi355x --snapshot-dir /opt/bench/snapshot
# Confirm 8 GPUs
rocm-smi -i | tee /opt/bench/snapshot/gpu_count.txt
REMOTE
"${SCP[@]}" -r "root@$IP:/opt/bench/snapshot/." "$SNAP_HOST/" || true
SCRATCH=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/scratch.path)
CONTAINER=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/container.ref)

spaces_env_push "$IP"

run_model() {
  local MODEL="$1"
  local BS="$2"
  local RUN_ID="clean-scratch-mi355x8-${MODEL}-$(date -u +%Y%m%d%H%M%S)"
  local LOG_DIR="$ROOT/logs/mi355x8/${MODEL}/${RUN_ID}"
  mkdir -p "$LOG_DIR"
  local SPACES_PREFIX="s3://${SPACES_BUCKET}/${RUN_ID}"
  local REMOTE_EVENTS="/opt/bench/train_${MODEL}_events.jsonl"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/batch_heartbeat"
  echo "full_mi355x8 droplet=$DROPLET_ID model=$MODEL" > "$ROOT/logs/active_run.json"

  python3 - <<PY
import json
open("$LOG_DIR/run_meta.json","w").write(json.dumps({
  "run_id": "$RUN_ID", "gpu": "MI355X-8x", "model": "$MODEL",
  "size": "$SIZE", "region": "$REGION", "image": "$IMAGE",
  "droplet_id": int("$DROPLET_ID"), "ip": "$IP", "nproc": 8,
  "batch_size": int("$BS"), "grad_accum": 1, "storage": "scratch",
  "spaces_uri": "$SPACES_PREFIX", "master_dtype": "fp32",
  "exclude_from_final_results": False,
}, indent=2)+"\n")
PY

  echo "=== TRAIN $MODEL nproc=$NPROC batch=$BS on $SCRATCH ===" | tee -a "$ROOT/logs/full_mi355x8.nohup"
  "${SSH[@]}" "root@$IP" \
    CONTAINER="$CONTAINER" SCRATCH="$SCRATCH" MODEL="$MODEL" BS="$BS" NPROC="$NPROC" \
    REMOTE_EVENTS="$REMOTE_EVENTS" SPACES_PREFIX="$SPACES_PREFIX" SPACES_ENDPOINT="$SPACES_ENDPOINT" \
    bash -s <<'REMOTE' | tee -a "$ROOT/logs/full_mi355x8.nohup"
set -euo pipefail
set -a; source /opt/bench/.spaces.env; set +a
rm -f "$REMOTE_EVENTS"
CKPT_DIR="$SCRATCH/bench/ckpts_clean_${MODEL}"
rm -rf "$CKPT_DIR"; mkdir -p "$CKPT_DIR"
docker run --rm --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /opt/bench-scripts:/scripts \
  -v /usr/local/bin/s5cmd:/host-bin/s5cmd:ro \
  -v "$SCRATCH":"$SCRATCH" -v /opt/bench:/opt/bench \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  "$CONTAINER" bash -lc "pip install -q 'transformers>=4.51.0' && \
    torchrun --standalone --nproc_per_node=${NPROC} /scripts/train_bench.py \
      --model ${MODEL} --master-dtype fp32 \
      --batch-size ${BS} --grad-accum 1 \
      --events ${REMOTE_EVENTS} \
      --ckpt-dir ${CKPT_DIR} \
      --max-steps 101 --num-ckpts 3 \
      --warmup-steps 10 --steady-steps 50 --ckpt-every 20 \
      --log-every 1 \
      --upload-after-ckpt --spaces-uri ${SPACES_PREFIX} --spaces-endpoint ${SPACES_ENDPOINT}"
echo TRAIN_DONE model=$MODEL
REMOTE
  "${SCP[@]}" "root@$IP:$REMOTE_EVENTS" "$LOG_DIR/events.jsonl"
  echo "WROTE $LOG_DIR"
}

# Batch sizes chosen to land ~80-90% VRAM on 8x (adjust if OOM; start conservative)
run_model 1.7B 8
run_model 8B 4
run_model 32B 1

spaces_env_clear "$IP"
rm -f "$ROOT/logs/active_run.json"
echo "=== FULL MI355X 8x DONE droplet=$DROPLET_ID ===" | tee -a "$ROOT/logs/full_mi355x8.nohup"
# Leave Droplet for lean snapshot; caller deletes or snapshots
echo "KEEP_DROPLET_ID=$DROPLET_ID IP=$IP"
