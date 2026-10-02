#!/usr/bin/env bash
# Full MI355X 1x checkpoint measurement (fp32 master). One Droplet, models 1.7B then 8B.
# Requires prior operator yes. Uses ~/.ssh/vinayak_ssh.
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

SIZE="gpu-mi355x1-288gb-spot"
REGION="mem1"
IMAGE="gpu-amd-base"
GPU="mi355x"
NAME="spot-bench-full-mi355x1-$(date -u +%Y%m%d%H%M%S)"
PROJECT_ID="a25347e6-2cdd-4b40-9d8b-def0047f3720"
SSH_KEY_ID="${SSH_KEY_ID:-58999301}"

SPACES_BUCKET="$SPACES_BUCKET_MEM1"
SPACES_REGION="$SPACES_REGION_MEM1"
SPACES_ENDPOINT="https://${SPACES_REGION}.digitaloceanspaces.com"

echo "=== FULL MI355X 1x CREATE ==="
echo "size=$SIZE region=$REGION image=$IMAGE price=\$2.97/hr name=$NAME"

CREATE_JSON=$(doctl compute droplet create "$NAME" \
  --size "$SIZE" --image "$IMAGE" --region "$REGION" \
  --ssh-keys "$SSH_KEY_ID" --tag-name spot-bench \
  --project-id "$PROJECT_ID" --enable-monitoring --wait -o json)
DROPLET_ID=$(python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print(d["id"])' <<<"$CREATE_JSON")
echo "DROPLET_ID=$DROPLET_ID"

python3 - <<PY
import json,datetime
path="$ROOT/manifest.json"
m=json.load(open(path))
m.setdefault("droplets",[]).append({
  "id": int("$DROPLET_ID"), "name": "$NAME", "size": "$SIZE", "region": "$REGION",
  "image": "$IMAGE", "status": "active",
  "created_at": datetime.datetime.utcnow().isoformat()+"Z", "deleted_at": None,
  "role": "full_mi355x1_checkpoint", "precision": "fp32 master + bf16 compute",
})
json.dump(m, open(path,"w"), indent=2)
print("manifest ok")
PY

# Wait SSH
IP=""
for _ in $(seq 1 40); do
  IP=$(doctl compute droplet get "$DROPLET_ID" -o json | python3 -c 'import json,sys;d=json.load(sys.stdin);d=d[0] if isinstance(d,list) else d
print(next((n["ip_address"] for n in d.get("networks",{}).get("v4",[]) if n.get("type")=="public"),""))')
  if [[ -n "$IP" ]] && "${SSH[@]}" -o ConnectTimeout=5 "root@$IP" "rocm-smi >/dev/null 2>&1"; then
    break
  fi
  sleep 10
done
echo "IP=$IP"
[[ -n "$IP" ]] || exit 1
echo "$DROPLET_ID" > /tmp/bench_full_mi355x1_id.txt
echo "$IP" > /tmp/bench_full_mi355x1_ip.txt

SNAP_HOST="$ROOT/snapshots/mi355x1/${NAME}"
mkdir -p "$SNAP_HOST"
doctl compute droplet get "$DROPLET_ID" -o json > "$SNAP_HOST/droplet.json"

"${SCP[@]}" -r "$ROOT/scripts" "root@$IP:/opt/bench-scripts"
"${SSH[@]}" "root@$IP" bash -s <<'REMOTE'
set -euo pipefail
if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sh; fi
if ! command -v s5cmd >/dev/null; then
  curl -sL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin s5cmd
fi
bash /opt/bench-scripts/setup_node.sh --gpu mi355x --snapshot-dir /opt/bench/snapshot
REMOTE

"${SCP[@]}" -r "root@$IP:/opt/bench/snapshot/." "$SNAP_HOST/" || true
SCRATCH=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/scratch.path)
CONTAINER=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/container.ref)

run_model() {
  local MODEL="$1"
  local RUN_ID="full-mi355x1-${MODEL}-$(date -u +%Y%m%d%H%M%S)"
  local LOG_DIR="$ROOT/logs/mi355x1/${MODEL}/${RUN_ID}"
  mkdir -p "$LOG_DIR"
  local SPACES_PREFIX="s3://${SPACES_BUCKET}/${RUN_ID}"
  local REMOTE_EVENTS="/opt/bench/train_${MODEL}_events.jsonl"

  python3 - <<PY
import json
open("$LOG_DIR/run_meta.json","w").write(json.dumps({
  "run_id": "$RUN_ID", "gpu": "MI355X-1x", "model": "$MODEL",
  "size": "$SIZE", "region": "$REGION", "image": "$IMAGE",
  "droplet_id": int("$DROPLET_ID"), "ip": "$IP",
  "spaces_bucket": "$SPACES_BUCKET", "spaces_region": "$SPACES_REGION",
  "cross_region": True, "smoke": False,
  "master_dtype": "fp32",
  "precision_label": "fp32 master + AdamW; bf16 compute",
  "exclude_from_final_results": False,
  "variant": "cold",
}, indent=2)+"\n")
PY

  echo "=== TRAIN $MODEL (fp32 master, 3 ckpts) ==="
spaces_env_push "$IP"
  "${SSH[@]}" "root@$IP" \
    CONTAINER="$CONTAINER" SCRATCH="$SCRATCH" MODEL="$MODEL" \
    SPACES_PREFIX="$SPACES_PREFIX" SPACES_ENDPOINT="$SPACES_ENDPOINT" \
    REMOTE_EVENTS="$REMOTE_EVENTS" \
    bash -s <<'REMOTE'
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
  -v "$SCRATCH":"$SCRATCH" \
  -v /opt/bench:/opt/bench \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  "$CONTAINER" \
  bash -lc "pip install -q 'transformers>=4.51.0' && \
    torchrun --standalone --nproc_per_node=1 /scripts/train_bench.py \
      --model ${MODEL} --master-dtype fp32 \
      --events ${REMOTE_EVENTS} \
      --ckpt-dir ${SCRATCH}/bench/ckpts \
      --upload-after-ckpt \
      --spaces-uri ${SPACES_PREFIX} \
      --spaces-endpoint ${SPACES_ENDPOINT} \
      --warmup-steps 10 --steady-steps 50 --num-ckpts 3 --ckpt-every 20 --log-every 5"
REMOTE

  # If upload failed inside container, host-upload any remaining local ckpts
  "${SSH[@]}" "root@$IP" \
    bash -s <<REMOTE || true
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a
for d in ${SCRATCH}/bench/ckpts/ckpt-step-*; do
  [ -d "\$d" ] || continue
  base=\$(basename "\$d")
  dest="${SPACES_PREFIX}/\${base}/"
  echo "host upload \$d -> \$dest"
  s5cmd --endpoint-url "$SPACES_ENDPOINT" cp "\$d/" "\$dest"
  rm -rf "\$d"
done
REMOTE

  "${SCP[@]}" "root@$IP:$REMOTE_EVENTS" "$LOG_DIR/events.jsonl" || true
  echo "DONE_MODEL=$MODEL LOG=$LOG_DIR"
}

run_model "1.7B"
run_model "8B"

echo "FULL_CKPT_PHASE_DONE DROPLET_ID=$DROPLET_ID IP=$IP"
echo "Next: 8B cold resume chain (3 runs) on this Droplet, then ask before DOCR/snapshot variants."
spaces_env_clear "$IP"
