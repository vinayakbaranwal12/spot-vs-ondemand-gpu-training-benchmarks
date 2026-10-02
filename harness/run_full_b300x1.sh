#!/usr/bin/env bash
# Full B300 1x checkpoint measurement (fp32 master) on /scratch.
# Spec: gpu-b300x1-288gb-lc-spot, mkc1, gpu-h100x1-base, NGC 25.11-py3 (native on driver 580).
# CUDA gate must pass (no Forward Compatibility) before any train.
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

SIZE="gpu-b300x1-288gb-lc-spot"
REGION="mkc1"
IMAGE="gpu-h100x1-base"
GPU="b300"
NAME="spot-bench-full-b300x1-$(date -u +%Y%m%d%H%M%S)"
LOG="$ROOT/logs/full_b300x1.nohup"
mkdir -p "$(dirname "$LOG")" "$ROOT/snapshots/b300x1"

SPACES_BUCKET="$SPACES_BUCKET_MKC1"
SPACES_REGION="$SPACES_REGION_MKC1"
SPACES_ENDPOINT="https://${SPACES_REGION}.digitaloceanspaces.com"

echo "=== FULL B300 1x CREATE ===" | tee "$LOG"
echo "size=$SIZE region=$REGION image=$IMAGE name=$NAME" | tee -a "$LOG"

bash "$ROOT/scripts/provision.sh" create \
  --name "$NAME" --size "$SIZE" --region "$REGION" --image "$IMAGE" \
  > /tmp/bench_b300_full_create.out 2>&1 || true
cat /tmp/bench_b300_full_create.out | tee -a "$LOG"
DROPLET_ID=$(awk -F= '/^CREATED_ID=/{print $2}' /tmp/bench_b300_full_create.out | tail -1)
[[ -n "$DROPLET_ID" ]] || { echo "CREATE_FAILED"; exit 2; }
echo "DROPLET_ID=$DROPLET_ID" | tee -a "$LOG"
echo "$DROPLET_ID" > /tmp/bench_full_b300x1_id.txt

IP=$(bash "$ROOT/scripts/provision.sh" wait-ssh --id "$DROPLET_ID" --gpu-tool nvidia-smi 2>>"$LOG" \
  | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/{ip=$0} END{print ip}')
echo "IP=$IP" | tee -a "$LOG"
echo "$IP" > /tmp/bench_full_b300x1_ip.txt
[[ -n "$IP" ]] || exit 1

SNAP_HOST="$ROOT/snapshots/b300x1/${NAME}"
mkdir -p "$SNAP_HOST"
doctl compute droplet get "$DROPLET_ID" -o json > "$SNAP_HOST/droplet.json"

python3 - <<PY
import json, datetime
path="$ROOT/manifest.json"
m=json.load(open(path))
m.setdefault("droplets",[]).append({
  "id": int("$DROPLET_ID"), "name": "$NAME", "size": "$SIZE", "region": "$REGION",
  "image": "$IMAGE", "status": "active",
  "created_at": datetime.datetime.utcnow().isoformat()+"Z", "deleted_at": None,
  "role": "full_b300x1_checkpoint", "precision": "fp32 master + bf16 compute",
  "container": "nvcr.io/nvidia/pytorch:25.11-py3",
})
json.dump(m, open(path,"w"), indent=2)
print("manifest ok")
PY

"${SSH[@]}" "root@$IP" 'mkdir -p /opt/bench-scripts /tmp/spot-bench-scripts-upload'
"${SCP[@]}" "$ROOT/scripts/"*.py "$ROOT/scripts/"*.sh "root@$IP:/tmp/spot-bench-scripts-upload/"
"${SSH[@]}" "root@$IP" 'cp -f /tmp/spot-bench-scripts-upload/* /opt/bench-scripts/'

set +e
"${SSH[@]}" "root@$IP" bash -s <<'REMOTE' 2>&1 | tee -a "$LOG"
set -euo pipefail
if ! command -v docker >/dev/null; then curl -fsSL https://get.docker.com | sh; fi
if ! command -v s5cmd >/dev/null; then
  curl -sL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin s5cmd
fi
bash /opt/bench-scripts/setup_node.sh --gpu b300 --snapshot-dir /opt/bench/snapshot
REMOTE
rc=${PIPESTATUS[0]}
set -e
if [[ $rc -ne 0 ]]; then
  if grep -qiE 'unauthorized|authentication required|denied|NGC_API_KEY|nvidia\.com.*401|pull access denied' "$LOG"; then
    echo "ANOMALY_STOP: NGC pull needs auth" | tee -a "$LOG"
    bash "$ROOT/scripts/provision.sh" delete --id "$DROPLET_ID" || true
    exit 3
  fi
  echo "SETUP_FAILED rc=$rc" | tee -a "$LOG"
  bash "$ROOT/scripts/provision.sh" delete --id "$DROPLET_ID" || true
  exit $rc
fi

"${SCP[@]}" -r "root@$IP:/opt/bench/snapshot/." "$SNAP_HOST/" || true
SCRATCH=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/scratch.path)
CONTAINER=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/container.ref)
[[ "$SCRATCH" == "/scratch" ]]
[[ "$CONTAINER" == *25.11-py3 ]] || { echo "ANOMALY_STOP: unexpected container $CONTAINER"; exit 3; }

# CUDA gate before any train
echo "=== CUDA compat gate ===" | tee -a "$LOG"
"${SSH[@]}" "root@$IP" CONTAINER="$CONTAINER" bash -s <<'GATE' 2>&1 | tee "$ROOT/logs/b300x1_full_cuda_gate.log" | tee -a "$LOG"
set -euo pipefail
echo HOST_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo HOST_CUDA_API=$(nvidia-smi | awk -F'CUDA Version: ' '/CUDA Version/ {print $2}' | awk '{print $1}')
docker run --rm --gpus all "$CONTAINER" true 2>&1 | tee /tmp/ngc_banner.txt | grep -E 'Forward Compatibility|Using CUDA|kernel driver|NVIDIA Release|CUDA' || true
if grep -qi 'Forward Compatibility mode ENABLED' /tmp/ngc_banner.txt; then
  echo GATE_FAIL=FORWARD_COMPAT_ENABLED
  exit 42
fi
docker run --rm --gpus all "$CONTAINER" python -c "import torch,json; a=torch.randn(256,256,device='cuda',dtype=torch.bfloat16); b=a@a; torch.cuda.synchronize(); print(json.dumps({'torch_cuda_is_available':bool(torch.cuda.is_available()),'device_name':torch.cuda.get_device_name(0),'compute_capability':'.'.join(map(str,torch.cuda.get_device_capability(0))),'bf16_matmul_ok':True,'torch_cuda_version':torch.version.cuda}))"
echo GATE_OK
GATE
gate_rc=${PIPESTATUS[0]}
if [[ $gate_rc -eq 42 ]]; then
  echo "ANOMALY_STOP: CUDA Forward Compatibility mode ENABLED" | tee -a "$LOG"
  bash "$ROOT/scripts/provision.sh" delete --id "$DROPLET_ID" || true
  exit 3
fi
[[ $gate_rc -eq 0 ]] || { echo "ANOMALY_STOP: CUDA/torch gate failed rc=$gate_rc"; bash "$ROOT/scripts/provision.sh" delete --id "$DROPLET_ID" || true; exit 3; }

run_model() {
  local MODEL="$1"
  local RUN_ID="clean-scratch-b300x1-${MODEL}-$(date -u +%Y%m%d%H%M%S)"
  local LOG_DIR="$ROOT/logs/b300x1/${MODEL}/${RUN_ID}"
  mkdir -p "$LOG_DIR"
  local SPACES_PREFIX="s3://${SPACES_BUCKET}/${RUN_ID}"
  local REMOTE_EVENTS="/opt/bench/train_${MODEL}_events.jsonl"

  python3 - <<PY
import json
open("$LOG_DIR/run_meta.json","w").write(json.dumps({
  "run_id": "$RUN_ID", "gpu": "B300-1x", "model": "$MODEL",
  "size": "$SIZE", "region": "$REGION", "image": "$IMAGE",
  "droplet_id": int("$DROPLET_ID"), "ip": "$IP",
  "spaces_bucket": "$SPACES_BUCKET", "spaces_region": "$SPACES_REGION",
  "cross_region": False, "smoke": False,
  "master_dtype": "fp32",
  "precision_label": "fp32 master + AdamW; bf16 compute",
  "exclude_from_final_results": False,
  "variant": "cold",
  "storage": "scratch",
  "container": "$CONTAINER",
  "overlap_docr_push": False,
  "note": "clean on /scratch; NGC 25.11-py3 native on driver 580; CUDA gate passed",
}, indent=2)+"\n")
PY

  echo "=== TRAIN $MODEL (fp32 master, 3 ckpts) on $SCRATCH ===" | tee -a "$LOG"
  spaces_env_push "$IP"
  "${SSH[@]}" "root@$IP" \
    CONTAINER="$CONTAINER" SCRATCH="$SCRATCH" MODEL="$MODEL" \
    SPACES_PREFIX="$SPACES_PREFIX" SPACES_ENDPOINT="$SPACES_ENDPOINT" \
    REMOTE_EVENTS="$REMOTE_EVENTS" \
    bash -s <<'REMOTE' | tee -a "$LOG"
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a
rm -f "$REMOTE_EVENTS"
CKPT_DIR=$SCRATCH/bench/ckpts_clean_${MODEL}
rm -rf "$CKPT_DIR"
docker run --rm --network=host --ipc=host --gpus all \
  -v /opt/bench-scripts:/scripts \
  -v /usr/local/bin/s5cmd:/host-bin/s5cmd:ro \
  -v "$SCRATCH":"$SCRATCH" \
  -v /opt/bench:/opt/bench \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  "$CONTAINER" \
  bash -lc "pip install -q 'transformers>=4.51.0' && \
    torchrun --standalone --nproc_per_node=1 /scripts/train_bench.py \
      --model ${MODEL} --master-dtype fp32 \
      --batch-size 0 --grad-accum 1 --target-mem-frac 0.85 \
      --events ${REMOTE_EVENTS} \
      --ckpt-dir ${CKPT_DIR} \
      --upload-after-ckpt \
      --spaces-uri ${SPACES_PREFIX} \
      --spaces-endpoint ${SPACES_ENDPOINT} \
      --warmup-steps 10 --steady-steps 50 --num-ckpts 3 --ckpt-every 20 --log-every 5"
REMOTE

  # Host-upload fallback for any remaining local ckpts
  "${SSH[@]}" "root@$IP" \
    bash -s <<REMOTE || true
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a
for d in ${SCRATCH}/bench/ckpts_clean_${MODEL}/ckpt-step-*; do
  [ -d "\$d" ] || continue
  base=\$(basename "\$d")
  dest="${SPACES_PREFIX}/\${base}/"
  echo "host upload \$d -> \$dest"
  s5cmd --endpoint-url "$SPACES_ENDPOINT" cp "\$d/" "\$dest"
done
REMOTE

  "${SCP[@]}" "root@$IP:$REMOTE_EVENTS" "$LOG_DIR/events.jsonl" || true
  python3 - <<PY
import json
from pathlib import Path
log=Path("$LOG_DIR")
meta=json.loads((log/"run_meta.json").read_text())
batch={}
ev=log/"events.jsonl"
if ev.exists():
  for line in ev.read_text().splitlines():
    if not line.strip(): continue
    e=json.loads(line)
    if e.get("event") in ("batch_selected","train_hparams","model_ready","step"):
      batch.update({k:e[k] for k in ("batch_size","grad_accum","tokens_per_step","target_mem_frac","seq_len") if k in e})
meta.update(batch)
meta["overlap_docr_push"]=False
meta["storage"]="scratch"
(log/"run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
print("run_meta batch fields:", batch)
PY
  echo "DONE_MODEL=$MODEL LOG=$LOG_DIR" | tee -a "$LOG"
}

run_model "1.7B"
run_model "8B"

spaces_env_clear "$IP"
echo "FULL_B300X1_CKPT_PHASE_DONE DROPLET_ID=$DROPLET_ID IP=$IP" | tee -a "$LOG"
echo "Next: lean snapshot from this Droplet, then 3 cold + 3 lean resumes (batch approved)."
