#!/usr/bin/env bash
# Clean fp32 checkpoint measurements to /scratch (DOSCRATCH NVMe), 3 per model.
# Marks prior boot-disk clean rows as storage=boot_disk excluded from final.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
# shellcheck source=/dev/null
source "$ROOT/scripts/spaces_env_remote.sh"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
IP="${1:-}"
ID="${2:-}"
[[ -n "$IP" && -n "$ID" ]] || { echo "usage: $0 <IP> <droplet_id>"; exit 1; }
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes)
SCP=(scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes)
ENDPOINT="https://${SPACES_REGION_MEM1}.digitaloceanspaces.com"
LOG="$ROOT/logs/full_mi355x1_clean_ckpts_scratch.log"

bash "$ROOT/scripts/preflight.sh" host

mkdir -p "$(dirname "$LOG")"
echo "=== clean ckpts on scratch: require no concurrent docker push ===" | tee "$LOG"
push_running() {
  "${SSH[@]}" "root@$IP" "ps -eo args= | grep -E '^docker push |^/usr/bin/docker push ' || true"
}
if [[ -n "$(push_running)" ]]; then
  echo "ABORT: docker push still running" | tee -a "$LOG"
  exit 2
fi

CONTAINER=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/container.ref)
SCRATCH=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/scratch.path)
[[ "$SCRATCH" == "/scratch" ]] || { echo "ABORT: scratch.path=$SCRATCH (want /scratch)" | tee -a "$LOG"; exit 2; }

bash "$ROOT/scripts/preflight.sh" droplet --ssh "root@$IP" --identity "$SSH_KEY"
bash "$ROOT/scripts/preflight.sh" container --ssh "root@$IP" --identity "$SSH_KEY" --container "$CONTAINER" --gpu mi355x

"${SCP[@]}" "$ROOT/scripts/"*.py "$ROOT/scripts/"*.sh "root@$IP:/opt/bench-scripts/"
"${SSH[@]}" "root@$IP" 'rm -rf /opt/bench-scripts/scripts; test -f /opt/bench-scripts/train_bench.py'

# Exclude prior boot-disk clean checkpoint runs from final
python3 - <<'PY'
import json
from pathlib import Path
root = Path(".")
for meta_path in root.glob("logs/mi355x1/*/clean-mi355x1-*/run_meta.json"):
    meta = json.loads(meta_path.read_text())
    if meta.get("storage") == "scratch":
        continue
    meta["storage"] = "boot_disk"
    meta["exclude_from_final_results"] = True
    meta["exclude_reason"] = "storage_boot_disk; checkpoints written to /mnt on vda1"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print("marked boot_disk:", meta_path.parent.name)
PY

run_clean() {
  local MODEL="$1"
  local RUN_ID="clean-scratch-mi355x1-${MODEL}-$(date -u +%Y%m%d%H%M%S)"
  local LOG_DIR="$ROOT/logs/mi355x1/${MODEL}/${RUN_ID}"
  mkdir -p "$LOG_DIR"
  local SPACES_PREFIX="s3://${SPACES_BUCKET_MEM1}/${RUN_ID}"
  python3 - <<PY
import json
open("$LOG_DIR/run_meta.json","w").write(json.dumps({
  "run_id": "$RUN_ID", "gpu": "MI355X-1x", "model": "$MODEL",
  "master_dtype": "fp32", "precision_label": "fp32 master + AdamW; bf16 compute",
  "exclude_from_final_results": False, "variant": "cold",
  "overlap_docr_push": False,
  "storage": "scratch",
  "note": "clean on /scratch DOSCRATCH NVMe; auto batch; overlap_docr_push=false",
  "droplet_id": int("$ID"),
}, indent=2)+"\n")
PY
  echo "=== CLEAN TRAIN $MODEL on $SCRATCH ===" | tee -a "$LOG"
  if [[ -n "$(push_running)" ]]; then
    echo "ABORT: docker push still running" | tee -a "$LOG"
    exit 2
  fi
spaces_env_push "$IP"
  "${SSH[@]}" "root@$IP" \
    CONTAINER="$CONTAINER" SCRATCH="$SCRATCH" ENDPOINT="$ENDPOINT" \
    MODEL="$MODEL" SPACES_PREFIX="$SPACES_PREFIX" \
    bash -s <<'REMOTE' >> "$LOG" 2>&1
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a
findmnt /scratch >/dev/null
docker run --rm -v /usr/local/bin/s5cmd:/host-bin/s5cmd:ro "$CONTAINER" /host-bin/s5cmd version
REMOTE_EVENTS=/opt/bench/train_clean_scratch_${MODEL}_events.jsonl
rm -f "$REMOTE_EVENTS"
CKPT_DIR=$SCRATCH/bench/ckpts_clean_${MODEL}
rm -rf "$CKPT_DIR"
docker run --rm --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /opt/bench-scripts:/scripts \
  -v /usr/local/bin/s5cmd:/host-bin/s5cmd:ro \
  -v "$SCRATCH":"$SCRATCH" -v /opt/bench:/opt/bench \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  "$CONTAINER" bash -lc "pip install -q 'transformers>=4.51.0' && \
    torchrun --standalone --nproc_per_node=1 /scripts/train_bench.py \
      --model ${MODEL} --master-dtype fp32 \
      --batch-size 0 --grad-accum 1 --target-mem-frac 0.85 \
      --events ${REMOTE_EVENTS} --ckpt-dir ${CKPT_DIR} \
      --upload-after-ckpt \
      --spaces-uri ${SPACES_PREFIX} --spaces-endpoint ${ENDPOINT} \
      --warmup-steps 10 --steady-steps 50 --num-ckpts 3 --ckpt-every 20 --log-every 5"
# Prove ckpts landed on scratch device
df -h "$CKPT_DIR" | tee /opt/bench/ckpt_scratch_${MODEL}_df.txt
findmnt -n -o SOURCE,TARGET "$SCRATCH"
du -sh "$CKPT_DIR"
echo CLEAN_DONE_${MODEL}
REMOTE
  "${SCP[@]}" "root@$IP:/opt/bench/train_clean_scratch_${MODEL}_events.jsonl" "$LOG_DIR/events.jsonl"
  "${SCP[@]}" "root@$IP:/opt/bench/ckpt_scratch_${MODEL}_df.txt" "$LOG_DIR/ckpt_df.txt" || true
  python3 - <<PY
import json
from pathlib import Path
log=Path("$LOG_DIR")
meta=json.loads((log/"run_meta.json").read_text())
batch={}
for line in (log/"events.jsonl").read_text().splitlines():
  if not line.strip(): continue
  e=json.loads(line)
  if e.get("event") in ("batch_selected","train_hparams"):
    batch.update({k:e[k] for k in ("batch_size","grad_accum","tokens_per_step","target_mem_frac","seq_len") if k in e})
  if e.get("event") in ("upload_start","upload_end","ckpt_save_end"):
    e["overlap_docr_push"]=False
meta.update(batch)
meta["overlap_docr_push"]=False
meta["storage"]="scratch"
(log/"run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
lines=[]
for line in (log/"events.jsonl").read_text().splitlines():
  if not line.strip(): continue
  e=json.loads(line)
  if e.get("event") in ("upload_start","upload_end","ckpt_save_end"):
    e["overlap_docr_push"]=False
  lines.append(json.dumps(e, sort_keys=True))
(log/"events.jsonl").write_text("\n".join(lines)+"\n")
print("run_meta batch fields:", batch)
PY
}

run_clean "1.7B"
run_clean "8B"
echo CLEAN_CKPT_SCRATCH_PHASE_DONE | tee -a "$LOG"
spaces_env_clear "$IP"
