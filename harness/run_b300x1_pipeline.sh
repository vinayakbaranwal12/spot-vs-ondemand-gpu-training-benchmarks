#!/usr/bin/env bash
# B300 1x pipeline after smoke: clean on scratch → lean snapshot → 3 cold + 3 lean.
# Container: NGC 25.11-py3 (native on gpu-h100x1-base driver 580). No manual driver install.
set -euo pipefail
ROOT="."
cd "$ROOT"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

LOG="$ROOT/logs/b300x1_pipeline.nohup"
mkdir -p "$(dirname "$LOG")"
echo "=== b300x1_pipeline start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

(
  while true; do date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/batch_heartbeat"; sleep 60; done
) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

anomaly() { echo "ANOMALY_STOP: $*" | tee -a "$LOG"; exit 3; }

# --- Phase 1: full clean (1.7B + 8B) ---
echo "=== PHASE1 full clean ===" | tee -a "$LOG"
bash "$ROOT/scripts/run_full_b300x1.sh" 2>&1 | tee -a "$LOG"
DID=$(cat /tmp/bench_full_b300x1_id.txt)
IP=$(cat /tmp/bench_full_b300x1_ip.txt)
[[ -n "$DID" && -n "$IP" ]] || anomaly "missing full droplet id/ip"

CLEAN_META=$(ls -dt "$ROOT"/logs/b300x1/8B/clean-scratch-b300x1-8B-*/run_meta.json | head -1)
[[ -f "$CLEAN_META" ]] || anomaly "missing clean 8B run_meta"
python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); assert m.get("batch_size"), m' "$CLEAN_META" \
  || anomaly "clean run_meta missing batch_size"
echo "CLEAN_META=$CLEAN_META" | tee -a "$LOG"

# --- Phase 2: lean snapshot (OS + pulled NGC image; scratch excluded) ---
SNAP_NAME="spot-bench-b300x1-ngc2511-lean-$(date -u +%Y%m%d%H%M%S)"
echo "=== PHASE2 lean snapshot $SNAP_NAME from $DID ===" | tee -a "$LOG"
# Wipe local ckpts on scratch so they are not needed; snapshot is boot disk only anyway.
ssh -i "${SSH_KEY:-$HOME/.ssh/id_ed25519}" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes \
  "root@$IP" 'rm -rf /scratch/bench/ckpts_clean_* || true; docker image inspect nvcr.io/nvidia/pytorch:25.11-py3 >/dev/null'

doctl compute droplet-action snapshot "$DID" --snapshot-name "$SNAP_NAME" --wait 2>&1 | tee -a "$LOG"
# Resolve snapshot image ID by name
LEAN_IMAGE=$(doctl compute image list --format ID,Name,Type,Created | awk -v n="$SNAP_NAME" '$0 ~ n {print $1; exit}')
[[ -n "$LEAN_IMAGE" ]] || {
  # Fallback: newest private snapshot matching prefix
  LEAN_IMAGE=$(doctl compute image list --format ID,Name,Type | awk '/spot-bench-b300x1-ngc2511-lean/ {print $1}' | tail -1)
}
[[ -n "$LEAN_IMAGE" ]] || anomaly "failed to resolve lean snapshot id for $SNAP_NAME"
echo "LEAN_IMAGE=$LEAN_IMAGE" | tee -a "$LOG"
echo "$LEAN_IMAGE" > /tmp/bench_b300_lean_image.txt

mkdir -p "$ROOT/snapshots/b300x1" "$ROOT/pricing"
doctl compute image get "$LEAN_IMAGE" -o json > "$ROOT/snapshots/b300x1/lean-snapshot-${LEAN_IMAGE}.json"
LEAN_JSON="$ROOT/snapshots/b300x1/lean-snapshot-${LEAN_IMAGE}.json" \
LEAN_MD="$ROOT/pricing/snapshot-b300x1-lean.md" \
SOURCE_DID="$DID" \
python3 - <<'PY'
import json, datetime, os
from pathlib import Path
img=json.load(open(os.environ["LEAN_JSON"]))
d=img[0] if isinstance(img,list) else img
md=Path(os.environ["LEAN_MD"])
size_gb=d.get("size_gigabytes")
cost=float(size_gb or 0)*0.06
md.write_text(
  "# B300 1x lean snapshot (OS + pulled NGC 25.11; scratch excluded)\n\n"
  f"- Snapshot ID: {d.get('id')}\n"
  f"- Name: {d.get('name')}\n"
  f"- Status: {d.get('status')}\n"
  f"- size_gigabytes: {size_gb}\n"
  f"- min_disk_size_gb: {d.get('min_disk_size')}\n"
  f"- regions: {d.get('regions')}\n"
  f"- created_at: {d.get('created_at')}\n"
  f"- Source Droplet ID: {os.environ['SOURCE_DID']}\n"
  "- Docker image: nvcr.io/nvidia/pytorch:25.11-py3 (CUDA 13.0.2; native on driver 580)\n"
  "- Digest: nvcr.io/nvidia/pytorch@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21\n"
  "- Snapshot storage rate (DigitalOcean published): $0.06 / GiB / month\n"
  f"- Estimated monthly cost: ${cost:.2f}\n"
  f"- Recorded: {datetime.datetime.utcnow().isoformat()}Z\n"
)
print("lean snapshot md written", md)
PY

# Delete the full Droplet after snapshot
bash "$ROOT/scripts/provision.sh" delete --id "$DID" 2>&1 | tee -a "$LOG" || true

# --- Phase 3: 3 cold + 3 lean resumes ---
echo "=== PHASE3 final resumes ===" | tee -a "$LOG"
RUN_META="$CLEAN_META" LEAN_IMAGE="$LEAN_IMAGE" \
  bash "$ROOT/scripts/run_final_scratch_batch_b300.sh" 2>&1 | tee -a "$LOG"

echo "=== b300x1_pipeline DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
