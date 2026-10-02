#!/usr/bin/env bash
# Scratch strict-loss reference: continue clean-scratch ckpt-80 through step 101.
# Usage: run_scratch_strict_loss_ref.sh <IP>
set -euo pipefail
ROOT="."
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"
# shellcheck source=/dev/null
source "$ROOT/scripts/spaces_env_remote.sh"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
IP="${1:?ip}"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes)
SCP=(scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes)
ENDPOINT="https://${SPACES_REGION_MEM1}.digitaloceanspaces.com"
CLEAN_META="$ROOT/logs/mi355x1/8B/clean-scratch-mi355x1-8B-20261001081617/run_meta.json"
CLEAN_EVENTS="$ROOT/logs/mi355x1/8B/clean-scratch-mi355x1-8B-20261001081617/events.jsonl"
BATCH_SIZE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["batch_size"])' "$CLEAN_META")
GRAD_ACCUM=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("grad_accum",1))' "$CLEAN_META")
CKPT80="s3://${SPACES_BUCKET_MEM1}/clean-scratch-mi355x1-8B-20261001081617/ckpt-step-80"
OUT_DIR="$ROOT/logs/mi355x1/8B/strict-loss-ref-scratch-$(date -u +%Y%m%d%H%M%S)"
mkdir -p "$OUT_DIR"

# Ensure scripts present
"${SSH[@]}" "root@$IP" 'mkdir -p /opt/bench-scripts'
"${SCP[@]}" "$ROOT/scripts/"*.py "$ROOT/scripts/"*.sh "root@$IP:/opt/bench-scripts/"

CONTAINER=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/container.ref)
SCRATCH=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/scratch.path)

spaces_env_push "$IP"

echo "=== download scratch ckpt-step-80 ===" | tee "$OUT_DIR/driver.log"
CKPT_LOCAL="$SCRATCH/bench/ref-ckpt-80-scratch"
"${SSH[@]}" "root@$IP" bash -s <<REMOTE
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a
rm -rf '$CKPT_LOCAL'; mkdir -p '$CKPT_LOCAL'
s5cmd --endpoint-url '$ENDPOINT' cp '${CKPT80%/}/*' '$CKPT_LOCAL/'
REMOTE

REMOTE_EVENTS=/opt/bench/strict_loss_ref_scratch_events.jsonl
echo "=== reference train: scratch ckpt-80, batch=$BATCH_SIZE, through step 101 ===" | tee -a "$OUT_DIR/driver.log"
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
      --ckpt-dir ${SCRATCH}/bench/ckpts_ref_scratch \
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
root=Path("$ROOT")
events=[json.loads(l) for l in (out/"events.jsonl").read_text().splitlines() if l.strip()]
by_step={e["step"]: e["loss"] for e in events if e.get("event")=="step"}
orig={}
for line in Path("$CLEAN_EVENTS").read_text().splitlines():
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
  "storage": "scratch",
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
  "clean_run_id": "clean-scratch-mi355x1-8B-20261001081617",
}
(out/"strict_loss_reference.json").write_text(json.dumps(result, indent=2)+"\n")
# Stable pointer for resume rows
(root/"logs/mi355x1/8B/strict-loss-ref-scratch-latest.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
if not anchor_ok:
  raise SystemExit("ANCHOR_FAIL: step-100 reference loss does not match scratch clean run")
print("ANCHOR_OK")

# Apply loss_match_strict to every scratch resume row vs ref step-101
ref101f=float(ref101)
updated=[]
for p in sorted((root/"logs/mi355x1/8B").glob("resume-*-scratch-*/resume_result.json")):
  r=json.loads(p.read_text())
  if r.get("resume_loss") in (None,""):
    continue
  a=float(r["resume_loss"])
  ok=within(a, ref101f)
  r["loss_match_strict"]=ok
  # Keep cross-resume consistency as loss_match if already set; else set from strict
  if r.get("loss_match") in ("", None):
    r["loss_match"]=ok
  r["loss_match_strict_note"]=f"vs scratch strict ref step-101 loss={ref101f} (from ckpt-80 cont)"
  r["strict_ref_step_101"]=ref101f
  p.write_text(json.dumps(r, indent=2)+"\n")
  meta_p=p.parent/"run_meta.json"
  if meta_p.exists():
    meta=json.loads(meta_p.read_text())
    meta["loss_match_strict"]=ok
    meta["strict_ref_step_101"]=ref101f
    meta_p.write_text(json.dumps(meta, indent=2)+"\n")
  updated.append((str(p.parent.name), a, ok))
print("STRICT_APPLIED", updated)
PY
echo "STRICT_SCRATCH_DONE out=$OUT_DIR"
