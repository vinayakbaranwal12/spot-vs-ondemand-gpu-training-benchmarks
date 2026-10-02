#!/usr/bin/env bash
# Batch: cold scratch 2-3 + lean snapshot scratch 1-3.
# Stops and exits non-zero on anomaly (capacity retry, lean pull, loss mismatch,
# R >50% off cold median, or any failure).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
RUN_META="$ROOT/logs/mi355x1/8B/clean-scratch-mi355x1-8B-20261001081617/run_meta.json"
RUN_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$RUN_META")
MODEL=8B
ENDPOINT="https://${SPACES_REGION_MEM1}.digitaloceanspaces.com"
SPACES_URI="s3://${SPACES_BUCKET_MEM1}/${RUN_ID}/ckpt-step-100"
SIZE="gpu-mi355x1-288gb-spot"
REGION="mem1"
COLD_IMAGE="gpu-amd-base"
LEAN_IMAGE="<snapshot-id>"
GPU="mi355x"
STRICT_REF="$ROOT/logs/mi355x1/8B/strict-loss-ref-20261001073603/strict_loss_reference.json"
LOG="$ROOT/logs/batch_scratch_resumes.log"
mkdir -p "$(dirname "$LOG")"
echo "=== batch scratch resumes $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee "$LOG"

anomaly() { echo "ANOMALY_STOP: $*" | tee -a "$LOG"; exit 3; }

cold_median_r() {
  python3 - <<'PY'
import json, statistics
from pathlib import Path
xs=[]
for p in Path("./logs/mi355x1/8B").glob("resume-cold-scratch-*/resume_result.json"):
  r=json.loads(p.read_text())
  if r.get("R_s") not in (None,""):
    xs.append(float(r["R_s"]))
print(statistics.median(xs) if xs else "")
PY
}

check_result() {
  local dir="$1" kind="$2"  # kind=cold|lean
  local result="$dir/resume_result.json"
  [[ -f "$result" ]] || anomaly "missing $result"
  local med
  med=$(cold_median_r)
  # Scratch-chain loss reference: first successful scratch resume (cold1), not boot-disk strict ref
  local SCRATCH_LOSS_REF="$ROOT/logs/mi355x1/8B/resume-cold-scratch-1-20261001095032/resume_result.json"
  python3 - "$result" "$kind" "$med" "$SCRATCH_LOSS_REF" <<'PY'
import json, sys
from pathlib import Path
result_path, kind, med_s, scratch_ref_path = sys.argv[1:5]
r=json.loads(Path(result_path).read_text())
if int(r.get("capacity_retries") or 0) > 0:
  print("capacity_retries", r.get("capacity_retries")); sys.exit(11)
if kind=="lean":
  pull=r.get("image_pull_s")
  local=r.get("image_already_local")
  try:
    pull_f=float(pull) if pull not in ("",None) else 999.0
  except Exception:
    pull_f=999.0
  if local is False or pull_f > 5.0:
    print(f"lean_pull_detected pull={pull} already_local={local}"); sys.exit(12)
# Cross-resume scratch loss identity (same ckpt-100 => same step-101 loss)
ref_p=Path(scratch_ref_path)
if ref_p.exists() and r.get("resume_loss") not in (None,"") and Path(result_path).resolve()!=ref_p.resolve():
  ref=json.loads(ref_p.read_text())
  b=float(ref["resume_loss"])
  a=float(r["resume_loss"])
  ok=abs(a-b)<=max(1e-3,1e-5*abs(b))
  r["loss_match_strict"]=ok
  r["loss_match"]=ok
  r["loss_match_note"]=f"scratch_cross_resume vs cold1 loss={b}"
  Path(result_path).write_text(json.dumps(r, indent=2)+"\n")
  if not ok:
    print(f"loss_mismatch a={a} b={b}"); sys.exit(13)
elif Path(result_path).resolve()==ref_p.resolve() or (ref_p.exists() and Path(result_path).name and "resume-cold-scratch-1-" in str(Path(result_path).parent)):
  # cold1 itself: mark as scratch reference anchor (not compared to boot-disk strict ref)
  r["loss_match"]=True
  r["loss_match_strict"]=True
  r["loss_match_note"]="scratch_resume_loss_anchor; boot-disk strict ref not applicable"
  Path(result_path).write_text(json.dumps(r, indent=2)+"\n")
try:
  med=float(med_s) if med_s not in ("",None) else 0.0
except Exception:
  med=0.0
R=float(r.get("R_s") or 0)
if med>0 and (R < 0.5*med or R > 1.5*med):
  print(f"R_outlier R={R} median={med}"); sys.exit(14)
print("CHECK_OK", kind, "R", R, "med", med, "pull", r.get("image_pull_s"), "local", r.get("image_already_local"))
PY
  local rc=$?
  case $rc in
    0) ;;
    11) anomaly "capacity retry in $dir" ;;
    12) anomaly "lean snapshot pulled image in $dir (invalid lean snap)" ;;
    13) anomaly "loss mismatch in $dir" ;;
    14) anomaly "R >50% off cold median in $dir" ;;
    *) anomaly "check_result failed rc=$rc for $dir" ;;
  esac
}

run_one() {
  local attempt="$1" variant="$2" image="$3"  # variant cold|snapshot
  local kind="$variant"
  [[ "$variant" == "snapshot" ]] && kind="lean"
  local TS; TS=$(date -u +%Y%m%d%H%M%S)
  local NAME EVENTS_DIR
  if [[ "$variant" == "cold" ]]; then
    NAME="spot-bench-resume-cold-scratch-mi355x1-${MODEL}-${attempt}-${TS}"
    EVENTS_DIR="$ROOT/logs/mi355x1/${MODEL}/resume-cold-scratch-${attempt}-${TS}"
  else
    NAME="spot-bench-resume-lean-scratch-mi355x1-${MODEL}-${attempt}-${TS}"
    EVENTS_DIR="$ROOT/logs/mi355x1/${MODEL}/resume-lean-scratch-${attempt}-${TS}"
  fi
  mkdir -p "$EVENTS_DIR"
  python3 - <<PY
import json, pathlib
meta=json.loads(open("$RUN_META").read())
meta.update({
  "variant": "$variant",
  "resume_attempt": int("$attempt"),
  "source_run_id": "$RUN_ID",
  "spaces_uri": "$SPACES_URI",
  "storage": "scratch",
  "exclude_from_final_results": False,
  "exclude_reason": "",
  "image": "$image",
  "snapshot_image_id": int("$image") if "$image".isdigit() else "",
  "note": f"$variant resume on /scratch; planned size=$SIZE region=$REGION image=$image",
})
pathlib.Path("$EVENTS_DIR/run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
PY
  echo "=== RUN $variant $attempt name=$NAME image=$image ===" | tee -a "$LOG"
  set +e
  bash "$ROOT/scripts/measure_resume.sh" \
    --name "$NAME" \
    --size "$SIZE" \
    --region "$REGION" \
    --image "$image" \
    --gpu "$GPU" \
    --spaces-uri "$SPACES_URI" \
    --spaces-endpoint "$ENDPOINT" \
    --events-dir "$EVENTS_DIR" \
    --model "$MODEL" \
    --run-meta "$RUN_META" \
    2>&1 | tee -a "$LOG" | tee "$EVENTS_DIR/driver.log"
  local RC=${PIPESTATUS[0]}
  set -e

  # Enrich image_already_local into resume_result from driver / env
  if [[ -f "$EVENTS_DIR/resume_result.json" ]]; then
    python3 - <<PY
import json, re
from pathlib import Path
d=Path("$EVENTS_DIR")
r=json.loads((d/"resume_result.json").read_text())
r["variant"] = "cold" if "$variant"=="cold" else "snapshot"
drv=(d/"driver.log").read_text() if (d/"driver.log").exists() else ""
m=re.search(r"image_already_local=(\S+)", drv)
if m:
  val=m.group(1)
  r["image_already_local"] = val in ("1","True","true")
m2=re.search(r"RESUME_SCRATCH=(\S+).*image_already_local=(\S+)", drv)
# also from events env_split
for line in (d/"events.jsonl").read_text().splitlines() if (d/"events.jsonl").exists() else []:
  e=json.loads(line)
  if e.get("event")=="env_split" and "image_already_local" in e:
    r["image_already_local"]=e["image_already_local"]
(d/"resume_result.json").write_text(json.dumps(r, indent=2)+"\n")
meta=json.loads((d/"run_meta.json").read_text())
for k in ("image_size_bytes","steps_lost","step_time_s","R_s","loss_match","loss_match_strict","storage","image_pull_s","image_already_local"):
  if k in r and r[k] not in ("", None):
    meta[k]=r[k]
meta["storage"]="scratch"
meta["exclude_from_final_results"]=False
(d/"run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
PY
  fi

  DID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("droplet_id",""))' "$EVENTS_DIR/resume_result.json" 2>/dev/null || true)
  if [[ -z "$DID" ]]; then
    DID=$(grep -E '^RESUME_DROPLET_ID=' "$EVENTS_DIR/driver.log" 2>/dev/null | tail -1 | cut -d= -f2 || true)
  fi
  if [[ -z "$DID" ]]; then
    DID=$(grep -E '^CREATED_ID=' "$EVENTS_DIR/driver.log" 2>/dev/null | tail -1 | cut -d= -f2 || true)
  fi

  # Ensure manifest has droplet then delete by ID
  if [[ -n "$DID" ]]; then
    python3 - <<PY
import json, datetime
from pathlib import Path
path=Path("$ROOT/manifest.json")
m=json.loads(path.read_text())
did=int("$DID")
ids={int(d["id"]) for d in m.get("droplets",[])}
if did not in ids:
  m.setdefault("droplets",[]).append({
    "id": did,
    "name": "$NAME",
    "size": "$SIZE",
    "region": "$REGION",
    "image": "$image",
    "status": "active",
    "created_at": datetime.datetime.utcnow().isoformat()+"Z",
    "deleted_at": None,
    "tag": "spot-bench",
  })
  path.write_text(json.dumps(m, indent=2)+"\n")
  print("manifest_added", did)
else:
  print("manifest_has", did)
PY
    echo "Deleting Droplet id=$DID" | tee -a "$LOG"
    bash "$ROOT/scripts/provision.sh" delete --id "$DID" || doctl compute droplet delete "$DID" --force || true
    # mark deleted if provision refuse
    python3 - <<PY
import json, datetime
from pathlib import Path
path=Path("$ROOT/manifest.json")
m=json.loads(path.read_text())
did=int("$DID")
for d in m.get("droplets",[]):
  if int(d.get("id",-1))==did:
    d["status"]="deleted"
    if not d.get("deleted_at"):
      d["deleted_at"]=datetime.datetime.utcnow().isoformat()+"Z"
    if d.get("created_at") and d.get("deleted_at"):
      try:
        c=datetime.datetime.fromisoformat(d["created_at"].replace("Z","+00:00"))
        e=datetime.datetime.fromisoformat(d["deleted_at"].replace("Z","+00:00"))
        d["runtime_s"]=int((e-c).total_seconds())
      except Exception:
        pass
path.write_text(json.dumps(m, indent=2)+"\n")
print("manifest_deleted", did)
PY
  fi

  if [[ $RC -ne 0 ]]; then
    # capacity?
    if grep -qiE 'capacity|no stock|unavailable|CREATE FAILED' "$EVENTS_DIR/driver.log"; then
      anomaly "create/capacity failure for $NAME (see driver.log)"
    fi
    anomaly "measure_resume failed rc=$RC for $NAME"
  fi
  check_result "$EVENTS_DIR" "$kind"
  python3 "$ROOT/scripts/analyze.py" --root "$ROOT" | tee -a "$LOG"
  echo "OK $variant $attempt dir=$EVENTS_DIR" | tee -a "$LOG"
}

# Spec gate: verify planned constants
echo "PLANNED size=$SIZE region=$REGION cold_image=$COLD_IMAGE lean_image=$LEAN_IMAGE tag=spot-bench" | tee -a "$LOG"

run_one 3 cold "$COLD_IMAGE"
run_one 1 snapshot "$LEAN_IMAGE"
run_one 2 snapshot "$LEAN_IMAGE"
run_one 3 snapshot "$LEAN_IMAGE"

echo BATCH_SCRATCH_RESUMES_DONE | tee -a "$LOG"
