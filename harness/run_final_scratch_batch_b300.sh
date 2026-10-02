#!/usr/bin/env bash
# B300 1x final scratch resumes: 3 cold + 3 lean (R includes pip_install_s).
# Requires clean 8B run_meta + lean snapshot image ID.
# Usage: RUN_META=... LEAN_IMAGE=... bash scripts/run_final_scratch_batch_b300.sh
set -euo pipefail
ROOT="."
cd "$ROOT"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

LOG="$ROOT/logs/final_scratch_batch_b300.nohup"
echo "=== final_scratch_batch_b300 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

(
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/batch_heartbeat"
    sleep 60
  done
) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

RUN_META="${RUN_META:?set RUN_META to clean-scratch-b300x1-8B run_meta.json}"
LEAN_IMAGE="${LEAN_IMAGE:?set LEAN_IMAGE to B300 lean snapshot id}"
[[ -f "$RUN_META" ]] || { echo "missing $RUN_META"; exit 2; }
RUN_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$RUN_META")
MODEL=8B
SPACES_URI="s3://${SPACES_BUCKET_MKC1}/${RUN_ID}/ckpt-step-100"
SPACES_ENDPOINT="https://${SPACES_REGION_MKC1}.digitaloceanspaces.com"
SIZE="gpu-b300x1-288gb-lc-spot"
REGION="mkc1"
COLD_IMAGE="gpu-h100x1-base"

anomaly() { echo "ANOMALY_STOP: $*" | tee -a "$LOG"; exit 3; }

check_result() {
  local dir="$1" kind="$2"
  local result="$dir/resume_result.json"
  [[ -f "$result" ]] || anomaly "missing $result"
  python3 - "$result" "$kind" <<'PY' || true
import json, sys
from pathlib import Path
result_path, kind = sys.argv[1:3]
r=json.loads(Path(result_path).read_text())
rc=0
if int(r.get("capacity_retries") or 0) > 0:
  print("capacity_retries_noted", r.get("capacity_retries"), "retry_wait_s", r.get("retry_wait_s"))
  # Capacity waits are expected spot risk; record, do not abort the row.
if kind=="lean":
  pull=r.get("image_pull_s"); local=r.get("image_already_local")
  try: pull_f=float(pull) if pull not in ("",None) else 999.0
  except Exception: pull_f=999.0
  if local is None:
    for line in (Path(result_path).parent/"events.jsonl").read_text().splitlines() if (Path(result_path).parent/"events.jsonl").exists() else []:
      e=json.loads(line)
      if e.get("event")=="env_split" and "image_already_local" in e:
        local=e["image_already_local"]; r["image_already_local"]=local
  if local is False or pull_f > 5.0:
    print(f"lean_pull_detected pull={pull} already_local={local}"); rc=12
if "pip_install_s" not in r or r.get("pip_install_s") in ("", None):
  print("missing pip_install_s"); rc=15
print("CHECK_OK", kind, "R", r.get("R_s"), "pip", r.get("pip_install_s"), "pull", r.get("image_pull_s"), "local", r.get("image_already_local"))
Path(result_path).write_text(json.dumps(r, indent=2)+"\n")
sys.exit(rc)
PY
  local rc=$?
  case $rc in
    0) ;;
    12) anomaly "lean snapshot pulled image in $dir" ;;
    15) anomaly "pip_install_s missing in $dir (R definition)" ;;
    *) anomaly "check failed rc=$rc for $dir" ;;
  esac
  python3 - "$result" <<'PY' || true
import json, statistics, sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
xs=[]
for p in Path("./logs/b300x1/8B").glob("resume-*-scratch-*/resume_result.json"):
  meta=json.loads((p.parent/"run_meta.json").read_text()) if (p.parent/"run_meta.json").exists() else {}
  if meta.get("exclude_from_final_results") or json.loads(p.read_text()).get("exclude_from_final_results"):
    continue
  if p.resolve()==Path(sys.argv[1]).resolve(): continue
  rr=json.loads(p.read_text())
  if rr.get("R_s") not in (None,""): xs.append(float(rr["R_s"]))
if len(xs)>=2:
  med=statistics.median(xs); R=float(r["R_s"])
  if R<0.5*med or R>1.5*med:
    print(f"R_outlier R={R} median={med}"); sys.exit(14)
print("R_OK", r.get("R_s"), "med", statistics.median(xs) if xs else None)
sys.exit(0)
PY
  local rc2=$?
  if [[ $rc2 -eq 14 ]]; then
    anomaly "R >50% off final median in $dir"
  fi
}

manifest_upsert() {
  local did="$1" name="$2" image="$3" created="$4"
  python3 - <<PY
import json
from pathlib import Path
path=Path("$ROOT/manifest.json")
m=json.loads(path.read_text())
did=int("$did")
for d in m.get("droplets",[]):
  if int(d.get("id",-1))==did:
    d.update({"status":"active","name":"$name","image":"$image","idle_s":d.get("idle_s",0)})
    if not d.get("created_at"): d["created_at"]="$created"
    path.write_text(json.dumps(m, indent=2)+"\n"); print("manifest_has", did); raise SystemExit
m.setdefault("droplets",[]).append({
  "id": did, "name": "$name", "size": "$SIZE", "region": "$REGION", "image": "$image",
  "status": "active", "created_at": "$created", "deleted_at": None, "tag": "spot-bench", "idle_s": 0,
})
path.write_text(json.dumps(m, indent=2)+"\n")
print("manifest_added", did)
PY
}

delete_and_account() {
  local did="$1"
  bash "$ROOT/scripts/provision.sh" delete --id "$did" || doctl compute droplet delete "$did" --force || true
  python3 - <<PY
import json, datetime
from pathlib import Path
path=Path("$ROOT/manifest.json")
m=json.loads(path.read_text())
did=int("$did")
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
    if "idle_s" not in d: d["idle_s"]=0
path.write_text(json.dumps(m, indent=2)+"\n")
print("manifest_deleted", did)
PY
}

create_no_wait() {
  local name="$1" image="$2"
  local attempt=0 max_attempts=40 sleep_s=90
  local did=""
  CREATE_CAPACITY_RETRIES=0
  CREATE_RETRY_WAIT_S=0
  CREATE_CAPACITY_ERROR=""
  while [[ $attempt -lt $max_attempts ]]; do
    attempt=$((attempt+1))
    # Unique name on retries (DO rejects duplicate names; also avoids stale create state)
    local try_name="$name"
    if [[ $attempt -gt 1 ]]; then
      try_name="${name}-r${attempt}-$(date -u +%H%M%S)"
    fi
    # All chatter to stderr/log only — stdout must be exactly the droplet ID.
    echo "CREATE_ATTEMPT $attempt/$max_attempts name=$try_name" | tee -a "$LOG" >&2
    bash "$ROOT/scripts/provision.sh" create \
      --name "$try_name" --size "$SIZE" --region "$REGION" --image "$image" \
      > /tmp/bench_b300_create_nowait.out 2>&1 || true
    cat /tmp/bench_b300_create_nowait.out >> "$LOG"
    did=$(awk -F= '/^CREATED_ID=/{print $2}' /tmp/bench_b300_create_nowait.out | tail -1)
    if [[ -n "$did" ]]; then
      echo "CREATED_OK id=$did name=$try_name" | tee -a "$LOG" >&2
      printf '%s' "$did"
      return 0
    fi
    if grep -qiE 'not available in this region|DropletLimit|capacity|currently unavailable' /tmp/bench_b300_create_nowait.out; then
      CREATE_CAPACITY_RETRIES=$((CREATE_CAPACITY_RETRIES+1))
      CREATE_RETRY_WAIT_S=$((CREATE_RETRY_WAIT_S+sleep_s))
      CREATE_CAPACITY_ERROR=$(grep -iE 'not available|422|capacity|unavailable' /tmp/bench_b300_create_nowait.out | tail -1 | tr -d '\r')
      echo "CAPACITY_WAIT sleep ${sleep_s}s (attempt $attempt)" | tee -a "$LOG" >&2
      sleep "$sleep_s"
      continue
    fi
    anomaly "create returned no CREATED_ID for $try_name (non-capacity error)"
  done
  anomaly "create capacity exhausted after $max_attempts attempts for $name"
}

wait_active_ssh() {
  local did="$1"
  bash "$ROOT/scripts/provision.sh" wait-ssh --id "$did" --gpu-tool nvidia-smi 2>>"$LOG" \
    | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/{ip=$0} END{print ip}'
}

run_one() {
  local attempt="$1" variant="$2" image="$3"
  local kind="$variant"; [[ "$variant" == "snapshot" ]] && kind="lean"
  local TS; TS=$(date -u +%Y%m%d%H%M%S)
  local NAME EVENTS_DIR
  if [[ "$variant" == "cold" ]]; then
    NAME="spot-bench-resume-cold-scratch-b300x1-${MODEL}-${attempt}-${TS}"
    EVENTS_DIR="$ROOT/logs/b300x1/${MODEL}/resume-cold-scratch-${attempt}-${TS}"
  else
    NAME="spot-bench-resume-lean-scratch-b300x1-${MODEL}-${attempt}-${TS}"
    EVENTS_DIR="$ROOT/logs/b300x1/${MODEL}/resume-lean-scratch-${attempt}-${TS}"
  fi
  mkdir -p "$EVENTS_DIR"
  python3 - <<PY
import json, pathlib
meta=json.loads(open("$RUN_META").read())
meta.update({
  "variant": "$variant", "resume_attempt": int("$attempt"), "source_run_id": "$RUN_ID",
  "spaces_uri": "$SPACES_URI", "storage": "scratch", "exclude_from_final_results": False,
  "exclude_reason": "", "image": "$image",
  "snapshot_image_id": int("$image") if str("$image").isdigit() else "",
  "note": "$variant resume attempt $attempt; R includes pip_install_s; NGC 25.11 native on driver 580; provision timed live",
  "gpu": "B300-1x",
})
pathlib.Path("$EVENTS_DIR/run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
PY
  echo "=== RUN $variant $attempt name=$NAME ===" | tee -a "$LOG"
  CREATE_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  python3 - <<PY
import json
from pathlib import Path
Path("$EVENTS_DIR/events.jsonl").write_text(json.dumps({
  "event":"provision_start","wall_utc":"$CREATE_START","mono_s":0.0,"simulated_reclaim":True
})+"\n")
PY
  local DID IP PROV_S
  DID=$(create_no_wait "$NAME" "$image")
  # Persist capacity from create_no_wait into the events dir for later enrichment.
  python3 - <<PY
import json
from pathlib import Path
p=Path("$EVENTS_DIR/capacity_create.json")
p.write_text(json.dumps({
  "capacity_retries": int("$CREATE_CAPACITY_RETRIES"),
  "retry_wait_s": float("$CREATE_RETRY_WAIT_S"),
  "capacity_error": """$CREATE_CAPACITY_ERROR""",
}, indent=2)+"\n")
PY
  manifest_upsert "$DID" "$NAME" "$image" "$CREATE_START"
  IP=$(wait_active_ssh "$DID")
  [[ -n "$IP" ]] || anomaly "no IP for $DID"
  PROV_S=$(python3 - <<PY
import datetime
start=datetime.datetime.fromisoformat("$CREATE_START".replace("Z","+00:00"))
now=datetime.datetime.now(datetime.timezone.utc)
print(int((now-start).total_seconds()))
PY
)
  python3 - <<PY
import json, datetime
from pathlib import Path
p=Path("$EVENTS_DIR/events.jsonl")
rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
rows=[r for r in rows if r.get("event")!="provision_end"]
rows.append({"event":"provision_end","wall_utc":datetime.datetime.utcnow().isoformat()+"Z","mono_s":float("$PROV_S"),"duration_s":int("$PROV_S"),"droplet_id":int("$DID"),"ip":"$IP","timed":True,"note":"create_no_wait then wait-ssh+nvidia-smi"})
p.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
PY
  stdbuf -oL -eL bash "$ROOT/scripts/measure_resume_existing.sh" \
    --droplet-id "$DID" --ip "$IP" --provision-s "$PROV_S" \
    --events-dir "$EVENTS_DIR" --model "$MODEL" --gpu b300 \
    --run-meta "$RUN_META" --spaces-uri "$SPACES_URI" \
    --spaces-endpoint "$SPACES_ENDPOINT" \
    --variant "$variant" | tee -a "$LOG"
  test -f "$EVENTS_DIR/resume_result.json"
  python3 - <<PY
import json, re
from pathlib import Path
d=Path("$EVENTS_DIR")
r=json.loads((d/"resume_result.json").read_text())
r["variant"]="$variant"; r["storage"]="scratch"; r["exclude_from_final_results"]=False
r["gpu"]="B300-1x"
cap_p=d/"capacity_create.json"
if cap_p.exists():
  cap=json.loads(cap_p.read_text())
  r["capacity_retries"]=cap.get("capacity_retries",0)
  r["retry_wait_s"]=cap.get("retry_wait_s",0)
  r["capacity_error"]=cap.get("capacity_error","")
for line in (d/"events.jsonl").read_text().splitlines() if (d/"events.jsonl").exists() else []:
  e=json.loads(line)
  if e.get("event")=="env_split" and "image_already_local" in e:
    r["image_already_local"]=e["image_already_local"]
refs=sorted(Path("logs/b300x1/8B").glob("resume-*-scratch-*/resume_result.json"))
for p in refs:
  meta=json.loads((p.parent/"run_meta.json").read_text()) if (p.parent/"run_meta.json").exists() else {}
  if meta.get("exclude_from_final_results") or json.loads(p.read_text()).get("exclude_from_final_results"):
    continue
  if p.resolve()==d.resolve()/"resume_result.json": continue
  ref=json.loads(p.read_text())
  if ref.get("resume_loss") in (None,""): continue
  a=float(r["resume_loss"]); b=float(ref["resume_loss"])
  ok=abs(a-b)<=max(1e-3,1e-5*abs(b))
  r["loss_match"]=ok
  r["loss_match_note"]=f"scratch_cross_resume vs {p.parent.name} loss={b}"
  break
(d/"resume_result.json").write_text(json.dumps(r, indent=2)+"\n")
meta=json.loads((d/"run_meta.json").read_text())
for k in ("R_s","storage","loss_match","loss_match_strict","step_time_s","steps_lost","image_pull_s","image_already_local","pip_install_s"):
  if k in r and r[k] not in ("",None): meta[k]=r[k]
meta["storage"]="scratch"; meta["exclude_from_final_results"]=False
(d/"run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
print("ENRICHED", r.get("R_s"), "pip", r.get("pip_install_s"), "local", r.get("image_already_local"))
PY
  check_result "$EVENTS_DIR" "$kind"
  echo "OK_$variant$attempt DID=$DID IP=$IP DIR=$EVENTS_DIR" | tee -a "$LOG"
  LAST_DID="$DID"
  LAST_IP="$IP"
  LAST_DIR="$EVENTS_DIR"
}

echo "PLANNED size=$SIZE region=$REGION cold_image=$COLD_IMAGE lean_image=$LEAN_IMAGE spaces=$SPACES_URI" | tee -a "$LOG"

# Reap any leftover Droplet from a prior interrupted attempt (e.g. cold1 finished, delete never ran).
while read -r did; do
  [[ -n "$did" ]] || continue
  echo "REAP orphan droplet $did" | tee -a "$LOG"
  delete_and_account "$did"
done < <(python3 - <<'PY'
import json, subprocess
from pathlib import Path
root=Path(".")
live=set()
try:
  out=subprocess.check_output(["doctl","compute","droplet","list","--tag-name","spot-bench","--format","ID","--no-header"], text=True)
  live={int(x) for x in out.split() if x.strip().isdigit()}
except Exception:
  pass
for p in root.glob("logs/b300x1/8B/resume-*-scratch-*/resume_result.json"):
  r=json.loads(p.read_text())
  did=r.get("droplet_id")
  if did and int(did) in live:
    print(int(did))
PY
)

for i in 1 2 3; do
  if ls logs/b300x1/8B/resume-cold-scratch-${i}-*/resume_result.json >/dev/null 2>&1; then
    echo "SKIP cold $i already done" | tee -a "$LOG"
  else
    run_one "$i" cold "$COLD_IMAGE"
    delete_and_account "$LAST_DID"
    python3 scripts/analyze.py --root . | tee -a "$LOG" || true
  fi
done

for i in 1 2 3; do
  if ls logs/b300x1/8B/resume-lean-scratch-${i}-*/resume_result.json >/dev/null 2>&1; then
    echo "SKIP lean $i already done" | tee -a "$LOG"
  else
    run_one "$i" snapshot "$LEAN_IMAGE"
    if [[ "$i" -eq 2 ]]; then
      echo "=== scratch strict-loss ref on lean2 $LAST_DID $LAST_IP ===" | tee -a "$LOG"
      bash "$ROOT/scripts/run_scratch_strict_loss_ref_b300.sh" "$LAST_IP" "$RUN_META" | tee -a "$LOG" || anomaly "strict loss failed"
    fi
    delete_and_account "$LAST_DID"
    python3 scripts/analyze.py --root . | tee -a "$LOG" || true
  fi
done

# ---------- Final table + checkpoint medians ----------
python3 - <<'PY' | tee -a "$LOG" | tee results/final_cold_vs_lean_b300x1.txt
import json, statistics
from pathlib import Path

root=Path(".")
phases=["provision_s","env_s","download_s","pip_install_s","load_s","first_step_s","R_s"]
rows=[]
for p in sorted((root/"logs/b300x1/8B").glob("resume-*-scratch-*/resume_result.json")):
  meta=json.loads((p.parent/"run_meta.json").read_text()) if (p.parent/"run_meta.json").exists() else {}
  r=json.loads(p.read_text())
  if meta.get("exclude_from_final_results") or r.get("exclude_from_final_results"):
    continue
  variant=r.get("variant") or meta.get("variant")
  kind="lean" if variant in ("snapshot","lean") else "cold"
  row={"kind":kind,"run":p.parent.name,"variant":variant,
       "loss_match_strict": r.get("loss_match_strict")}
  for k in phases:
    v=r.get(k,"")
    try: row[k]=float(v)
    except Exception: row[k]=None
  rows.append(row)

print("FINAL_ROWS", len(rows))
for r in rows:
  print(f"  {r['kind']:4} {r['run']} R={r['R_s']} pip={r['pip_install_s']} prov={r['provision_s']} dl={r['download_s']} loss_match_strict={r['loss_match_strict']}")

def med(xs):
  xs=[x for x in xs if x is not None]
  return statistics.median(xs) if xs else None
def mn(xs):
  xs=[x for x in xs if x is not None]
  return min(xs) if xs else None
def mx(xs):
  xs=[x for x in xs if x is not None]
  return max(xs) if xs else None

print("\n=== Cold vs Lean medians (final only; min/max in brackets) ===")
print(f"{'phase':24} {'cold_med [min,max]':>26} {'lean_med [min,max]':>26} {'delta_med':>10} {'lean/cold':>10}")
for k in phases:
  cvals=[r[k] for r in rows if r["kind"]=="cold"]
  lvals=[r[k] for r in rows if r["kind"]=="lean"]
  c,l=med(cvals),med(lvals)
  delta=(l-c) if (c is not None and l is not None) else None
  ratio=(l/c) if (c not in (None,0) and l is not None) else None
  def bracket(m,a,b):
    if m is None: return "-"
    return f"{m:.3f} [{a:.3f},{b:.3f}]"
  print(f"{k:24} {bracket(c,mn(cvals),mx(cvals)):>26} {bracket(l,mn(lvals),mx(lvals)):>26} "
        f"{(f'{delta:.3f}' if delta is not None else '-'):>10} {(f'{ratio:.3f}' if ratio is not None else '-'):>10}")

cold_n=sum(1 for r in rows if r["kind"]=="cold")
lean_n=sum(1 for r in rows if r["kind"]=="lean")
print("\nCAVEAT: Spaces bucket <bucket> is co-located in mkc1 with B300 Droplets")
print("(cross_region=false). Compare download_s here to MI355X final (atl1 bucket / mem1 Droplets)")
print("to isolate how much of MI355X download_s was cross-region transfer.")
print(f"\nn_cold={cold_n} n_lean={lean_n}")
if cold_n<3 or lean_n<3:
  raise SystemExit(f"INCOMPLETE final set cold={cold_n} lean={lean_n}")
print("B300_1X_COMPLETE")
PY

python3 - <<'PY' | tee -a "$LOG" | tee results/checkpoint_medians_b300x1.txt
import json, statistics
from pathlib import Path
from collections import defaultdict

root=Path(".")

def collect(model):
  # Prefer clean-scratch runs; keep one row per step with upload_s populated.
  paths=sorted((root/f"logs/b300x1/{model}").glob("clean-scratch-b300x1-*/events.jsonl"))
  if not paths:
    return []
  ev=[]
  for line in paths[-1].read_text().splitlines():
    if line.strip():
      ev.append(json.loads(line))
  by_step={}
  for e in ev:
    if e.get("event")=="ckpt_uploaded_and_local_deleted":
      step=e.get("step")
      by_step[step]={
        "model": model,
        "step": step,
        "ckpt_bytes": e.get("ckpt_bytes"),
        "local_save_s": e.get("local_save_s"),
        "upload_s": e.get("upload_s"),
      }
  for e in ev:
    if e.get("event")=="ckpt_save_end":
      step=e.get("step")
      if step not in by_step:
        by_step[step]={
          "model": model,
          "step": step,
          "ckpt_bytes": e.get("ckpt_bytes"),
          "local_save_s": e.get("duration_s"),
          "upload_s": None,
        }
  rows=list(by_step.values())
  with_upload=[r for r in rows if r.get("upload_s") not in (None,"")]
  return sorted(with_upload or rows, key=lambda r: (r["step"] or 0))

all_rows=collect("1.7B")+collect("8B")
print("CHECKPOINT_ROWS", len(all_rows), "(deduped; prefer upload_s)")
for r in all_rows:
  print(f"  {r['model']:4} step={r['step']} ckpt_bytes={r['ckpt_bytes']} local_save_s={r['local_save_s']} upload_s={r['upload_s']}")

def med(xs):
  xs=[float(x) for x in xs if x not in (None,"")]
  return statistics.median(xs) if xs else None

print("\n=== Checkpoint medians (clean scratch; by model) ===")
print(f"{'model':6} {'n':>3} {'local_save_s_med':>16} {'upload_s_med':>14} {'ckpt_GiB_med':>12}")
for model in ("1.7B","8B"):
  rs=[r for r in all_rows if r["model"]==model]
  gb=[(r["ckpt_bytes"] or 0)/1024**3 for r in rs]
  print(f"{model:6} {len(rs):3} {med([r['local_save_s'] for r in rs]) or float('nan'):16.3f} "
        f"{med([r['upload_s'] for r in rs]) or float('nan'):14.3f} {med(gb) or float('nan'):12.3f}")

# Overall 8B median C/U used for overhead model
rs8=[r for r in all_rows if r["model"]=="8B"]
print("\n8B_median_local_save_s", med([r["local_save_s"] for r in rs8]))
print("8B_median_upload_s", med([r["upload_s"] for r in rs8]))
print("B300_CKPT_MEDIANS_DONE")
PY

echo "=== B300 FINAL BATCH DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
