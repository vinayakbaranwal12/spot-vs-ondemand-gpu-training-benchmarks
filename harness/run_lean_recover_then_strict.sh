#!/usr/bin/env bash
# Recover lean1 on existing Droplet, run lean 2-3 with create (no --wait) + poll,
# then scratch strict-loss ref on lean3 Droplet before delete.
set -euo pipefail
ROOT="."
cd "$ROOT"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

LOG="$ROOT/logs/lean_recover_then_strict.nohup"
echo "=== lean_recover_then_strict start $(date -u +%Y-%m-%dT%H:%M:%SZ) ROOT=$ROOT ==="

# Heartbeat every 60s (Mac sleep survival relies on outer caffeinate)
(
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/batch_heartbeat"
    echo "heartbeat $(cat "$ROOT/logs/batch_heartbeat") pid=$$" >> "$ROOT/logs/batch_heartbeat.log"
    sleep 60
  done
) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
RUN_META="$ROOT/logs/mi355x1/8B/clean-scratch-mi355x1-8B-20261001081617/run_meta.json"
RUN_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$RUN_META")
MODEL=8B
ENDPOINT="https://${SPACES_REGION_MEM1}.digitaloceanspaces.com"
SPACES_URI="s3://${SPACES_BUCKET_MEM1}/${RUN_ID}/ckpt-step-100"
SIZE="gpu-mi355x1-288gb-spot"
REGION="mem1"
LEAN_IMAGE="<snapshot-id>"
COLD_IMAGE="gpu-amd-base"
GPU="mi355x"

anomaly() { echo "ANOMALY_STOP: $*"; exit 3; }

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

check_lean() {
  local dir="$1"
  local result="$dir/resume_result.json"
  [[ -f "$result" ]] || anomaly "missing $result"
  local med; med=$(cold_median_r)
  local SCRATCH_LOSS_REF="$ROOT/logs/mi355x1/8B/resume-cold-scratch-1-20261001095032/resume_result.json"
  set +e
  python3 - "$result" "$med" "$SCRATCH_LOSS_REF" <<'PY'
import json, sys
from pathlib import Path
result_path, med_s, scratch_ref_path = sys.argv[1:4]
r=json.loads(Path(result_path).read_text())
rc=0
if int(r.get("capacity_retries") or 0) > 0:
  print("capacity_retries", r.get("capacity_retries")); rc=11
pull=r.get("image_pull_s"); local=r.get("image_already_local")
try:
  pull_f=float(pull) if pull not in ("",None) else 999.0
except Exception:
  pull_f=999.0
# Enrich already_local from events if missing
if local is None:
  for line in (Path(result_path).parent/"events.jsonl").read_text().splitlines() if (Path(result_path).parent/"events.jsonl").exists() else []:
    e=json.loads(line)
    if e.get("event")=="env_split" and "image_already_local" in e:
      local=e["image_already_local"]; r["image_already_local"]=local
if local is False or pull_f > 5.0:
  print(f"lean_pull_detected pull={pull} already_local={local}"); rc=12
ref_p=Path(scratch_ref_path)
if ref_p.exists() and r.get("resume_loss") not in (None,""):
  ref=json.loads(ref_p.read_text())
  b=float(ref["resume_loss"]); a=float(r["resume_loss"])
  ok=abs(a-b)<=max(1e-3,1e-5*abs(b))
  r["loss_match"]=ok
  # provisional cross-resume only; scratch strict overwrites loss_match_strict later
  r["loss_match_note"]=f"scratch_cross_resume vs cold1 loss={b}"
  Path(result_path).write_text(json.dumps(r, indent=2)+"\n")
  if not ok:
    print(f"loss_mismatch a={a} b={b}"); rc=13
try:
  med=float(med_s) if med_s not in ("",None) else 0.0
except Exception:
  med=0.0
R=float(r.get("R_s") or 0)
# Cold 2/3 R omitted pip (was unaccounted). Lean/cold4 include pip_install_s in R.
# Compare like-for-like against cold median by subtracting pip when present.
pip_f=float(r.get("pip_install_s") or 0)
R_cmp=R-pip_f
if med>0 and (R_cmp < 0.5*med or R_cmp > 1.5*med):
  print(f"R_outlier R={R} R_cmp={R_cmp} median={med} pip={pip_f}"); rc=14
print("CHECK_OK lean R", R, "R_cmp", R_cmp, "med", med, "pull", pull, "local", local, "pip", pip_f)
Path(result_path).write_text(json.dumps(r, indent=2)+"\n")
sys.exit(rc)
PY
  local rc=$?
  set -e
  case $rc in
    0) ;;
    11) anomaly "capacity retry in $dir" ;;
    12) anomaly "lean snapshot pulled image in $dir (invalid lean snap)" ;;
    13) anomaly "loss mismatch in $dir" ;;
    14) anomaly "R >50% off cold median in $dir" ;;
    *) anomaly "check_lean failed rc=$rc for $dir" ;;
  esac
}

manifest_upsert() {
  local did="$1" name="$2" image="$3" created="$4"
  python3 - <<PY
import json, datetime
from pathlib import Path
path=Path("$ROOT/manifest.json")
m=json.loads(path.read_text())
did=int("$did")
for d in m.get("droplets",[]):
  if int(d.get("id",-1))==did:
    d["status"]="active"; d["name"]="$name"; d["image"]="$image"
    if not d.get("created_at"): d["created_at"]="$created"
    path.write_text(json.dumps(m, indent=2)+"\n"); print("manifest_has", did); raise SystemExit
m.setdefault("droplets",[]).append({
  "id": did, "name": "$name", "size": "$SIZE", "region": "$REGION", "image": "$image",
  "status": "active", "created_at": "$created", "deleted_at": None, "tag": "spot-bench",
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
path.write_text(json.dumps(m, indent=2)+"\n")
print("manifest_deleted", did)
PY
}

create_no_wait() {
  local name="$1" image="$2"
  # No --wait: returns ID quickly so the durable script cannot die mid-wait.
  bash "$ROOT/scripts/provision.sh" create \
    --name "$name" --size "$SIZE" --region "$REGION" --image "$image" 2>&1 | tee /tmp/bench_create_nowait.out
  local did
  did=$(awk -F= '/^CREATED_ID=/{print $2}' /tmp/bench_create_nowait.out | tail -1)
  [[ -n "$did" ]] || anomaly "create returned no CREATED_ID for $name"
  echo "$did"
}

wait_active_ssh() {
  local did="$1"
  local st ip
  for i in $(seq 1 80); do
    st=$(doctl compute droplet get "$did" -o json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["status"])')
    echo "poll=$i status=$st id=$did"
    [[ "$st" == "active" ]] && break
    sleep 15
  done
  [[ "$st" == "active" ]] || anomaly "droplet $did never active"
  ip=$(bash "$ROOT/scripts/provision.sh" wait-ssh --id "$did" --gpu-tool rocm-smi 2>/dev/null | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/{ip=$0} END{print ip}')
  [[ -n "$ip" ]] || anomaly "no ssh ip for $did"
  echo "$ip"
}

run_existing() {
  local did="$1" ip="$2" provision_s="$3" events_dir="$4"
  stdbuf -oL -eL bash "$ROOT/scripts/measure_resume_existing.sh" \
    --droplet-id "$did" --ip "$ip" --provision-s "$provision_s" \
    --events-dir "$events_dir" --model "$MODEL" \
    --run-meta "$RUN_META" \
    --spaces-uri "$SPACES_URI" \
    --variant snapshot
  test -f "$events_dir/resume_result.json"
  # Enrich image_already_local
  python3 - <<PY
import json, re
from pathlib import Path
d=Path("$events_dir")
r=json.loads((d/"resume_result.json").read_text())
r["variant"]="snapshot"; r["storage"]="scratch"
drv=(d/"driver.log").read_text() if (d/"driver.log").exists() else ""
m=re.search(r"image_already_local=(\S+)", drv)
if m: r["image_already_local"]=m.group(1) in ("1","True","true")
for line in (d/"events.jsonl").read_text().splitlines() if (d/"events.jsonl").exists() else []:
  e=json.loads(line)
  if e.get("event")=="env_split" and "image_already_local" in e:
    r["image_already_local"]=e["image_already_local"]
(d/"resume_result.json").write_text(json.dumps(r, indent=2)+"\n")
meta=json.loads((d/"run_meta.json").read_text())
for k in ("R_s","storage","loss_match","loss_match_strict","step_time_s","steps_lost","image_pull_s","image_already_local"):
  if k in r and r[k] not in ("",None): meta[k]=r[k]
meta["storage"]="scratch"; meta["exclude_from_final_results"]=False
(d/"run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
print("ENRICHED", r.get("R_s"), "pull", r.get("image_pull_s"), "local", r.get("image_already_local"))
PY
  check_lean "$events_dir"
}

# ---------- LEAN 1 recover ----------
LEAN1_DID="<droplet-id>"
LEAN1_IP="<droplet-ip>"
LEAN1_DIR=logs/mi355x1/8B/resume-lean-scratch-1-20261001141701
if [[ -f "$LEAN1_DIR/resume_result.json" ]]; then
  echo "LEAN1 already has result; skipping measure"
  # Fix provision_s if it baked in operator idle (create→SSH was ~2 min, not hours)
  python3 - <<'PY'
import json
from pathlib import Path
from datetime import datetime
d=Path("logs/mi355x1/8B/resume-lean-scratch-1-20261001141701")
r=json.loads((d/"resume_result.json").read_text())
# Reconstruct timed provision from events if present; else use 120s cap for snap boot
ev=[json.loads(l) for l in (d/"events.jsonl").read_text().splitlines() if l.strip()]
start=next(e for e in ev if e.get("event")=="provision_start")
# Prefer first env_start as SSH-ready proxy when provision_end includes idle
env=next((e for e in ev if e.get("event")=="env_start"), None)
end=next((e for e in ev if e.get("event")=="provision_end"), None)
def parse(s):
  return datetime.fromisoformat(s.replace("Z","+00:00"))
ps_old=float(r.get("provision_s") or 0)
if env:
  ps=int((parse(env["wall_utc"])-parse(start["wall_utc"])).total_seconds())
else:
  ps=min(int(ps_old), 180) if ps_old else 120
idle=max(0, int(ps_old-ps)) if ps_old>ps else 0
if ps_old > 600:  # clearly includes idle
  delta=ps_old-ps
  r["provision_s"]=float(ps)
  r["provision_s_note"]=f"de-idled: was {ps_old}s wall including operator gap; timed create-to-env_start={ps}s"
  r["R_s"]=float(r["R_s"])-delta
  (d/"resume_result.json").write_text(json.dumps(r, indent=2)+"\n")
  print("LEAN1_PROVISION_FIXED", ps, "idle_removed", delta, "R", r["R_s"])
  # manifest idle
  import json as J
  from pathlib import Path as P
  mp=P("manifest.json"); m=J.loads(mp.read_text())
  for dd in m["droplets"]:
    if str(dd.get("id",""))=="<droplet-id>":
      dd["idle_s"]=max(int(dd.get("idle_s") or 0), idle)
      dd["idle_s_note"]="driver dead / Mac-sleep gap before recovered lean1 measure"
  mp.write_text(J.dumps(m, indent=2)+"\n")
PY
  check_lean "$LEAN1_DIR"
else
  PROV1=$(python3 - <<'PY'
import datetime
start=datetime.datetime.fromisoformat('2026-10-01T14:17:01+00:00')
now=datetime.datetime.now(datetime.timezone.utc)
print(int((now-start).total_seconds()))
PY
)
  python3 - <<PY
import json, datetime
from pathlib import Path
p=Path("$LEAN1_DIR/events.jsonl")
rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
rows=[r for r in rows if r.get("event")!="provision_end"]
if not any(r.get("event")=="provision_start" for r in rows):
  rows.insert(0,{"event":"provision_start","wall_utc":"2026-10-01T14:17:01.758253Z","mono_s":0.0,"simulated_reclaim":True})
rows.append({"event":"provision_end","wall_utc":datetime.datetime.utcnow().isoformat()+"Z","mono_s":float("$PROV1"),"duration_s":int("$PROV1"),"droplet_id":$LEAN1_DID,"ip":"$LEAN1_IP","note":"recovered after batch create--wait drop"})
p.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
PY
  run_existing "$LEAN1_DID" "$LEAN1_IP" "$PROV1" "$LEAN1_DIR"
fi
delete_and_account "$LEAN1_DID"
python3 scripts/analyze.py --root .
echo LEAN1_DONE

# ---------- LEAN 2 and 3 ----------
for attempt in 2 3; do
  TS=$(date -u +%Y%m%d%H%M%S)
  NAME="spot-bench-resume-lean-scratch-mi355x1-${MODEL}-${attempt}-${TS}"
  EVENTS_DIR="$ROOT/logs/mi355x1/${MODEL}/resume-lean-scratch-${attempt}-${TS}"
  mkdir -p "$EVENTS_DIR"
  python3 - <<PY
import json, pathlib
meta=json.loads(open("$RUN_META").read())
meta.update({
  "variant": "snapshot", "resume_attempt": int("$attempt"), "source_run_id": "$RUN_ID",
  "spaces_uri": "$SPACES_URI", "storage": "scratch", "exclude_from_final_results": False,
  "exclude_reason": "", "image": "$LEAN_IMAGE", "snapshot_image_id": int("$LEAN_IMAGE"),
  "note": "snapshot resume on /scratch; planned size=$SIZE region=$REGION image=$LEAN_IMAGE",
})
pathlib.Path("$EVENTS_DIR/run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
PY
  echo "=== RUN lean $attempt name=$NAME ==="
  CREATE_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  python3 - <<PY
import json
from pathlib import Path
Path("$EVENTS_DIR/events.jsonl").write_text(json.dumps({
  "event":"provision_start","wall_utc":"$CREATE_START","mono_s":0.0,"simulated_reclaim":True
})+"\n")
PY
  DID=$(create_no_wait "$NAME" "$LEAN_IMAGE")
  manifest_upsert "$DID" "$NAME" "$LEAN_IMAGE" "$CREATE_START"
  IP=$(wait_active_ssh "$DID")
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
rows.append({"event":"provision_end","wall_utc":datetime.datetime.utcnow().isoformat()+"Z","mono_s":float("$PROV_S"),"duration_s":int("$PROV_S"),"droplet_id":int("$DID"),"ip":"$IP"})
p.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
PY
  run_existing "$DID" "$IP" "$PROV_S" "$EVENTS_DIR"
  echo "LEAN${attempt}_OK dir=$EVENTS_DIR did=$DID ip=$IP"

  if [[ "$attempt" -eq 3 ]]; then
    echo "=== scratch strict-loss ref on lean3 Droplet $DID $IP ==="
    bash "$ROOT/scripts/run_scratch_strict_loss_ref.sh" "$IP"
    echo STRICT_ON_LEAN3_DONE
  fi

  delete_and_account "$DID"
  python3 scripts/analyze.py --root .
done

# ---------- COLD 4 replacement for excluded cold 1 (timed provision) ----------
echo "=== COLD4 replacement for excluded cold1 ==="
TS=$(date -u +%Y%m%d%H%M%S)
NAME="spot-bench-resume-cold-scratch-mi355x1-${MODEL}-4-${TS}"
EVENTS_DIR="$ROOT/logs/mi355x1/${MODEL}/resume-cold-scratch-4-${TS}"
mkdir -p "$EVENTS_DIR"
python3 - <<PY
import json, pathlib
meta=json.loads(open("$RUN_META").read())
meta.update({
  "variant": "cold", "resume_attempt": 4, "source_run_id": "$RUN_ID",
  "spaces_uri": "$SPACES_URI", "storage": "scratch", "exclude_from_final_results": False,
  "exclude_reason": "", "image": "$COLD_IMAGE", "snapshot_image_id": "",
  "note": "cold4 replacement; provision timed create-to-SSH+GPU (no --wait reconstruct)",
  "replaces_run": "resume-cold-scratch-1-20261001095032",
})
pathlib.Path("$EVENTS_DIR/run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
PY
CREATE_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 - <<PY
import json
from pathlib import Path
Path("$EVENTS_DIR/events.jsonl").write_text(json.dumps({
  "event":"provision_start","wall_utc":"$CREATE_START","mono_s":0.0,"simulated_reclaim":True
})+"\n")
PY
DID=$(create_no_wait "$NAME" "$COLD_IMAGE")
manifest_upsert "$DID" "$NAME" "$COLD_IMAGE" "$CREATE_START"
IP=$(wait_active_ssh "$DID")
# Timed provision: create start -> SSH + GPU visible (wait-ssh already gated on rocm-smi)
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
rows.append({"event":"provision_end","wall_utc":datetime.datetime.utcnow().isoformat()+"Z","mono_s":float("$PROV_S"),"duration_s":int("$PROV_S"),"droplet_id":int("$DID"),"ip":"$IP","timed":True,"note":"create_no_wait then wait-ssh+rocm-smi"})
p.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
PY
# cold variant via existing helper but force variant cold
stdbuf -oL -eL bash "$ROOT/scripts/measure_resume_existing.sh" \
  --droplet-id "$DID" --ip "$IP" --provision-s "$PROV_S" \
  --events-dir "$EVENTS_DIR" --model "$MODEL" \
  --run-meta "$RUN_META" --spaces-uri "$SPACES_URI" --variant cold
test -f "$EVENTS_DIR/resume_result.json"
python3 - <<PY
import json, statistics
from pathlib import Path
d=Path("$EVENTS_DIR")
r=json.loads((d/"resume_result.json").read_text())
ref=json.loads(Path("logs/mi355x1/8B/resume-cold-scratch-2-20261001135013/resume_result.json").read_text())
a=float(r["resume_loss"]); b=float(ref["resume_loss"])
ok=abs(a-b)<=max(1e-3,1e-5*abs(b))
r.update(storage="scratch", loss_match=ok, steps_lost=0,
         loss_match_note=f"scratch_cross_resume vs cold2 loss={b}")
(d/"resume_result.json").write_text(json.dumps(r, indent=2)+"\n")
print("COLD4", r["R_s"], "loss_match", ok, "pip", r.get("pip_install_s"))
if not ok: raise SystemExit("LOSS_MISMATCH")
# R vs cold median of non-excluded colds
xs=[]
for p in Path("logs/mi355x1/8B").glob("resume-cold-scratch-*/resume_result.json"):
  meta=json.loads((p.parent/"run_meta.json").read_text()) if (p.parent/"run_meta.json").exists() else {}
  if meta.get("exclude_from_final_results"): continue
  rr=json.loads(p.read_text())
  if rr.get("R_s") not in (None,""): xs.append(float(rr["R_s"]))
import statistics as st
med=st.median(xs); R=float(r["R_s"])
if med>0 and (R<0.5*med or R>1.5*med): raise SystemExit(f"R_OUTLIER {R} {med}")
print("COLD4_OK")
PY
delete_and_account "$DID"
python3 scripts/analyze.py --root .
echo COLD4_DONE

echo ALL_LEAN_AND_STRICT_DONE
