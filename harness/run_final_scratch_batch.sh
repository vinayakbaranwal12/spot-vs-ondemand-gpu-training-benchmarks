#!/usr/bin/env bash
# Final consistent MI355X 1x scratch resumes.
# Order: lean 2, lean 3, scratch strict on lean3, cold 4, cold 5, cold 6, lean 4.
# All final rows use R including pip_install_s; provision timed live (create no --wait + wait-ssh/GPU).
set -euo pipefail
ROOT="."
cd "$ROOT"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

LOG="$ROOT/logs/final_scratch_batch.nohup"
echo "=== final_scratch_batch start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

(
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/batch_heartbeat"
    sleep 60
  done
) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

RUN_META="$ROOT/logs/mi355x1/8B/clean-scratch-mi355x1-8B-20261001081617/run_meta.json"
RUN_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$RUN_META")
MODEL=8B
SPACES_URI="s3://${SPACES_BUCKET_MEM1}/${RUN_ID}/ckpt-step-100"
SIZE="gpu-mi355x1-288gb-spot"
REGION="mem1"
LEAN_IMAGE="<snapshot-id>"
COLD_IMAGE="gpu-amd-base"

anomaly() { echo "ANOMALY_STOP: $*" | tee -a "$LOG"; exit 3; }

final_median_r() {
  python3 - <<'PY'
import json, statistics
from pathlib import Path
xs=[]
for p in Path("./logs/mi355x1/8B").glob("resume-*-scratch-*/resume_result.json"):
  meta_p=p.parent/"run_meta.json"
  meta=json.loads(meta_p.read_text()) if meta_p.exists() else {}
  if meta.get("exclude_from_final_results"): continue
  r=json.loads(p.read_text())
  if r.get("exclude_from_final_results"): continue
  if r.get("R_s") not in (None,""): xs.append(float(r["R_s"]))
print(statistics.median(xs) if xs else "")
PY
}

check_result() {
  local dir="$1" kind="$2"  # cold|lean
  local result="$dir/resume_result.json"
  [[ -f "$result" ]] || anomaly "missing $result"
  python3 - "$result" "$kind" <<'PY' || true
import json, sys
from pathlib import Path
result_path, kind = sys.argv[1:3]
r=json.loads(Path(result_path).read_text())
rc=0
if int(r.get("capacity_retries") or 0) > 0:
  print("capacity_retries", r.get("capacity_retries")); rc=11
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
# loss vs first non-excluded scratch resume with matching loss if any; else skip strict here
# (scratch strict_ref applied after lean3)
if "pip_install_s" not in r or r.get("pip_install_s") in ("", None):
  print("missing pip_install_s"); rc=15
print("CHECK_OK", kind, "R", r.get("R_s"), "pip", r.get("pip_install_s"), "pull", r.get("image_pull_s"), "local", r.get("image_already_local"))
Path(result_path).write_text(json.dumps(r, indent=2)+"\n")
sys.exit(rc)
PY
  local rc=$?
  case $rc in
    0) ;;
    11) anomaly "capacity retry in $dir" ;;
    12) anomaly "lean snapshot pulled image in $dir" ;;
    15) anomaly "pip_install_s missing in $dir (R definition)" ;;
    *) anomaly "check failed rc=$rc for $dir" ;;
  esac
  # R outlier vs current final median (if we have prior finals)
  python3 - "$result" <<'PY' || true
import json, statistics, sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
xs=[]
for p in Path("./logs/mi355x1/8B").glob("resume-*-scratch-*/resume_result.json"):
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
  [[ $rc2 -eq 14 ]] && anomaly "R >50% off final median in $dir"
  [[ $rc2 -eq 0 || $rc2 -eq 0 ]] || true
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
  # Send create chatter to log; only echo the ID on stdout for command substitution.
  bash "$ROOT/scripts/provision.sh" create \
    --name "$name" --size "$SIZE" --region "$REGION" --image "$image" \
    > /tmp/bench_create_nowait.out 2>&1 || true
  cat /tmp/bench_create_nowait.out >> "$LOG"
  local did
  did=$(awk -F= '/^CREATED_ID=/{print $2}' /tmp/bench_create_nowait.out | tail -1)
  [[ -n "$did" ]] || anomaly "create returned no CREATED_ID for $name (see create_nowait.out)"
  echo "$did" >&2
  printf '%s' "$did"
}

wait_active_ssh() {
  local did="$1"
  local st ip
  for i in $(seq 1 80); do
    st=$(doctl compute droplet get "$did" -o json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["status"])')
    echo "poll=$i status=$st id=$did" >> "$LOG"
    [[ "$st" == "active" ]] && break
    sleep 15
  done
  [[ "$st" == "active" ]] || anomaly "droplet $did never active"
  # wait-ssh may print chatter; keep only the IPv4 on stdout for callers.
  ip=$(bash "$ROOT/scripts/provision.sh" wait-ssh --id "$did" --gpu-tool rocm-smi 2>>"$LOG" \
    | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/{ip=$0} END{print ip}')
  [[ -n "$ip" ]] || anomaly "no ssh ip for $did"
  echo "SSH_READY id=$did ip=$ip" >> "$LOG"
  printf '%s' "$ip"
}

run_one() {
  local attempt="$1" variant="$2" image="$3"  # variant cold|snapshot
  local kind="$variant"; [[ "$variant" == "snapshot" ]] && kind="lean"
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
  "variant": "$variant", "resume_attempt": int("$attempt"), "source_run_id": "$RUN_ID",
  "spaces_uri": "$SPACES_URI", "storage": "scratch", "exclude_from_final_results": False,
  "exclude_reason": "", "image": "$image",
  "snapshot_image_id": int("$image") if str("$image").isdigit() else "",
  "note": "$variant resume attempt $attempt; R includes pip_install_s; provision timed live",
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
  manifest_upsert "$DID" "$NAME" "$image" "$CREATE_START"
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
rows.append({"event":"provision_end","wall_utc":datetime.datetime.utcnow().isoformat()+"Z","mono_s":float("$PROV_S"),"duration_s":int("$PROV_S"),"droplet_id":int("$DID"),"ip":"$IP","timed":True,"note":"create_no_wait then wait-ssh+rocm-smi"})
p.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
PY
  stdbuf -oL -eL bash "$ROOT/scripts/measure_resume_existing.sh" \
    --droplet-id "$DID" --ip "$IP" --provision-s "$PROV_S" \
    --events-dir "$EVENTS_DIR" --model "$MODEL" \
    --run-meta "$RUN_META" --spaces-uri "$SPACES_URI" \
    --variant "$variant" | tee -a "$LOG"
  test -f "$EVENTS_DIR/resume_result.json"
  python3 - <<PY
import json, re
from pathlib import Path
d=Path("$EVENTS_DIR")
r=json.loads((d/"resume_result.json").read_text())
r["variant"]="$variant"; r["storage"]="scratch"; r["exclude_from_final_results"]=False
drv=(d/"driver.log").read_text() if (d/"driver.log").exists() else ""
m=re.search(r"image_already_local=(\S+)", drv)
if m: r["image_already_local"]=m.group(1) in ("1","True","true")
for line in (d/"events.jsonl").read_text().splitlines() if (d/"events.jsonl").exists() else []:
  e=json.loads(line)
  if e.get("event")=="env_split" and "image_already_local" in e:
    r["image_already_local"]=e["image_already_local"]
# cross-resume loss vs first final-eligible cold/lean with loss if available
refs=sorted(Path("logs/mi355x1/8B").glob("resume-*-scratch-*/resume_result.json"))
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
  # return via globals
  LAST_DID="$DID"
  LAST_IP="$IP"
  LAST_DIR="$EVENTS_DIR"
}

# ---------- LEAN 2, LEAN 3 ----------
if [[ "${SKIP_LEAN2:-0}" == "1" ]] || ls logs/mi355x1/8B/resume-lean-scratch-2-*/resume_result.json >/dev/null 2>&1; then
  echo "SKIP lean2 (already done or SKIP_LEAN2=1)" | tee -a "$LOG"
else
  run_one 2 snapshot "$LEAN_IMAGE"
  delete_and_account "$LAST_DID"
  python3 scripts/analyze.py --root . | tee -a "$LOG"
fi

if [[ "${SKIP_LEAN3:-0}" == "1" ]] || ls logs/mi355x1/8B/resume-lean-scratch-3-*/resume_result.json >/dev/null 2>&1; then
  echo "SKIP lean3+strict (already done or SKIP_LEAN3=1)" | tee -a "$LOG"
else
  run_one 3 snapshot "$LEAN_IMAGE"
  LEAN3_DID="$LAST_DID"
  LEAN3_IP="$LAST_IP"
  LEAN3_DIR="$LAST_DIR"

  echo "=== scratch strict-loss ref on lean3 $LEAN3_DID $LEAN3_IP ===" | tee -a "$LOG"
  bash "$ROOT/scripts/run_scratch_strict_loss_ref.sh" "$LEAN3_IP" | tee -a "$LOG"
  echo STRICT_ON_LEAN3_DONE | tee -a "$LOG"
  delete_and_account "$LEAN3_DID"
  python3 scripts/analyze.py --root . | tee -a "$LOG"
fi

# ---------- COLD 4, 5, 6 ----------
run_one 4 cold "$COLD_IMAGE"
delete_and_account "$LAST_DID"
python3 scripts/analyze.py --root . | tee -a "$LOG"
run_one 5 cold "$COLD_IMAGE"
delete_and_account "$LAST_DID"
python3 scripts/analyze.py --root . | tee -a "$LOG"
run_one 6 cold "$COLD_IMAGE"
delete_and_account "$LAST_DID"
python3 scripts/analyze.py --root . | tee -a "$LOG"

# ---------- LEAN 4 (replacement for excluded lean1) ----------
run_one 4 snapshot "$LEAN_IMAGE"
delete_and_account "$LAST_DID"
python3 scripts/analyze.py --root . | tee -a "$LOG"

# ---------- Final table ----------
python3 - <<'PY' | tee -a "$LOG" | tee results/final_cold_vs_lean_mi355x1.txt
import json, statistics, csv
from pathlib import Path
from collections import defaultdict

root=Path(".")
phases=["provision_s","env_s","download_s","pip_install_s","load_s","first_step_s","R_s"]
rows=[]
for p in sorted((root/"logs/mi355x1/8B").glob("resume-*-scratch-*/resume_result.json")):
  meta=json.loads((p.parent/"run_meta.json").read_text()) if (p.parent/"run_meta.json").exists() else {}
  r=json.loads(p.read_text())
  if meta.get("exclude_from_final_results") or r.get("exclude_from_final_results"):
    continue
  variant=r.get("variant") or meta.get("variant")
  kind="lean" if variant in ("snapshot","lean") else "cold"
  row={"kind":kind,"run":p.parent.name,"variant":variant}
  for k in phases:
    v=r.get(k,"")
    try: row[k]=float(v)
    except Exception: row[k]=None
  row["image_pull_s"]=r.get("image_pull_s")
  row["image_already_local"]=r.get("image_already_local")
  rows.append(row)

print("FINAL_ROWS", len(rows))
for r in rows:
  print(f"  {r['kind']:4} {r['run']} R={r['R_s']} pip={r['pip_install_s']} prov={r['provision_s']}")

def med(xs):
  xs=[x for x in xs if x is not None]
  return statistics.median(xs) if xs else None

print("\n=== Cold vs Lean medians (final only) ===")
print(f"{'phase':18} {'cold_med':>12} {'lean_med':>12} {'delta':>12} {'lean/cold':>10}")
for k in phases:
  c=med([r[k] for r in rows if r["kind"]=="cold"])
  l=med([r[k] for r in rows if r["kind"]=="lean"])
  delta=(l-c) if (c is not None and l is not None) else None
  ratio=(l/c) if (c not in (None,0) and l is not None) else None
  def fmt(x):
    return f"{x:.3f}" if isinstance(x,float) else "-"
  print(f"{k:18} {fmt(c):>12} {fmt(l):>12} {fmt(delta) if delta is not None else '-':>12} {fmt(ratio) if ratio is not None else '-':>10}")

cold_n=sum(1 for r in rows if r["kind"]=="cold")
lean_n=sum(1 for r in rows if r["kind"]=="lean")
print(f"\nn_cold={cold_n} n_lean={lean_n}")
if cold_n<3 or lean_n<3:
  raise SystemExit(f"INCOMPLETE final set cold={cold_n} lean={lean_n}")
print("MI355X_1X_COMPLETE")
PY

echo ALL_FINAL_DONE | tee -a "$LOG"
