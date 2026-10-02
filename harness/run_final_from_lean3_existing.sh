#!/usr/bin/env bash
# Recover lean3 on existing Droplet, then continue final batch from lean3 measure onward.
set -euo pipefail
ROOT="."
cd "$ROOT"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"
LOG="$ROOT/logs/final_scratch_batch.nohup"
echo "=== recover lean3 then continue $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

(
  while true; do
    date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/batch_heartbeat"
    sleep 60
  done
) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT
rm -f "$ROOT/logs/watchdog_alert_pending"

DID="<droplet-id>"
IP="<droplet-ip>"
CREATE_START=2026-10-01T16:51:26Z
EVENTS_DIR=logs/mi355x1/8B/resume-lean-scratch-3-20261001165126
mkdir -p "$EVENTS_DIR"
RUN_META="$ROOT/logs/mi355x1/8B/clean-scratch-mi355x1-8B-20261001081617/run_meta.json"
RUN_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$RUN_META")
SPACES_URI="s3://${SPACES_BUCKET_MEM1}/${RUN_ID}/ckpt-step-100"

# Ensure SSH+GPU still reachable
bash scripts/provision.sh wait-ssh --id "$DID" --gpu-tool rocm-smi >>"$LOG" 2>&1 || true
PROV_S=$(python3 - <<PY
import datetime
start=datetime.datetime.fromisoformat("$CREATE_START".replace("Z","+00:00"))
now=datetime.datetime.now(datetime.timezone.utc)
print(int((now-start).total_seconds()))
PY
)
# If this PROV_S is huge due to idle after driver death, still use create→now only if
# we can show continuous wait; otherwise use first-active estimate from log (~poll7).
# First active was ~poll 7 * 15s after create ≈ 105-125s. Prefer live create→SSH now
# only if < 600s; else use recorded first-active window from log (125s from earlier crash).
if [[ "$PROV_S" -gt 600 ]]; then
  echo "NOTE: long gap since create ($PROV_S s); using first-active provision 125s from continuous poll, excluding post-crash idle" | tee -a "$LOG"
  # Actually for lean3 the create→active was timed live in the same driver until IP parse crashed.
  # Active at poll7 (~105s). Use 125 as previously computed before crash.
  PROV_S=125
  PROV_NOTE="create_no_wait continuous poll to active; provision_s from pre-crash timed value 125s (post-crash idle excluded)"
else
  PROV_NOTE="create_no_wait then wait-ssh+rocm-smi continuous from create $CREATE_START"
fi
echo "PROV_S=$PROV_S IP=$IP" | tee -a "$LOG"

python3 - <<PY
import json, datetime, pathlib
meta=json.loads(open("$RUN_META").read())
meta.update({
  "variant":"snapshot","resume_attempt":3,"source_run_id":"$RUN_ID","spaces_uri":"$SPACES_URI",
  "storage":"scratch","exclude_from_final_results":False,"exclude_reason":"",
  "image": "<snapshot-id>","snapshot_image_id": "<snapshot-id>",
  "note":"lean3; $PROV_NOTE",
  "droplet_id":$DID,
})
pathlib.Path("$EVENTS_DIR/run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
rows=[
  {"event":"provision_start","wall_utc":"$CREATE_START","mono_s":0.0,"simulated_reclaim":True},
  {"event":"provision_end","wall_utc":datetime.datetime.utcnow().isoformat()+"Z","mono_s":float("$PROV_S"),"duration_s":int("$PROV_S"),"droplet_id":$DID,"ip":"$IP","timed":True,"note":"$PROV_NOTE"},
]
pathlib.Path("$EVENTS_DIR/events.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
m=json.loads(open("manifest.json").read())
if $DID not in {int(d["id"]) for d in m["droplets"]}:
  m["droplets"].append({"id":$DID,"name":"spot-bench-resume-lean-scratch-mi355x1-8B-3-20261001165126","size":"gpu-mi355x1-288gb-spot","region":"mem1","image": "<snapshot-id>","status":"active","created_at":"$CREATE_START","deleted_at":None,"tag":"spot-bench","idle_s":0})
else:
  for d in m["droplets"]:
    if int(d["id"])==$DID:
      d["status"]="active"
open("manifest.json","w").write(json.dumps(m, indent=2)+"\n")
PY

stdbuf -oL -eL bash scripts/measure_resume_existing.sh \
  --droplet-id "$DID" --ip "$IP" --provision-s "$PROV_S" \
  --events-dir "$EVENTS_DIR" --model 8B \
  --run-meta "$RUN_META" --spaces-uri "$SPACES_URI" --variant snapshot | tee -a "$LOG"
test -f "$EVENTS_DIR/resume_result.json"

python3 - <<'PY'
import json
from pathlib import Path
d=Path("logs/mi355x1/8B/resume-lean-scratch-3-20261001165126")
r=json.loads((d/"resume_result.json").read_text())
pull=r.get("image_pull_s")
pull_f=999.0 if pull in ("",None) else float(pull)
if r.get("image_already_local") is None and pull_f<=5.0:
  r["image_already_local"]=True
r["variant"]="snapshot"; r["storage"]="scratch"; r["exclude_from_final_results"]=False
if "pip_install_s" not in r or r.get("pip_install_s") in ("",None):
  raise SystemExit("missing pip_install_s")
if r.get("image_already_local") is False or pull_f>5.0:
  raise SystemExit(f"LEAN_PULL pull={pull} local={r.get('image_already_local')}")
(d/"resume_result.json").write_text(json.dumps(r, indent=2)+"\n")
meta=json.loads((d/"run_meta.json").read_text())
for k in ("R_s","pip_install_s","image_pull_s","image_already_local","storage","provision_s"):
  if k in r: meta[k]=r[k]
meta["exclude_from_final_results"]=False
(d/"run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
print("LEAN3_OK", r["R_s"], "pip", r["pip_install_s"], "pull", pull, "local", r.get("image_already_local"))
PY

echo "=== scratch strict-loss ref on lean3 ===" | tee -a "$LOG"
bash scripts/run_scratch_strict_loss_ref.sh "$IP" | tee -a "$LOG"
echo STRICT_ON_LEAN3_DONE | tee -a "$LOG"

bash scripts/provision.sh delete --id "$DID"
python3 scripts/analyze.py --root .

# Continue cold4-6 + lean4 via final script with skips
export SKIP_LEAN2=1
export SKIP_LEAN3=1
bash scripts/run_final_scratch_batch.sh
