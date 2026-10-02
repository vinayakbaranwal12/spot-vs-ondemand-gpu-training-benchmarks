#!/usr/bin/env bash
set -euo pipefail
ROOT="."
cd "$ROOT"
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"
LOG="$ROOT/logs/final_scratch_batch.nohup"
echo "=== recover lean2 existing then continue $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

(
  while true; do date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/logs/batch_heartbeat"; sleep 60; done
) &
HB_PID=$!
trap 'kill $HB_PID 2>/dev/null || true' EXIT

DID="<droplet-id>"
EVENTS_DIR=logs/mi355x1/8B/resume-lean-scratch-2-20261001151651
mkdir -p "$EVENTS_DIR"
RUN_META="$ROOT/logs/mi355x1/8B/clean-scratch-mi355x1-8B-20261001081617/run_meta.json"
RUN_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_id"])' "$RUN_META")
SPACES_URI="s3://${SPACES_BUCKET_MEM1}/${RUN_ID}/ckpt-step-100"

# wait active + ssh
for i in $(seq 1 80); do
  ST=$(doctl compute droplet get $DID -o json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["status"])')
  echo "poll=$i status=$ST" | tee -a "$LOG"
  [[ "$ST" == "active" ]] && break
  sleep 15
done
IP=$(bash scripts/provision.sh wait-ssh --id $DID --gpu-tool rocm-smi 2>/dev/null | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/{ip=$0} END{print ip}')
echo "IP=$IP" | tee -a "$LOG"
PROV_S=$(python3 - <<'PY'
import datetime
start=datetime.datetime.fromisoformat('2026-10-01T15:16:51+00:00')
now=datetime.datetime.now(datetime.timezone.utc)
print(int((now-start).total_seconds()))
PY
)
# This provision is continuous from this recovery process after create returned; create itself succeeded at 15:16:51.
# Timed: create start -> SSH+GPU now (same driver session; not a multi-hour reconstruct).
python3 - <<PY
import json, datetime, pathlib
meta=json.loads(open("$RUN_META").read())
meta.update({
  "variant":"snapshot","resume_attempt":2,"source_run_id":"$RUN_ID","spaces_uri":"$SPACES_URI",
  "storage":"scratch","exclude_from_final_results":False,"exclude_reason":"",
  "image": "<snapshot-id>","snapshot_image_id": "<snapshot-id>",
  "note":"lean2; provision timed create_no_wait then wait-ssh+rocm-smi (recovered ID capture bug, same continuous driver)",
  "droplet_id":$DID,
})
pathlib.Path("$EVENTS_DIR/run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
rows=[{"event":"provision_start","wall_utc":"2026-10-01T15:16:51Z","mono_s":0.0,"simulated_reclaim":True},
      {"event":"provision_end","wall_utc":datetime.datetime.utcnow().isoformat()+"Z","mono_s":float("$PROV_S"),"duration_s":int("$PROV_S"),"droplet_id":$DID,"ip":"$IP","timed":True,"note":"create_no_wait succeeded; ID capture fixed mid-flight; wait-ssh+rocm-smi continuous"}]
pathlib.Path("$EVENTS_DIR/events.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
# manifest
m=json.loads(open("manifest.json").read())
if $DID not in {int(d["id"]) for d in m["droplets"]}:
  m["droplets"].append({"id":$DID,"name":"spot-bench-resume-lean-scratch-mi355x1-8B-2-20261001151651","size":"gpu-mi355x1-288gb-spot","region":"mem1","image": "<snapshot-id>","status":"active","created_at":"2026-10-01T15:16:51Z","deleted_at":None,"tag":"spot-bench","idle_s":0})
  open("manifest.json","w").write(json.dumps(m, indent=2)+"\n")
PY

stdbuf -oL -eL bash scripts/measure_resume_existing.sh \
  --droplet-id "$DID" --ip "$IP" --provision-s "$PROV_S" \
  --events-dir "$EVENTS_DIR" --model 8B \
  --run-meta "$RUN_META" --spaces-uri "$SPACES_URI" --variant snapshot | tee -a "$LOG"
test -f "$EVENTS_DIR/resume_result.json"

python3 - <<PY
import json, re
from pathlib import Path
d=Path("$EVENTS_DIR")
r=json.loads((d/"resume_result.json").read_text())
r["variant"]="snapshot"; r["storage"]="scratch"; r["exclude_from_final_results"]=False
for line in (d/"events.jsonl").read_text().splitlines():
  e=json.loads(line)
  if e.get("event")=="env_split" and "image_already_local" in e:
    r["image_already_local"]=e["image_already_local"]
if "pip_install_s" not in r or r.get("pip_install_s") in ("",None):
  raise SystemExit("missing pip_install_s")
(d/"resume_result.json").write_text(json.dumps(r, indent=2)+"\n")
meta=json.loads((d/"run_meta.json").read_text())
for k in ("R_s","pip_install_s","image_pull_s","image_already_local","storage"):
  if k in r: meta[k]=r[k]
meta["exclude_from_final_results"]=False
(d/"run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
print("LEAN2_OK", r["R_s"], "pip", r["pip_install_s"], "pull", r.get("image_pull_s"), "local", r.get("image_already_local"))
if r.get("image_already_local") is False or (lambda v: 999.0 if v in ("",None) else float(v))(r.get("image_pull_s"))>5.0:
  raise SystemExit("LEAN_PULL")
PY

bash scripts/provision.sh delete --id "$DID"
python3 scripts/analyze.py --root .

# Continue with remaining from lean3 onward by calling a truncated final script
# Patch: run final script starting at lean3 - simplest is env SKIP_LEAN2=1
export SKIP_LEAN2=1
bash scripts/run_final_scratch_batch.sh
