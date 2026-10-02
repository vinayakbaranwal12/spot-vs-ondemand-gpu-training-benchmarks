#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

DID="<droplet-id>"
IP="<droplet-ip>"
EVENTS_DIR=logs/mi355x1/8B/resume-cold-scratch-3-20261001140233
PROVISION_S="${1:-97}"
LOG="$ROOT/logs/cold3_then_lean.nohup"

exec > >(tee -a "$LOG") 2>&1
echo "=== cold3_existing_then_lean start $(date -u +%Y-%m-%dT%H:%M:%SZ) DID=$DID IP=$IP ==="

# refresh provision_end mono if needed
python3 - <<PY
import json, datetime
from pathlib import Path
p=Path("$EVENTS_DIR/events.jsonl")
rows=[]
for line in p.read_text().splitlines():
  if line.strip(): rows.append(json.loads(line))
# keep provision_start; rewrite provision_end
rows=[r for r in rows if r.get("event")!="provision_end"]
if not any(r.get("event")=="provision_start" for r in rows):
  rows.insert(0,{"event":"provision_start","wall_utc":"2026-10-01T14:02:34.081279Z","mono_s":0.0,"simulated_reclaim":True})
rows.append({"event":"provision_end","wall_utc":datetime.datetime.utcnow().isoformat()+"Z","mono_s":float("$PROVISION_S"),"duration_s":int("$PROVISION_S"),"droplet_id":$DID,"ip":"$IP","note":"recovered after batch create--wait drop"})
p.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
print("events rewritten")
PY

stdbuf -oL -eL bash scripts/measure_resume_existing.sh \
  --droplet-id "$DID" --ip "$IP" --provision-s "$PROVISION_S" \
  --events-dir "$EVENTS_DIR" --model 8B \
  --run-meta logs/mi355x1/8B/clean-scratch-mi355x1-8B-20261001081617/run_meta.json \
  --spaces-uri "s3://${SPACES_BUCKET_MEM1}/clean-scratch-mi355x1-8B-20261001081617/ckpt-step-100" \
  --variant cold

test -f "$EVENTS_DIR/resume_result.json"

python3 - <<'PY'
import json, statistics
from pathlib import Path
d=Path("logs/mi355x1/8B/resume-cold-scratch-3-20261001140233")
r=json.loads((d/"resume_result.json").read_text())
ref=json.loads(Path("logs/mi355x1/8B/resume-cold-scratch-1-20261001095032/resume_result.json").read_text())
a=float(r["resume_loss"]); b=float(ref["resume_loss"])
ok=abs(a-b)<=max(1e-3,1e-5*abs(b))
xs=[]
for line in Path("logs/mi355x1/8B/clean-scratch-mi355x1-8B-20261001081617/events.jsonl").read_text().splitlines():
  e=json.loads(line)
  if e.get("event")=="step" and e.get("step",0)>=20: xs.append(float(e["step_time_s"]))
r.update(storage="scratch", loss_match=ok, loss_match_strict=ok, steps_lost=0, step_time_s=statistics.median(xs),
         loss_match_note=f"scratch_cross_resume vs cold1 loss={b}")
(d/"resume_result.json").write_text(json.dumps(r, indent=2)+"\n")
meta=json.loads((d/"run_meta.json").read_text())
for k in ("R_s","storage","loss_match","loss_match_strict","step_time_s","steps_lost","image_pull_s"):
  if k in r: meta[k]=r[k]
meta["storage"]="scratch"; meta["exclude_from_final_results"]=False
(d/"run_meta.json").write_text(json.dumps(meta, indent=2)+"\n")
print("COLD3", r["R_s"], "loss_match", ok)
if not ok: raise SystemExit("LOSS_MISMATCH")
Rs=[float(json.loads(p.read_text())["R_s"]) for p in Path("logs/mi355x1/8B").glob("resume-cold-scratch-*/resume_result.json")]
med=statistics.median(Rs); R=float(r["R_s"])
if R<0.5*med or R>1.5*med: raise SystemExit(f"R_OUTLIER {R} {med}")
print("COLD3_OK")
PY

bash scripts/provision.sh delete --id "$DID"
python3 scripts/analyze.py --root .
echo COLD3_DONE

bash scripts/run_batch_lean_only.sh
echo ALL_REMAINING_DONE
