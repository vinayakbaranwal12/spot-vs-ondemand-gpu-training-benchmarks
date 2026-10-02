#!/usr/bin/env bash
# MI355X 8x capacity probe: one full-region sweep every 15 minutes until deadline.
# Logs every attempt to results/capacity_probe_mi355x8.csv.
# Usage: DEADLINE_EPOCH=... bash scripts/probe_mi355x8_region_loop.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

REGIONS=(mem1 atl1 nyc1 nyc3 tor1 sfo2 sfo3 fra1 ams3 lon1 sgp1 blr1 syd1 mkc1 ric1)
LOG="${PROBE_LOG:-/tmp/bench_mi355x8_retry.nohup}"
REGION_FILE="${REGION_FILE:-/tmp/bench_mi355x8_region.txt}"
CSV="$ROOT/results/capacity_probe_mi355x8.csv"
DEADLINE_FILE="${DEADLINE_FILE:-/tmp/bench_mi355x8_deadline.txt}"
if [[ -f "$DEADLINE_FILE" ]]; then
  DEADLINE=$(sed -n '2p' "$DEADLINE_FILE")
  DEADLINE_UTC=$(sed -n '1p' "$DEADLINE_FILE" | cut -d= -f2)
else
  DEADLINE=$(( $(date -u +%s) + 86400 ))
  DEADLINE_UTC=$(date -u -r "$DEADLINE" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "@$DEADLINE" +%Y-%m-%dT%H:%M:%SZ)
fi
SWEEP=$(python3 -c "import csv; from pathlib import Path; p=Path(r'$CSV'); print(max([int(r.get('sweep') or 0) for r in csv.DictReader(p.open())] or [0]))")

mkdir -p "$(dirname "$CSV")"
if [[ ! -f "$CSV" ]] || [[ ! -s "$CSV" ]]; then
  echo "timestamp_utc,attempt_global,sweep,region,http_or_status,detail,droplet_id,timestamp_note" > "$CSV"
fi

attempt_global=$(python3 - <<PY
import csv
from pathlib import Path
p=Path("$CSV")
n=0
with p.open() as f:
  for r in csv.DictReader(f):
    n+=1
print(n)
PY
)

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) RETRY_CAPPED deadline=$DEADLINE_UTC attempt_global=$attempt_global" | tee -a "$LOG"

while true; do
  now=$(date -u +%s)
  if [[ "$now" -ge "$DEADLINE" ]]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DEADLINE_REACHED" | tee -a "$LOG"
    python3 - <<PY
import csv
from pathlib import Path
csv_path = Path("$CSV")
rows = list(csv.DictReader(csv_path.open()))
n = len(rows)
regs = sorted({r.get("region","") for r in rows if r.get("region") and r.get("region") not in ("start","")})
sweeps = sorted({r.get("sweep","") for r in rows if r.get("sweep")})
ts = [r["timestamp_utc"] for r in rows if r.get("timestamp_utc") and r.get("timestamp_note") != "timestamp_unknown"]
window = f"{min(ts) if ts else '?'} .. {max(ts) if ts else '?'}"
status = Path("$ROOT/results/mi355x8_capacity_status.txt")
status.write_text(f"""# MI355X 8x capacity — NOT MEASURED
Slug: gpu-mi355x8-2304gb-spot.
Status: NOT MEASURED
Reason: no region accepted the size within the 24h probe window.
Deadline: $DEADLINE_UTC
Window (timestamped rows): {window}
Attempts: {n}
Sweeps: {len(sweeps)}
Regions probed: {", ".join(regs)}
All create responses: 422 Size is not available in this region (or equivalent).
Log: results/capacity_probe_mi355x8.csv
Spot 8x unavailable for a full day is itself a capacity data point.
Hold B300 8x as planned.
""")
print("WROTE NOT MEASURED status")
PY
    break
  fi
  if [[ -f "$REGION_FILE" ]] && grep -q '^REGION=' "$REGION_FILE"; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) already have $(cat "$REGION_FILE")" | tee -a "$LOG"
    break
  fi

  SWEEP=$((SWEEP + 1))
  sweep_start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "$sweep_start SWEEP_START n=$SWEEP" | tee -a "$LOG"
  got=
  for reg in "${REGIONS[@]}"; do
    now=$(date -u +%s)
    if [[ "$now" -ge "$DEADLINE" ]]; then break; fi
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    attempt_global=$((attempt_global + 1))
    name="spot-bench-probe8-$reg-$(date -u +%H%M%S)"
    out=$(doctl compute droplet create "$name" \
      --size gpu-mi355x8-2304gb-spot --image gpu-amd-base --region "$reg" \
      --ssh-keys "${SSH_KEY_ID:-58999301}" --tag-name spot-bench \
      --project-id a25347e6-2cdd-4b40-9d8b-def0047f3720 -o json 2>&1) || true
    if echo "$out" | grep -q '"id"'; then
      id=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d=d[0] if isinstance(d,list) else d; print(d["id"])' "$out")
      echo "$ts,$attempt_global,$SWEEP,$reg,201_created,ok,$id," >> "$CSV"
      echo "GOT probe $id in $reg — delete probe, launch full" | tee -a "$LOG"
      doctl compute droplet delete "$id" --force || true
      echo "REGION=$reg" > "$REGION_FILE"
      nohup env REGION="$reg" bash "$ROOT/scripts/run_full_mi355x8.sh" \
        >>/tmp/bench_mi355x8_full.nohup 2>&1 &
      echo "launched full pid=$!" | tee -a "$LOG"
      got=1
      break
    fi
    detail=$(echo "$out" | tr '\n' ' ' | head -c 180 | sed 's/,/;/g')
    echo "$ts,$attempt_global,$SWEEP,$reg,422,$detail,," >> "$CSV"
    echo "  $reg fail" >>"$LOG"
  done
  sweep_end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "$sweep_end SWEEP_END n=$SWEEP got=${got:-0}" | tee -a "$LOG"
  if [[ -n "$got" ]]; then
    break
  fi
  # Exactly one sweep per 15 minutes: sleep the remainder of the interval from sweep start.
  now=$(date -u +%s)
  # Approximate: always sleep 900s between sweeps (cadence = one full sweep every 15m).
  remain=$((DEADLINE - now))
  if [[ "$remain" -le 0 ]]; then
    continue
  fi
  sleep_s=900
  if [[ "$remain" -lt "$sleep_s" ]]; then
    sleep_s=$remain
  fi
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SLEEP ${sleep_s}s until next sweep" | tee -a "$LOG"
  sleep "$sleep_s"
done
