#!/usr/bin/env bash
# Watchdog for spot-bench Droplets.
# - Alerts if heartbeat is stale while Droplets are active.
# - Auto-deletes by ID when: no active train process for >=30 minutes AND
#   related checkpoints are confirmed in Spaces (or resume source ckpt exists).
# Never deletes while torchrun/train_bench is running.
set -euo pipefail
ROOT="."
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8)
HB="$ROOT/logs/batch_heartbeat"
ALERT="$ROOT/logs/watchdog.log"
OK="$ROOT/logs/watchdog_ok.log"
IDLE_LOG="$ROOT/logs/watchdog_autodelete.log"
STATE_DIR="$ROOT/logs/watchdog_state"
mkdir -p "$ROOT/logs" "$STATE_DIR"

IDLE_LIMIT_S="${WATCHDOG_IDLE_LIMIT_S:-1800}"  # 30 minutes
HB_ALERT_S="${WATCHDOG_HB_ALERT_S:-600}"

spaces_has_prefix() {
  local endpoint="$1" uri="$2"
  # uri like s3://bucket/prefix — list one object
  AWS_ACCESS_KEY_ID="${SPACES_KEY}" AWS_SECRET_ACCESS_KEY="${SPACES_SECRET}" \
    s5cmd --endpoint-url "$endpoint" ls "${uri%/}/" 2>/dev/null | head -n 1 | grep -q .
}

confirm_spaces_for_droplet() {
  local ip="$1"
  # Prefer Spaces URI from remote events / env
  local uri endpoint
  uri=$("${SSH[@]}" "root@$ip" 'python3 - <<'"'"'PY'"'"'
import glob, json, os, re
cands=[]
for p in glob.glob("/opt/bench/**/*.jsonl", recursive=True)+glob.glob("/opt/bench/**/run_meta.json", recursive=True):
  try:
    if p.endswith(".json"):
      m=json.load(open(p))
      for k in ("spaces_uri","spaces_prefix","spaces_uri_ckpt"):
        if m.get(k): cands.append(str(m[k]))
    else:
      for line in open(p):
        if "s3://" in line:
          m=re.search(r"s3://[A-Za-z0-9._\-/]+", line)
          if m: cands.append(m.group(0))
  except Exception:
    pass
# also common clean prefixes from env
print(cands[0] if cands else "")
PY' 2>/dev/null || true)
  if [[ -z "$uri" ]]; then
    return 1
  fi
  # strip to directory prefix
  uri="${uri%/}"
  if [[ "$uri" == *"/ckpt-step-"* ]]; then
    uri="${uri%/*}"
  fi
  if [[ "$uri" == *mkc1* ]] || [[ "$uri" == *"<bucket>"* ]]; then
    endpoint="https://${SPACES_REGION_MKC1}.digitaloceanspaces.com"
  else
    endpoint="https://${SPACES_REGION_MEM1}.digitaloceanspaces.com"
  fi
  if command -v s5cmd >/dev/null 2>&1 && spaces_has_prefix "$endpoint" "$uri"; then
    echo "$uri"
    return 0
  fi
  # Fallback: docker-less remote s5cmd
  if "${SSH[@]}" "root@$ip" "test -x /usr/local/bin/s5cmd || command -v s5cmd >/dev/null"; then
    if "${SSH[@]}" "root@$ip" "set -a; [[ -f /opt/bench/.spaces.env ]] && source /opt/bench/.spaces.env; set +a; s5cmd --endpoint-url '$endpoint' ls '${uri%/}/' 2>/dev/null | head -n 1 | grep -q ."; then
      echo "$uri"
      return 0
    fi
  fi
  return 1
}

droplet_train_running() {
  local ip="$1"
  "${SSH[@]}" "root@$ip" 'pgrep -f "torchrun|train_bench.py" >/dev/null' 2>/dev/null
}

droplet_idle_seconds() {
  local id="$1" ip="$2"
  local state="$STATE_DIR/${id}.json"
  local now_s running
  now_s=$(date +%s)
  if droplet_train_running "$ip"; then
    python3 - <<PY
import json
from pathlib import Path
Path("$state").write_text(json.dumps({"last_active_s": $now_s, "ip": "$ip", "running": True})+"\n")
print(0)
PY
    return
  fi
  python3 - <<PY
import json, time
from pathlib import Path
p=Path("$state")
now=int("$now_s")
if p.exists():
  d=json.loads(p.read_text())
  last=int(d.get("last_active_s") or now)
else:
  # first observation of idle: start the idle clock now (do not use create_at)
  last=now
  d={"last_active_s": now, "ip": "$ip", "running": False}
p.write_text(json.dumps({**d, "last_active_s": last, "running": False, "ip": "$ip"})+"\n")
print(now-last)
PY
}

account_delete() {
  local did="$1" reason="$2"
  bash "$ROOT/scripts/provision.sh" delete --id "$did" >>"$IDLE_LOG" 2>&1 || \
    doctl compute droplet delete "$did" --force >>"$IDLE_LOG" 2>&1 || true
  python3 - <<PY
import json, datetime
from pathlib import Path
path=Path("$ROOT/manifest.json")
m=json.loads(path.read_text()) if path.exists() else {"droplets":[]}
did=int("$did")
for d in m.get("droplets",[]):
  if int(d.get("id",-1))==did:
    d["status"]="deleted"
    d["deleted_at"]=datetime.datetime.utcnow().isoformat()+"Z"
    d["autodelete_reason"]="""$reason"""
    break
else:
  m.setdefault("droplets",[]).append({
    "id": did, "status": "deleted",
    "deleted_at": datetime.datetime.utcnow().isoformat()+"Z",
    "autodelete_reason": """$reason""",
  })
path.write_text(json.dumps(m, indent=2)+"\n")
PY
  rm -f "$STATE_DIR/${did}.json"
}

while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  age=999999
  if [[ -f "$HB" ]]; then
    age=$(( $(date +%s) - $(stat -f %m "$HB" 2>/dev/null || stat -c %Y "$HB") ))
  fi

  DROPLETS=()
  while IFS= read -r _row; do
    [[ -n "${_row:-}" ]] && DROPLETS+=("$_row")
  done < <(doctl compute droplet list --tag-name spot-bench -o json 2>/dev/null \
    | python3 -c '
import json,sys
ds=json.load(sys.stdin)
for d in ds:
  if d.get("status")!="active":
    continue
  ip=next((n["ip_address"] for n in d.get("networks",{}).get("v4",[]) if n.get("type")=="public"),"")
  print("%s\t%s\t%s" % (d["id"], ip, d.get("name","")))
' || true)

  active_ids=""
  for row in "${DROPLETS[@]:-}"; do
    [[ -z "${row:-}" ]] && continue
    did=$(echo "$row" | awk -F'\t' '{print $1}')
    ip=$(echo "$row" | awk -F'\t' '{print $2}')
    name=$(echo "$row" | awk -F'\t' '{print $3}')
    active_ids="${active_ids:+$active_ids,}$did"
    [[ -n "$ip" ]] || continue

    # Skip if local active_run.json claims this droplet
    if [[ -f "$ROOT/logs/active_run.json" ]] && grep -q "$did" "$ROOT/logs/active_run.json" 2>/dev/null; then
      # still verify process; if process dead but claim present, idle clock still runs
      :
    fi

    idle_s=$(droplet_idle_seconds "$did" "$ip" || echo 0)
    if droplet_train_running "$ip"; then
      echo "$now ok droplet=$did running idle_s=0 name=$name" >>"$OK"
      continue
    fi

    if [[ "$idle_s" -ge "$IDLE_LIMIT_S" ]]; then
      spaces_uri=""
      if spaces_uri=$(confirm_spaces_for_droplet "$ip"); then
        reason="idle_no_train_${idle_s}s_spaces_ok:${spaces_uri}"
        msg="$now WATCHDOG_AUTODELETE id=$did name=$name idle_s=$idle_s reason=$reason"
        echo "$msg" | tee -a "$ALERT" | tee -a "$IDLE_LOG"
        account_delete "$did" "$reason"
      else
        msg="$now WATCHDOG_IDLE_NO_SPACES id=$did name=$name idle_s=$idle_s (not deleting; Spaces not confirmed)"
        echo "$msg" | tee -a "$ALERT"
      fi
    else
      echo "$now ok droplet=$did idle_s=$idle_s name=$name" >>"$OK"
    fi
  done

  if [[ -n "${active_ids:-}" && "$age" -gt "$HB_ALERT_S" ]]; then
    msg="$now WATCHDOG_ALERT active_droplets=$active_ids heartbeat_age_s=$age (>${HB_ALERT_S}s). Driver likely dead; Mac sleep or process exit."
    echo "$msg" | tee -a "$ALERT"
    echo "$msg" > "$ROOT/logs/watchdog_alert_pending"
  else
    echo "$now ok active=${active_ids:-none} heartbeat_age_s=$age" >>"$OK"
  fi
  sleep 60
done
