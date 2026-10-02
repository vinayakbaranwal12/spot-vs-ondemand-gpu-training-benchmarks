#!/usr/bin/env bash
# Probe which region accepts gpu-mi355x8-2304gb-spot. Prints REGION=... or exits 3.
set -euo pipefail
ROOT="."
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

SIZE="gpu-mi355x8-2304gb-spot"
IMAGE="gpu-amd-base"
SSH_KEY_ID="${SSH_KEY_ID:-58999301}"
PROJECT_ID="a25347e6-2cdd-4b40-9d8b-def0047f3720"
# Prefer mem1 first, then other common GPU regions
REGIONS=(mem1 atl1 nyc1 nyc3 tor1 sfo2 sfo3 fra1 ams3 lon1 sgp1 blr1 syd1)

NAME="spot-bench-probe-mi355x8-$(date -u +%H%M%S)"
for reg in "${REGIONS[@]}"; do
  echo "PROBE region=$reg" >&2
  out=$(doctl compute droplet create "$NAME-$reg" \
    --size "$SIZE" --image "$IMAGE" --region "$reg" \
    --ssh-keys "$SSH_KEY_ID" --tag-name spot-bench \
    --project-id "$PROJECT_ID" -o json 2>&1) || true
  if echo "$out" | grep -q '"id"'; then
    id=$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); d=d[0] if isinstance(d,list) else d; print(d["id"])' "$out")
    echo "CREATED_PROBE id=$id region=$reg" >&2
    # Immediate delete — probe only
    doctl compute droplet delete "$id" --force >/dev/null 2>&1 || true
    echo "REGION=$reg"
    exit 0
  fi
  echo "$out" | head -c 300 >&2
  echo >&2
done
echo "NO_REGION_ACCEPTS_MI355X8" >&2
exit 3
