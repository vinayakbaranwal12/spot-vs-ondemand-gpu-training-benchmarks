#!/usr/bin/env bash
# Create / delete Droplets for the benchmark. Delete by ID only. Never --tag-name delete.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${ROOT}/manifest.json"
CONFIG="${ROOT}/config/run-config.yaml"

TAG="spot-bench"
PROJECT_ID="a25347e6-2cdd-4b40-9d8b-def0047f3720"
SSH_KEY_ID="${SSH_KEY_ID:-58999301}"
SSH_KEY_FALLBACK="${SSH_KEY_FALLBACK:-57674834}"

usage() {
  cat <<EOF
Usage:
  $0 create --name NAME --size SLUG --region REGION --image IMAGE [--wait]
  $0 delete --id DROPLET_ID
  $0 wait-ssh --id DROPLET_ID [--gpu-tool rocm-smi|nvidia-smi]
  $0 get-ip --id DROPLET_ID
EOF
}

require_doctl() {
  command -v doctl >/dev/null || { echo "doctl missing"; exit 1; }
  doctl account get >/dev/null
}

manifest_init() {
  if [[ ! -f "$MANIFEST" ]]; then
    echo '{"droplets":[],"runs":[]}' > "$MANIFEST"
  fi
}

manifest_add_droplet() {
  local id="$1" name="$2" size="$3" region="$4" image="$5" status="$6"
  python3 - "$MANIFEST" "$id" "$name" "$size" "$region" "$image" "$status" <<'PY'
import json,sys,datetime
path, did, name, size, region, image, status = sys.argv[1:]
with open(path) as f:
    m=json.load(f)
m.setdefault("droplets", []).append({
    "id": int(did),
    "name": name,
    "size": size,
    "region": region,
    "image": image,
    "status": status,
    "created_at": datetime.datetime.utcnow().isoformat()+"Z",
    "deleted_at": None,
})
with open(path,"w") as f:
    json.dump(m,f,indent=2)
    f.write("\n")
PY
}

manifest_mark_deleted() {
  local id="$1"
  python3 - "$MANIFEST" "$id" <<'PY'
import json,sys,datetime
path, did = sys.argv[1:]
with open(path) as f:
    m=json.load(f)
for d in m.get("droplets",[]):
    if int(d.get("id",-1))==int(did):
        d["status"]="deleted"
        d["deleted_at"]=datetime.datetime.utcnow().isoformat()+"Z"
with open(path,"w") as f:
    json.dump(m,f,indent=2)
    f.write("\n")
PY
}

cmd_create() {
  local name="" size="" region="" image="" wait_flag=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --name) name="$2"; shift 2 ;;
      --size) size="$2"; shift 2 ;;
      --region) region="$2"; shift 2 ;;
      --image) image="$2"; shift 2 ;;
      --wait) wait_flag=(--wait); shift ;;
      *) echo "unknown arg: $1"; usage; exit 1 ;;
    esac
  done
  [[ -n "$name" && -n "$size" && -n "$region" && -n "$image" ]] || { usage; exit 1; }

  echo "=== CREATE PLAN (awaiting operator yes outside this script) ==="
  echo "name=$name"
  echo "size=$size"
  echo "region=$region"
  echo "image=$image"
  echo "tag=$TAG"
  echo "project_id=$PROJECT_ID"
  echo "ssh_key_id=$SSH_KEY_ID"
  # price if available
  doctl compute size list -o json | python3 -c "
import json,sys
want='$size'
for s in json.load(sys.stdin):
    if s.get('slug')==want:
        print(f\"price_hourly={s.get('price_hourly')}\")
        break
else:
    print('price_hourly=UNKNOWN')
"

  require_doctl
  manifest_init

  set +e
  # Under `set -u`, empty arrays unbound; expand only when --wait was passed.
  out=$(doctl compute droplet create "$name" \
    --size "$size" \
    --image "$image" \
    --region "$region" \
    --ssh-keys "$SSH_KEY_ID" \
    --tag-name "$TAG" \
    --project-id "$PROJECT_ID" \
    --enable-monitoring \
    ${wait_flag[@]+"${wait_flag[@]}"} \
    -o json 2>&1)
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "create with primary SSH key failed; retrying with fallback $SSH_KEY_FALLBACK" >&2
    echo "$out" >&2
    out=$(doctl compute droplet create "$name" \
      --size "$size" \
      --image "$image" \
      --region "$region" \
      --ssh-keys "$SSH_KEY_FALLBACK" \
      --tag-name "$TAG" \
      --project-id "$PROJECT_ID" \
      --enable-monitoring \
      ${wait_flag[@]+"${wait_flag[@]}"} \
      -o json)
  fi

  id=$(python3 -c 'import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; print(d["id"])' <<<"$out")
  echo "CREATED_ID=$id"
  mkdir -p "${ROOT}/snapshots"
  echo "$out" > "${ROOT}/snapshots/last-create.json"
  manifest_add_droplet "$id" "$name" "$size" "$region" "$image" "active"
  echo "$id"
}

cmd_delete() {
  local id=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --id) id="$2"; shift 2 ;;
      *) usage; exit 1 ;;
    esac
  done
  [[ -n "$id" ]] || { usage; exit 1; }
  require_doctl
  # Safety: only delete IDs recorded in manifest
  python3 - "$MANIFEST" "$id" <<'PY'
import json,sys
path, did=sys.argv[1:]
with open(path) as f:
    m=json.load(f)
ids={int(d["id"]) for d in m.get("droplets",[])}
if int(did) not in ids:
    raise SystemExit(f"REFUSING delete: {did} not in manifest.json")
print("manifest_ok")
PY
  doctl compute droplet delete "$id" --force
  manifest_mark_deleted "$id"
  echo "DELETED_ID=$id"
}

cmd_get_ip() {
  local id=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --id) id="$2"; shift 2 ;;
      *) usage; exit 1 ;;
    esac
  done
  doctl compute droplet get "$id" -o json | python3 -c '
import json,sys
d=json.load(sys.stdin)
d=d[0] if isinstance(d,list) else d
for n in d.get("networks",{}).get("v4",[]):
    if n.get("type")=="public":
        print(n["ip_address"]); break
'
}

cmd_wait_ssh() {
  local id="" gpu_tool="rocm-smi"
  local ssh_key="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --id) id="$2"; shift 2 ;;
      --gpu-tool) gpu_tool="$2"; shift 2 ;;
      *) usage; exit 1 ;;
    esac
  done
  local ip
  local SSH=(ssh -i "$ssh_key" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o ConnectTimeout=5 -o BatchMode=yes)
  for _ in $(seq 1 90); do
    ip=$(cmd_get_ip --id "$id" || true)
    if [[ -n "${ip:-}" ]]; then
      if "${SSH[@]}" "root@$ip" "true" 2>/dev/null; then
        if "${SSH[@]}" "root@$ip" "$gpu_tool" >/dev/null 2>&1; then
          echo "SSH_READY ip=$ip gpu_tool=$gpu_tool" >&2
          # stdout is ONLY the IP (callers capture it)
          echo "$ip"
          return 0
        fi
        # SSH up but GPU tool not ready yet
        :
      fi
    fi
    sleep 10
  done
  echo "TIMEOUT waiting for SSH/GPU on droplet $id" >&2
  return 1
}

main() {
  local cmd="${1:-}"; shift || true
  case "$cmd" in
    create) cmd_create "$@" ;;
    delete) cmd_delete "$@" ;;
    wait-ssh) cmd_wait_ssh "$@" ;;
    get-ip) cmd_get_ip "$@" ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
