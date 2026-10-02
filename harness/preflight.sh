#!/usr/bin/env bash
# Preflight: verify each tool from the context where the phase will call it.
# Usage:
#   preflight.sh host
#   preflight.sh droplet --ssh root@IP [--identity KEY]
#   preflight.sh container --ssh root@IP [--identity KEY] --container IMAGE --gpu mi355x|b300
set -euo pipefail

die() { echo "PREFLIGHT_FAIL: $*" >&2; exit 1; }
ok() { echo "PREFLIGHT_OK: $*"; }

MODE="${1:-}"; shift || true
SSH_TARGET=""
IDENTITY=""
CONTAINER=""
GPU="mi355x"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh) SSH_TARGET="$2"; shift 2 ;;
    --identity) IDENTITY="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    *) die "unknown arg $1" ;;
  esac
done

ssh_cmd() {
  local args=(-o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15)
  [[ -n "$IDENTITY" ]] && args+=(-i "$IDENTITY")
  ssh "${args[@]}" "$SSH_TARGET" "$@"
}

case "$MODE" in
  host)
    command -v doctl >/dev/null || die "doctl missing on host"
    doctl version >/dev/null || die "doctl version failed"
    ok "host doctl $(doctl version | head -1)"
    command -v s5cmd >/dev/null || die "s5cmd missing on host"
    s5cmd version >/dev/null || die "s5cmd version failed"
    ok "host s5cmd $(s5cmd version 2>&1 | head -1)"
    command -v python3 >/dev/null || die "python3 missing on host"
    python3 --version >/dev/null || die "python3 version failed"
    ok "host $(python3 --version 2>&1)"
    command -v ssh >/dev/null || die "ssh missing"
    ok "host ssh present"
    ;;
  droplet)
    [[ -n "$SSH_TARGET" ]] || die "droplet mode needs --ssh"
    ssh_cmd 'command -v docker >/dev/null' || die "docker missing on droplet"
    ssh_cmd 'docker version --format "{{.Server.Version}}" >/dev/null' || die "docker version failed on droplet"
    ok "droplet docker $(ssh_cmd 'docker version --format {{.Server.Version}}')"
    ssh_cmd 'command -v s5cmd >/dev/null' || die "s5cmd missing on droplet host"
    ssh_cmd 's5cmd version >/dev/null' || die "s5cmd version failed on droplet host"
    ok "droplet host s5cmd $(ssh_cmd 's5cmd version 2>&1 | head -1')"
    ssh_cmd 'command -v python3 >/dev/null' || die "python3 missing on droplet"
    ok "droplet $(ssh_cmd 'python3 --version 2>&1')"
    # Scratch must be mounted and NOT on the boot disk
    ssh_cmd 'findmnt /scratch >/dev/null' || die "scratch not mounted at /scratch"
    ssh_cmd 'python3 -c "import subprocess; r=subprocess.check_output([\"findmnt\",\"-n\",\"-o\",\"SOURCE\",\"/\"],text=True).strip(); s=subprocess.check_output([\"findmnt\",\"-n\",\"-o\",\"SOURCE\",\"/scratch\"],text=True).strip(); assert r!=s, (r,s)"' \
      || die "SCRATCH is on the boot disk (must be NVMe scratch, not / on vda)"
    ok "droplet scratch=$(ssh_cmd 'findmnt -n -o SOURCE,TARGET,SIZE /scratch')"
    ;;
  container)
    [[ -n "$SSH_TARGET" && -n "$CONTAINER" ]] || die "container mode needs --ssh and --container"
    # s5cmd from the mounted path used by train_bench
    ssh_cmd "docker run --rm -v /usr/local/bin/s5cmd:/host-bin/s5cmd:ro $CONTAINER /host-bin/s5cmd version" \
      >/dev/null || die "s5cmd not executable at /host-bin/s5cmd inside container"
    ok "container /host-bin/s5cmd $(ssh_cmd "docker run --rm -v /usr/local/bin/s5cmd:/host-bin/s5cmd:ro $CONTAINER /host-bin/s5cmd version 2>&1 | head -1")"
    # python + torch inside container with GPU devices
    case "$GPU" in
      mi355x|MI355X)
        ssh_cmd "docker run --rm --network=host --ipc=host --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined $CONTAINER python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count())'" \
          >/dev/null || die "python/torch GPU check failed in ROCm container"
        ;;
      b300|B300)
        ssh_cmd "docker run --rm --gpus all --ipc=host $CONTAINER python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count())'" \
          >/dev/null || die "python/torch GPU check failed in NVIDIA container"
        ;;
      *) die "unknown gpu $GPU" ;;
    esac
    ok "container python/torch GPU ready"
    ;;
  *)
    die "usage: preflight.sh host|droplet|container ..."
    ;;
esac

echo "PREFLIGHT_PASS mode=$MODE"
