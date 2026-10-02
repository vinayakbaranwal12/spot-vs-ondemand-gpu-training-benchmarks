#!/usr/bin/env bash
# Prepare a GPU Droplet: find scratch NVMe, pull container, write environment snapshot.
# Usage: setup_node.sh --gpu mi355x|b300 --snapshot-dir DIR [--smoke]
set -euo pipefail

GPU=""
SNAP_DIR=""
SMOKE=0
CONTAINER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --snapshot-dir) SNAP_DIR="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "unknown: $1"; exit 1 ;;
  esac
done

[[ -n "$GPU" && -n "$SNAP_DIR" ]] || { echo "need --gpu and --snapshot-dir"; exit 1; }
mkdir -p "$SNAP_DIR"

# Default containers (verified tags exist on Hub / NGC auth realm)
if [[ -z "$CONTAINER" ]]; then
  case "$GPU" in
    mi355x|MI355X)
      CONTAINER="rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0"
      ;;
    b300|B300)
      # Newest NGC PyTorch whose CUDA runs natively on DO gpu-h100x1-base driver 580
      # (CUDA 13.0.2). Overrides "current container" (26.07 / CUDA 13.3.1) for B300 only:
      # 26.07 falls back to Forward Compatibility on driver 580, and no public DO base
      # image ships driver >=610. Blackwell optimized since NGC 25.01 (includes sm_103).
      CONTAINER="nvcr.io/nvidia/pytorch:25.11-py3"
      ;;
    *) echo "unknown gpu $GPU"; exit 1 ;;
  esac
fi

echo "CONTAINER=$CONTAINER" | tee "$SNAP_DIR/container.txt"

# Discover and mount scratch NVMe. Prefer labeled DOSCRATCH / largest non-root disk.
# Never use /mnt when it sits on the boot disk (common DO default bind that fooled us).
mount_scratch() {
  mkdir -p /scratch
  if findmnt /scratch >/dev/null 2>&1; then
    echo /scratch
    return
  fi
  local dev=""
  # Prefer LABEL=DOSCRATCH (DigitalOcean GPU scratch)
  if blkid -L DOSCRATCH >/dev/null 2>&1; then
    dev=$(blkid -L DOSCRATCH)
  elif [[ -b /dev/vdc1 ]]; then
    dev=/dev/vdc1
  elif [[ -b /dev/nvme1n1p1 ]]; then
    dev=/dev/nvme1n1p1
  else
    # Largest unmounted disk partition > 100G
    dev=$(lsblk -nb -o NAME,SIZE,TYPE,MOUNTPOINT | awk '
      $3=="part" && $4=="" {name="/dev/"$1; size=$2; if(size>100*1024*1024*1024 && size>best){best=size; d=name}}
      END{print d}')
  fi
  [[ -n "$dev" && -b "$dev" ]] || { echo "NO_SCRATCH_DEVICE" >&2; return 1; }
  local fstype
  fstype=$(blkid -s TYPE -o value "$dev" || true)
  if [[ -z "$fstype" ]]; then
    echo "formatting $dev as ext4 (scratch, data loss OK)" >&2
    mkfs.ext4 -F -L DOSCRATCH "$dev"
    fstype=ext4
  fi
  local uuid
  uuid=$(blkid -s UUID -o value "$dev")
  mount UUID="$uuid" /scratch
  if ! grep -q "$uuid" /etc/fstab 2>/dev/null; then
    echo "UUID=$uuid /scratch ext4 defaults,nofail 0 2" >> /etc/fstab
  fi
  echo /scratch
}

find_scratch() {
  local scratch
  scratch=$(mount_scratch) || true
  if [[ -z "$scratch" || "$scratch" == "NO_SCRATCH_DEVICE" ]]; then
    echo "FATAL: could not mount scratch NVMe" >&2
    return 1
  fi
  # Reject boot-disk paths
  local root_src scratch_src
  root_src=$(findmnt -n -o SOURCE /)
  scratch_src=$(findmnt -n -o SOURCE "$scratch")
  if [[ "$root_src" == "$scratch_src" ]]; then
    echo "FATAL: scratch $scratch is on boot disk ($root_src)" >&2
    return 1
  fi
  echo "$scratch"
}

SCRATCH="$(find_scratch)"
mkdir -p "$SCRATCH/bench"
echo "SCRATCH=$SCRATCH" | tee "$SNAP_DIR/scratch_path.txt"

{
  echo "=== df -h ==="
  df -h
  echo "=== lsblk ==="
  lsblk || true
  echo "=== scratch ==="
  echo "$SCRATCH"
  echo "=== write test ==="
  dd if=/dev/zero of="$SCRATCH/bench/write_test.bin" bs=1M count=256 oflag=direct 2>&1 || \
    dd if=/dev/zero of="$SCRATCH/bench/write_test.bin" bs=1M count=256 2>&1
  rm -f "$SCRATCH/bench/write_test.bin"
} > "$SNAP_DIR/disk_and_network.txt" 2>&1

# GPU info
case "$GPU" in
  mi355x|MI355X)
    (rocm-smi || true; rocminfo 2>/dev/null | head -80 || true) > "$SNAP_DIR/gpu_info.txt" 2>&1
    ;;
  b300|B300)
    (nvidia-smi || true; nvidia-smi -q 2>/dev/null | head -120 || true) > "$SNAP_DIR/gpu_info.txt" 2>&1
    ;;
esac

# Pull container only if missing. Lean snapshots must already contain the image
# (containerd store under /var/lib/containerd); a real pull means the snapshot is invalid.
IMAGE_ALREADY_LOCAL=0
if docker image inspect "$CONTAINER" >/dev/null 2>&1; then
  IMAGE_ALREADY_LOCAL=1
  echo "image already local; skipping docker pull: $CONTAINER"
  IMAGE_PULL_S=0
else
  echo "Pulling $CONTAINER ..."
  PULL_START=$(date +%s.%N)
  docker pull "$CONTAINER"
  PULL_END=$(date +%s.%N)
  IMAGE_PULL_S=$(python3 -c "print(float('$PULL_END')-float('$PULL_START'))")
fi

DIGEST=$(docker image inspect "$CONTAINER" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo "DIGEST_UNKNOWN")
IMAGE_SIZE_BYTES=$(docker image inspect "$CONTAINER" --format '{{.Size}}' 2>/dev/null || echo "0")
IMAGE_SIZE_HUMAN=$(docker image inspect "$CONTAINER" --format '{{.Size}}' 2>/dev/null | awk '{printf "%.2f GiB", $1/1024/1024/1024}')

echo "image_already_local=$IMAGE_ALREADY_LOCAL" | tee -a "$SNAP_DIR/versions.txt"
echo "image_pull_s=$IMAGE_PULL_S" | tee -a "$SNAP_DIR/versions.txt"
echo "image_digest=$DIGEST" | tee -a "$SNAP_DIR/versions.txt"
echo "image_ref=$CONTAINER" | tee -a "$SNAP_DIR/versions.txt"
echo "image_size_bytes=$IMAGE_SIZE_BYTES ($IMAGE_SIZE_HUMAN)" | tee -a "$SNAP_DIR/versions.txt"

# Time a minimal container start (GPU visible hello)
START_T0=$(date +%s.%N)
case "$GPU" in
  mi355x|MI355X)
    docker run --rm --network=host --ipc=host \
      --device=/dev/kfd --device=/dev/dri --group-add video \
      --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
      "$CONTAINER" \
      bash -lc 'python3 -c "import torch; print(torch.cuda.device_count(), torch.__version__)"' \
      > "$SNAP_DIR/container_start_probe.txt" 2>&1
    ;;
  b300|B300)
    docker run --rm --gpus all --ipc=host \
      "$CONTAINER" \
      bash -lc 'python3 -c "import torch; print(torch.cuda.device_count(), torch.__version__)"' \
      > "$SNAP_DIR/container_start_probe.txt" 2>&1
    ;;
esac
START_T1=$(date +%s.%N)
CONTAINER_START_S=$(python3 -c "print(float('$START_T1')-float('$START_T0'))")
echo "container_start_s=$CONTAINER_START_S" | tee -a "$SNAP_DIR/versions.txt"

python3 - <<PY
import json
open("$SNAP_DIR/env_timing.json","w").write(json.dumps({
  "image_ref": "$CONTAINER",
  "image_digest": "$DIGEST",
  "image_already_local": bool(int("$IMAGE_ALREADY_LOCAL")),
  "image_pull_s": float("$IMAGE_PULL_S"),
  "container_start_s": float("$CONTAINER_START_S"),
  "image_size_bytes": int("$IMAGE_SIZE_BYTES" or 0),
  "env_s": float("$IMAGE_PULL_S")+float("$CONTAINER_START_S"),
}, indent=2)+"\n")
print("wrote env_timing.json")
PY

# Fuller version snapshot inside container (not counted in container_start_s above)
docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  "$CONTAINER" \
  bash -lc 'set -e; echo OS=$(cat /etc/os-release | head -5); python3 - <<EOF
import torch, sys
print("python", sys.version)
print("torch", torch.__version__)
try:
  import transformers
  print("transformers", transformers.__version__)
except Exception as e:
  print("transformers", "MISSING", e)
EOF
' > "$SNAP_DIR/container_versions_raw.txt" 2>&1 || \
docker run --rm --gpus all --ipc=host \
  "$CONTAINER" \
  bash -lc 'python3 - <<EOF
import torch, sys
print("python", sys.version)
print("torch", torch.__version__)
try:
  import transformers
  print("transformers", transformers.__version__)
except Exception as e:
  print("transformers", "MISSING", e)
EOF
' > "$SNAP_DIR/container_versions_raw.txt" 2>&1

{
  echo "=== host ==="
  uname -a
  echo "=== docker ==="
  docker version --format '{{.Server.Version}}' 2>/dev/null || true
  echo "=== container versions ==="
  cat "$SNAP_DIR/container_versions_raw.txt"
} >> "$SNAP_DIR/versions.txt"

# Ensure transformers supports Qwen3; upgrade in a derived note (actual upgrade at train time if needed)
echo "setup_node done scratch=$SCRATCH container=$CONTAINER"
echo "$SCRATCH" > "$SNAP_DIR/scratch.path"
echo "$CONTAINER" > "$SNAP_DIR/container.ref"
