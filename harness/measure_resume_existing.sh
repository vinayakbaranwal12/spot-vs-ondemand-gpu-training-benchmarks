#!/usr/bin/env bash
# Timed snapshot-variant resume on an already-created Droplet (operator yes already given).
# Does NOT create or delete the Droplet.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$HOME/.bench.env"
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
export PATH="/opt/homebrew/bin:$PATH"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes)
SCP=(scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes)

DROPLET_ID="" IP="" PROVISION_S="" EVENTS_DIR="" MODEL="8B" RUN_META="" SPACES_URI="" VARIANT="snapshot" GPU="mi355x" ENDPOINT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --droplet-id) DROPLET_ID="$2"; shift 2 ;;
    --ip) IP="$2"; shift 2 ;;
    --provision-s) PROVISION_S="$2"; shift 2 ;;
    --events-dir) EVENTS_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --run-meta) RUN_META="$2"; shift 2 ;;
    --spaces-uri) SPACES_URI="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --spaces-endpoint) ENDPOINT="$2"; shift 2 ;;
    *) echo "unknown $1"; exit 1 ;;
  esac
done
[[ -n "$DROPLET_ID" && -n "$IP" && -n "$EVENTS_DIR" && -n "$RUN_META" && -n "$SPACES_URI" ]] || { echo missing args; exit 1; }
case "$GPU" in
  mi355x|MI355X)
    GPU="mi355x"
    [[ -n "$ENDPOINT" ]] || ENDPOINT="https://${SPACES_REGION_MEM1}.digitaloceanspaces.com"
    ;;
  b300|B300)
    GPU="b300"
    [[ -n "$ENDPOINT" ]] || ENDPOINT="https://${SPACES_REGION_MKC1}.digitaloceanspaces.com"
    ;;
  *) echo "unknown --gpu $GPU"; exit 1 ;;
esac
BATCH_SIZE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["batch_size"])' "$RUN_META")
GRAD_ACCUM=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("grad_accum",1))' "$RUN_META")
mkdir -p "$EVENTS_DIR"
EVENTS="$EVENTS_DIR/events.jsonl"
RESULT_JSON="$EVENTS_DIR/resume_result.json"
# preserve existing provision events if present
[[ -s "$EVENTS" ]] || : > "$EVENTS"

log_event() {
  local event="$1"; shift
  python3 - "$EVENTS" "$event" "$@" <<'PY'
import json,sys,time,datetime
path=sys.argv[1]; event=sys.argv[2]
fields={}
args=sys.argv[3:]
for i in range(0,len(args),2):
    k=args[i]; v=args[i+1]
    try:
        if "." in v: v=float(v)
        else: v=int(v)
    except Exception:
        pass
    fields[k]=v
rec={"event":event,"wall_utc":datetime.datetime.utcnow().isoformat()+"Z","mono_s":time.perf_counter(),**fields}
open(path,"a").write(json.dumps(rec)+"\n")
print(json.dumps(rec))
PY
}

# Sync scripts flat
"${SSH[@]}" "root@$IP" 'mkdir -p /tmp/spot-bench-scripts-upload /opt/bench-scripts'
"${SCP[@]}" "$ROOT/scripts/"*.py "$ROOT/scripts/"*.sh "root@$IP:/tmp/spot-bench-scripts-upload/"
"${SSH[@]}" "root@$IP" 'cp -f /tmp/spot-bench-scripts-upload/* /opt/bench-scripts/ && rm -rf /opt/bench-scripts/scripts /tmp/spot-bench-scripts-upload && grep -q grad-accum /opt/bench-scripts/train_bench.py'

# env (image should already be in snapshot for lean; cold pulls here)
t0=$(date +%s)
log_event env_start variant "$VARIANT"
"${SSH[@]}" "root@$IP" bash -s -- "$GPU" <<'REMOTE'
set -euo pipefail
GPU="$1"
mkdir -p /opt/bench /var/log/spot-bench
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
if ! command -v s5cmd >/dev/null; then
  curl -sL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin s5cmd
fi
bash /opt/bench-scripts/setup_node.sh --gpu "$GPU" --snapshot-dir /opt/bench/snapshot
REMOTE
t1=$(date +%s)
env_s=$((t1-t0))
log_event env_end duration_s "$env_s"
image_pull_s=""; container_start_s=""
if "${SSH[@]}" "root@$IP" 'test -f /opt/bench/snapshot/env_timing.json'; then
  image_pull_s=$("${SSH[@]}" "root@$IP" 'python3 -c "import json; print(json.load(open(\"/opt/bench/snapshot/env_timing.json\")).get(\"image_pull_s\",\"\"))"')
  container_start_s=$("${SSH[@]}" "root@$IP" 'python3 -c "import json; print(json.load(open(\"/opt/bench/snapshot/env_timing.json\")).get(\"container_start_s\",\"\"))"')
  # Detect lean pull: empty/0 pull with image already local flagged in setup output when present
  already_local=$("${SSH[@]}" "root@$IP" 'python3 -c "import json; print(json.load(open(\"/opt/bench/snapshot/env_timing.json\")).get(\"image_already_local\",\"\"))"')
  log_event env_split image_pull_s "${image_pull_s:-0}" container_start_s "${container_start_s:-0}" image_already_local "${already_local}"
fi
SCRATCH=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/scratch.path)
"${SSH[@]}" "root@$IP" 'python3 -c "import subprocess; r=subprocess.check_output([\"findmnt\",\"-n\",\"-o\",\"SOURCE\",\"/\"],text=True).strip(); s=subprocess.check_output([\"findmnt\",\"-n\",\"-o\",\"SOURCE\",\"/scratch\"],text=True).strip(); assert r!=s and s, (r,s)"' \
  || { echo "FATAL: /scratch missing or on boot disk"; exit 2; }
[[ "$SCRATCH" == "/scratch" ]] || { echo "FATAL: scratch.path=$SCRATCH"; exit 2; }
CONTAINER=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/container.ref)

# shellcheck source=/dev/null
source "$ROOT/scripts/spaces_env_remote.sh"
spaces_env_push "$IP"

CKPT_LOCAL="$SCRATCH/bench/resume-ckpt"
t0=$(date +%s)
log_event download_start spaces_uri "$SPACES_URI"
"${SSH[@]}" "root@$IP" bash -s <<REMOTE
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a
rm -rf '$CKPT_LOCAL'; mkdir -p '$CKPT_LOCAL'
s5cmd --endpoint-url '$ENDPOINT' cp '${SPACES_URI%/}/*' '$CKPT_LOCAL/'
REMOTE
t1=$(date +%s)
download_s=$((t1-t0))
log_event download_end duration_s "$download_s"

REMOTE_EVENTS=/opt/bench/resume_events.jsonl
PIP_FILE=/opt/bench/pip_install_s.txt
t0=$(date +%s)
log_event resume_train_start
if [[ "$GPU" == "b300" ]]; then
  DOCKER_GPU_ARGS=(--gpus all)
else
  DOCKER_GPU_ARGS=(--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
fi
"${SSH[@]}" "root@$IP" bash -s <<REMOTE
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a
rm -f $REMOTE_EVENTS $PIP_FILE
docker run --rm --network=host --ipc=host \
  ${DOCKER_GPU_ARGS[*]} \
  -v /opt/bench-scripts:/scripts \
  -v /usr/local/bin/s5cmd:/host-bin/s5cmd:ro \
  -v "$SCRATCH":"$SCRATCH" -v /opt/bench:/opt/bench \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  "$CONTAINER" bash -lc 'set -euo pipefail
    t0=\$(date +%s)
    pip install -q "transformers>=4.51.0"
    t1=\$(date +%s)
    echo \$((t1-t0)) > $PIP_FILE
    torchrun --standalone --nproc_per_node=1 /scripts/train_bench.py \
      --model $MODEL --master-dtype fp32 \
      --batch-size $BATCH_SIZE --grad-accum $GRAD_ACCUM \
      --events $REMOTE_EVENTS \
      --ckpt-dir $SCRATCH/bench/ckpts \
      --resume-from $CKPT_LOCAL \
      --max-steps 3 --log-every 1 \
      --warmup-steps 0 --steady-steps 0 --num-ckpts 0'
REMOTE
t1=$(date +%s)
pip_install_s=$("${SSH[@]}" "root@$IP" "cat $PIP_FILE 2>/dev/null || echo 0")
log_event pip_install duration_s "${pip_install_s:-0}" note "timed inside container before torchrun; previously unaccounted wall during resume_train (not in first_step_s or container_start_s)"
spaces_env_clear "$IP"
"${SCP[@]}" "root@$IP:$REMOTE_EVENTS" "$EVENTS_DIR/remote_resume_events.jsonl"

python3 - "$EVENTS_DIR/remote_resume_events.jsonl" "$RESULT_JSON" "${PROVISION_S:-0}" "$env_s" "$download_s" "$DROPLET_ID" "${image_pull_s:-}" "${container_start_s:-}" "$BATCH_SIZE" "$GRAD_ACCUM" "$VARIANT" "${pip_install_s:-0}" <<'PY'
import json,sys
path,out,provision_s,env_s,download_s,did,image_pull_s,container_start_s,batch_size,grad_accum,variant,pip_install_s=sys.argv[1:]
load_s=None; first_step_s=None; loss=None; step=None
for line in open(path):
  e=json.loads(line)
  if e.get("event")=="ckpt_load_end":
    load_s=e.get("duration_s")
  if e.get("event")=="step" and first_step_s is None:
    first_step_s=e.get("step_time_s"); loss=e.get("loss"); step=e.get("step")
pip_f=float(pip_install_s or 0)
# Choice: include pip_install_s in R for every row (cold and lean both pip-install today).
R=float(provision_s)+float(env_s)+float(download_s)+pip_f+float(load_s or 0)+float(first_step_s or 0)
res={
  "variant": variant,
  "provision_s": float(provision_s),
  "env_s": float(env_s),
  "image_pull_s": float(image_pull_s) if image_pull_s not in ("",None) else "",
  "container_start_s": float(container_start_s) if container_start_s not in ("",None) else "",
  "download_s": float(download_s),
  "pip_install_s": pip_f,
  "pip_install_phase_note": "previously unaccounted inside resume_train wall; not inside first_step_s or container_start_s; now added into R_s",
  "load_s": load_s,
  "first_step_s": first_step_s,
  "R_s": R,
  "capacity_retries": 0,
  "retry_wait_s": 0.0,
  "droplet_id": int(did),
  "resume_step": step,
  "resume_loss": loss,
  "batch_size": int(batch_size),
  "grad_accum": int(grad_accum),
  "loss_match": "",
  "loss_match_strict": "",
  "simulated_reclaim": True,
}
json.dump(res, open(out,"w"), indent=2)
print(json.dumps(res, indent=2))
PY
echo "RESULT=$RESULT_JSON KEEP_DROPLET_ID=$DROPLET_ID"
