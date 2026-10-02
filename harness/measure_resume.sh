#!/usr/bin/env bash
# Timed simulated reclaim + resume sequence.
# R = provision_s + env_s + download_s + load_s + first_step_s
# Approval for replacement Droplet MUST happen before this script starts timing.
#
# Usage:
#   measure_resume.sh \
#     --name NAME --size SLUG --region REGION --image IMAGE \
#     --gpu mi355x|b300 --spaces-uri s3://bucket/prefix/ckpt-dir \
#     --spaces-endpoint https://REGION.digitaloceanspaces.com \
#     --events-dir DIR --model 8B|1.7B [--smoke]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$HOME/.bench.env"

NAME="" SIZE="" REGION="" IMAGE="" GPU="" SPACES_URI="" SPACES_ENDPOINT=""
EVENTS_DIR="" MODEL="8B" SMOKE=0
GPU_TOOL="rocm-smi"
BATCH_SIZE=""
GRAD_ACCUM=""
RUN_META=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --spaces-uri) SPACES_URI="$2"; shift 2 ;;
    --spaces-endpoint) SPACES_ENDPOINT="$2"; shift 2 ;;
    --events-dir) EVENTS_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
    --run-meta) RUN_META="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    *) echo "unknown $1"; exit 1 ;;
  esac
done

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH=(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes)
SCP=(scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes)

# Prefer explicit CLI; else load from original run_meta.json (never auto-tune on resume)
if [[ -n "$RUN_META" && -f "$RUN_META" ]]; then
  if [[ -z "$BATCH_SIZE" ]]; then
    BATCH_SIZE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("batch_size",""))' "$RUN_META")
  fi
  if [[ -z "$GRAD_ACCUM" ]]; then
    GRAD_ACCUM=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("grad_accum",1))' "$RUN_META")
  fi
fi
[[ -n "$BATCH_SIZE" ]] || { echo "RESUME requires --batch-size or --run-meta with batch_size"; exit 1; }
GRAD_ACCUM="${GRAD_ACCUM:-1}"

[[ -n "$NAME" && -n "$SIZE" && -n "$REGION" && -n "$IMAGE" && -n "$GPU" && -n "$SPACES_URI" && -n "$EVENTS_DIR" ]] || {
  echo "missing required args"; exit 1
}

case "$GPU" in
  mi355x|MI355X) GPU_TOOL="rocm-smi" ;;
  b300|B300) GPU_TOOL="nvidia-smi" ;;
esac

mkdir -p "$EVENTS_DIR"
RESULT_JSON="$EVENTS_DIR/resume_result.json"
EVENTS="$EVENTS_DIR/events.jsonl"
: > "$EVENTS"

# Secrets stay in shell env from sourced ~/.bench.env for spaces_env_push only;
# never export onto remote ssh argv (see scripts/spaces_env_remote.sh).

capacity_retries=0
retry_wait_s=0
provision_s=0
env_s=0
download_s=0
load_s=0
first_step_s=0
DROPLET_ID=""
IP=""

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

# --- provision (timed on local machine) ---
t_prov0=$(date +%s)
log_event provision_start simulated_reclaim true

set +e
CREATE_OUT=$("$ROOT/scripts/provision.sh" create \
  --name "$NAME" --size "$SIZE" --region "$REGION" --image "$IMAGE" --wait 2>&1)
CREATE_RC=$?
set -e
if [[ $CREATE_RC -ne 0 ]]; then
  capacity_retries=$((capacity_retries+1))
  log_event capacity_error attempt 1 detail "$(echo "$CREATE_OUT" | tr '\n' ' ' | sed -E "s/${SPACES_KEY}/REDACTED/g; s/${SPACES_SECRET}/REDACTED/g")"
  echo "CREATE FAILED. Not counting wait in R. See events." >&2
  echo "$CREATE_OUT" >&2
  exit 2
fi

DROPLET_ID=$(echo "$CREATE_OUT" | awk -F= '/^CREATED_ID=/{print $2}' | tail -1)
IP_RAW=$("$ROOT/scripts/provision.sh" wait-ssh --id "$DROPLET_ID" --gpu-tool "$GPU_TOOL")
# wait-ssh may print status on stderr; keep only the last IPv4-looking token from stdout
IP=$(echo "$IP_RAW" | awk '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/{ip=$0} END{print ip}')
[[ -n "$IP" ]] || { echo "bad IP from wait-ssh: [$IP_RAW]"; exit 2; }
t_prov1=$(date +%s)
provision_s=$((t_prov1 - t_prov0))
log_event provision_end duration_s "$provision_s" droplet_id "$DROPLET_ID" ip "$IP"

# Copy scripts flat to droplet mount root (avoid nested scripts/)
"${SSH[@]}" "root@$IP" 'mkdir -p /tmp/spot-bench-scripts-upload /opt/bench-scripts'
"${SCP[@]}" "$ROOT/scripts/"*.py "$ROOT/scripts/"*.sh "root@$IP:/tmp/spot-bench-scripts-upload/"
"${SSH[@]}" "root@$IP" 'cp -f /tmp/spot-bench-scripts-upload/* /opt/bench-scripts/ && rm -rf /opt/bench-scripts/scripts /tmp/spot-bench-scripts-upload && grep -q grad-accum /opt/bench-scripts/train_bench.py'

# --- env (container pull) timed on droplet wall via ssh date ---
# Measure remotely with monotonic via a small wrapper
t_env0=$(date +%s)
log_event env_start
"${SSH[@]}" "root@$IP" bash -s -- "$GPU" <<'REMOTE'
set -euo pipefail
GPU="$1"
mkdir -p /opt/bench /var/log/spot-bench
# docker may need install on AI/ML image - usually present
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
# s5cmd
if ! command -v s5cmd >/dev/null; then
  curl -sL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin s5cmd
fi
bash /opt/bench-scripts/setup_node.sh --gpu "$GPU" --snapshot-dir /opt/bench/snapshot
REMOTE
t_env1=$(date +%s)
env_s=$((t_env1 - t_env0))
log_event env_end duration_s "$env_s"

SCRATCH=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/scratch.path)
CONTAINER=$("${SSH[@]}" "root@$IP" cat /opt/bench/snapshot/container.ref)
# Scratch must be DOSCRATCH NVMe, not boot disk
"${SSH[@]}" "root@$IP" 'python3 -c "import subprocess; r=subprocess.check_output([\"findmnt\",\"-n\",\"-o\",\"SOURCE\",\"/\"],text=True).strip(); s=subprocess.check_output([\"findmnt\",\"-n\",\"-o\",\"SOURCE\",\"/scratch\"],text=True).strip(); assert r!=s and s, (r,s)"' \
  || { echo "FATAL: /scratch missing or on boot disk (got SCRATCH=$SCRATCH)"; exit 2; }
[[ "$SCRATCH" == "/scratch" ]] || { echo "FATAL: scratch.path=$SCRATCH (want /scratch)"; exit 2; }

# Split image_pull_s / container_start_s from setup snapshot when present
image_pull_s=""
container_start_s=""
image_already_local=""
image_size_bytes=""
if "${SSH[@]}" "root@$IP" 'test -f /opt/bench/snapshot/env_timing.json'; then
  image_pull_s=$("${SSH[@]}" "root@$IP" 'python3 -c "import json; print(json.load(open(\"/opt/bench/snapshot/env_timing.json\")).get(\"image_pull_s\",\"\"))"')
  container_start_s=$("${SSH[@]}" "root@$IP" 'python3 -c "import json; print(json.load(open(\"/opt/bench/snapshot/env_timing.json\")).get(\"container_start_s\",\"\"))"')
  image_already_local=$("${SSH[@]}" "root@$IP" 'python3 -c "import json; print(json.load(open(\"/opt/bench/snapshot/env_timing.json\")).get(\"image_already_local\",\"\"))"')
  image_size_bytes=$("${SSH[@]}" "root@$IP" 'python3 -c "import json; print(json.load(open(\"/opt/bench/snapshot/env_timing.json\")).get(\"image_size_bytes\",\"\"))"')
  log_event env_split image_pull_s "${image_pull_s:-0}" container_start_s "${container_start_s:-0}" image_already_local "${image_already_local:-}" image_size_bytes "${image_size_bytes:-}"
fi
echo "RESUME_SCRATCH=$SCRATCH image_pull_s=${image_pull_s:-} image_already_local=${image_already_local:-}"

# --- download ---
# shellcheck source=/dev/null
source "$ROOT/scripts/spaces_env_remote.sh"
spaces_env_push "$IP"

CKPT_LOCAL="$SCRATCH/bench/resume-ckpt"
t_dl0=$(date +%s)
log_event download_start spaces_uri "$SPACES_URI"
"${SSH[@]}" "root@$IP" bash -s <<REMOTE
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a
rm -rf '$CKPT_LOCAL'; mkdir -p '$CKPT_LOCAL'
s5cmd --endpoint-url '$SPACES_ENDPOINT' cp '${SPACES_URI%/}/*' '$CKPT_LOCAL/'
REMOTE
t_dl1=$(date +%s)
download_s=$((t_dl1 - t_dl0))
log_event download_end duration_s "$download_s"

# --- load + first step via train_bench --resume-from --max-steps ---
# Run a short resume: load + a few steps; parse events for load_s and first_step_s
SMOKE_ARGS=""
[[ "$SMOKE" -eq 1 ]] && SMOKE_ARGS="--smoke"

REMOTE_EVENTS="/opt/bench/resume_events.jsonl"
PIP_FILE=/opt/bench/pip_install_s.txt
NPROC=$("${SSH[@]}" "root@$IP" "nvidia-smi -L 2>/dev/null | wc -l || rocm-smi -i 2>/dev/null | grep -c 'GPU\\[' || echo 1")
NPROC=${NPROC//$'\r'/}
NPROC=${NPROC##*}
[[ -z "$NPROC" || "$NPROC" -lt 1 ]] && NPROC=1

DOCKER_GPU_ARGS=()
case "$GPU" in
  mi355x|MI355X)
    DOCKER_GPU_ARGS=(--device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined)
    ;;
  b300|B300)
    DOCKER_GPU_ARGS=(--gpus all)
    ;;
esac

t_train0=$(date +%s)
log_event resume_train_start
"${SSH[@]}" "root@$IP" bash -s <<REMOTE
set -euo pipefail
set -a
# shellcheck disable=SC1091
source /opt/bench/.spaces.env
set +a
CONTAINER='$CONTAINER'
SCRATCH='$SCRATCH'
NPROC='$NPROC'
MODEL='$MODEL'
rm -f $REMOTE_EVENTS $PIP_FILE
docker run --rm --network=host --ipc=host \
  ${DOCKER_GPU_ARGS[*]} \
  -v /opt/bench-scripts:/scripts \
  -v /usr/local/bin/s5cmd:/host-bin/s5cmd:ro \
  -v "\$SCRATCH":"\$SCRATCH" \
  -v /opt/bench:/opt/bench \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY \
  "\$CONTAINER" \
  bash -lc 'set -euo pipefail
    t0=\$(date +%s)
    pip install -q "transformers>=4.51.0"
    t1=\$(date +%s)
    echo \$((t1-t0)) > '"$PIP_FILE"'
    torchrun --standalone --nproc_per_node='\$NPROC' /scripts/train_bench.py \
      --model '\$MODEL' \
      --master-dtype fp32 \
      --batch-size '"$BATCH_SIZE"' --grad-accum '"$GRAD_ACCUM"' \
      --events '"$REMOTE_EVENTS"' \
      --ckpt-dir '\$SCRATCH'/bench/ckpts \
      --resume-from '"$CKPT_LOCAL"' \
      --max-steps 3 \
      --warmup-steps 0 --steady-steps 0 --num-ckpts 0 \
      '"$SMOKE_ARGS"''
REMOTE
t_train1=$(date +%s)
pip_install_s=$("${SSH[@]}" "root@$IP" "cat $PIP_FILE 2>/dev/null || echo 0")
log_event pip_install duration_s "${pip_install_s:-0}" note "timed inside container before torchrun; previously unaccounted wall during resume_train (not in first_step_s or container_start_s)"
spaces_env_clear "$IP"

# Fetch remote events and extract load / first step durations
"${SCP[@]}" "root@$IP:$REMOTE_EVENTS" "$EVENTS_DIR/remote_resume_events.jsonl" || true
ORIG_EVENTS=""
if [[ -n "$RUN_META" && -f "$RUN_META" ]]; then
  cand="$(dirname "$RUN_META")/events.jsonl"
  [[ -f "$cand" ]] && ORIG_EVENTS="$cand"
fi
# Steady-state step time from source training events (not first_step_s)
STEADY_STEP=""
if [[ -n "$ORIG_EVENTS" || ( -n "$RUN_META" && -f "$(dirname "$RUN_META")/events.jsonl" ) ]]; then
  cand="${ORIG_EVENTS:-$(dirname "$RUN_META")/events.jsonl}"
  if [[ -f "$cand" ]]; then
    STEADY_STEP=$(python3 -c 'import json,statistics,sys
xs=[]
for line in open(sys.argv[1]):
  e=json.loads(line)
  if e.get("event")=="step" and e.get("step",0)>=20 and "step_time_s" in e:
    xs.append(float(e["step_time_s"]))
print(statistics.median(xs) if xs else "")' "$cand")
  fi
fi
python3 - "$EVENTS_DIR/remote_resume_events.jsonl" "$RESULT_JSON" "$provision_s" "$env_s" "$download_s" "$capacity_retries" "$retry_wait_s" "$DROPLET_ID" "${image_pull_s:-}" "${container_start_s:-}" "$BATCH_SIZE" "$GRAD_ACCUM" "${ORIG_EVENTS}" "${image_size_bytes:-}" "${STEADY_STEP}" "${pip_install_s:-0}" <<'PY'
import json,sys
path, out, provision_s, env_s, download_s, cap_r, retry_w, did, image_pull_s, container_start_s, batch_size, grad_accum, orig_events, image_size_bytes, steady_step, pip_install_s = sys.argv[1:]
load_s=None
first_step_s=None
loss=None
step=None
if path:
  try:
    for line in open(path):
      e=json.loads(line)
      if e.get("event")=="ckpt_load_end":
        load_s=e.get("duration_s")
      if e.get("event")=="step" and first_step_s is None:
        first_step_s=e.get("step_time_s")
        loss=e.get("loss")
        step=e.get("step")
  except FileNotFoundError:
    pass
loss_match=""
loss_match_note=""
if orig_events and loss is not None and step is not None:
  try:
    by_step={}
    for line in open(orig_events):
      e=json.loads(line)
      if e.get("event")=="step" and "loss" in e:
        by_step[int(e["step"])]=float(e["loss"])
    if int(step) in by_step:
      o=by_step[int(step)]; n=float(loss)
      loss_match = abs(o-n) <= max(1e-3, 1e-5*abs(o))
      loss_match_note="exact_step_vs_original_events"
    else:
      # Original run often logs every N steps; compare nearest prior logged step only as note
      priors=[s for s in by_step if s < int(step)]
      if priors:
        ps=max(priors)
        loss_match_note=f"no_original_loss_at_step_{step}; nearest_prior_step_{ps}_loss_{by_step[ps]}"
      else:
        loss_match_note=f"no_original_loss_at_step_{step}"
  except Exception as ex:
    loss_match_note=f"compare_error:{ex}"
pip_f=float(pip_install_s or 0)
# Include pip_install_s in R for every row (cold + lean both install today).
R = float(provision_s)+float(env_s)+float(download_s)+pip_f+float(load_s or 0)+float(first_step_s or 0)
# Resume from durable ckpt-100 to step 101 => 0 steps lost after last checkpoint
steps_lost = 0 if step in (101, "101") else ""
res={
  "variant": "cold",
  "provision_s": float(provision_s),
  "env_s": float(env_s),
  "image_pull_s": float(image_pull_s) if image_pull_s not in ("", None) else "",
  "container_start_s": float(container_start_s) if container_start_s not in ("", None) else "",
  "image_size_bytes": int(float(image_size_bytes)) if image_size_bytes not in ("", None) else "",
  "download_s": float(download_s),
  "pip_install_s": pip_f,
  "pip_install_phase_note": "previously unaccounted inside resume_train wall; not inside first_step_s or container_start_s; now added into R_s",
  "load_s": load_s,
  "first_step_s": first_step_s,
  "step_time_s": float(steady_step) if steady_step not in ("", None) else "",
  "steps_lost": steps_lost,
  "R_s": R,
  "capacity_retries": int(cap_r),
  "retry_wait_s": float(retry_w),
  "droplet_id": int(did),
  "resume_step": step,
  "resume_loss": loss,
  "batch_size": int(batch_size),
  "grad_accum": int(grad_accum),
  "loss_match": loss_match,
  "loss_match_note": loss_match_note,
  "storage": "scratch",
  "simulated_reclaim": True,
}
json.dump(res, open(out,"w"), indent=2)
print(json.dumps(res, indent=2))
PY

echo "RESUME_DROPLET_ID=$DROPLET_ID"
echo "RESULT=$RESULT_JSON"
