#!/usr/bin/env bash
# Push Spaces creds to a Droplet via scp env file (chmod 600), never on ssh argv.
# Usage: source this file; spaces_env_push IP; ... remote work sourcing /opt/bench/.spaces.env;
#        spaces_env_clear IP
spaces_env_push() {
  local ip="${1:?ip}"
  local tmp
  tmp="$(mktemp)"
  umask 077
  # Values may contain special chars; write as single-quoted shell assignments safely
  python3 - "$tmp" <<'PY'
import os, sys
path=sys.argv[1]
def shquote(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"
open(path,"w").write(
  "export AWS_ACCESS_KEY_ID=%s\nexport AWS_SECRET_ACCESS_KEY=%s\n"
  % (shquote(os.environ["SPACES_KEY"]), shquote(os.environ["SPACES_SECRET"]))
)
PY
  ssh -i "${SSH_KEY:-$HOME/.ssh/id_ed25519}" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes \
    "root@$ip" 'mkdir -p /opt/bench && rm -f /opt/bench/.spaces.env'
  scp -i "${SSH_KEY:-$HOME/.ssh/id_ed25519}" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes \
    "$tmp" "root@$ip:/opt/bench/.spaces.env"
  ssh -i "${SSH_KEY:-$HOME/.ssh/id_ed25519}" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes \
    "root@$ip" 'chmod 600 /opt/bench/.spaces.env'
  rm -f "$tmp"
}
spaces_env_clear() {
  local ip="${1:?ip}"
  ssh -i "${SSH_KEY:-$HOME/.ssh/id_ed25519}" -o StrictHostKeyChecking=no -o IdentitiesOnly=yes -o BatchMode=yes \
    "root@$ip" 'rm -f /opt/bench/.spaces.env'
}
