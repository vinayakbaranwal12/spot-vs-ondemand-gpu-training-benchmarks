#!/usr/bin/env bash
# Launch batch + watchdog fully detached (survives Cursor shell exit + Mac sleep via caffeinate).
ROOT="."
python3 - <<'PY'
import subprocess, pathlib, time
root = pathlib.Path(".")
jobs = [
  ("batch", ["caffeinate","-dimsu","bash",str(root/"scripts/run_lean_recover_then_strict.sh")], root/"logs/lean_recover_then_strict.nohup", "/tmp/bench_lean_recover.pid"),
  ("watchdog", ["caffeinate","-dimsu","bash",str(root/"scripts/batch_watchdog.sh")], root/"logs/watchdog_runner.nohup", "/tmp/bench_watchdog.pid"),
]
for name, cmd, log, pidfile in jobs:
  out = open(log, "a")
  out.write(f"\n=== launch_detached {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===\n")
  out.flush()
  p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=out, cwd=str(root), start_new_session=True, close_fds=True)
  pathlib.Path(pidfile).write_text(str(p.pid)+"\n")
  print(name, p.pid)
PY
