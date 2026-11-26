#!/usr/bin/env bash
# Start / stop / inspect the live report servers (run on the HOST, not in the container; they need only python3).
#
#   scripts/start_report_server.sh generation data/datasets/physiworld_pilot_v7_v0.7.0 [--port 60193]   live dataset report (generation)
#   scripts/start_report_server.sh training   outputs/sweeps/fix_v1 [--port 60194]                       live training page (charts)
#   scripts/start_report_server.sh status                                                                what is running, on which port
#   scripts/start_report_server.sh stop [generation|training]                                            stop one or both
#
# Both listen on 127.0.0.1 only (use an SSH tunnel:  ssh -L 3000:127.0.0.1:<port> user@server, then open http://localhost:3000/ for the
# generation report or http://localhost:3000/live.html for the training page). --public binds 0.0.0.0 instead (no login, plain HTTP).
# Closing your SSH session does not stop them. PID and port files: logs/report_<kind>.pid / .port, log: logs/report_<kind>.log.
# Details: src/physiworld_uav/visualization/README.md
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO" || exit 1
mkdir -p logs
cmd="${1:-status}"; shift || true
pidf() { echo "logs/report_$1.pid"; }; portf() { echo "logs/report_$1.port"; }
running() { [ -f "$(pidf "$1")" ] && kill -0 "$(cat "$(pidf "$1")")" 2>/dev/null; }
stop_one() { if running "$1"; then kill "$(cat "$(pidf "$1")")"; echo "stopped $1 (pid $(cat "$(pidf "$1")"))"; else echo "$1 is not running (from this script)"; fi; rm -f "$(pidf "$1")" "$(portf "$1")"; }
case "$cmd" in
  generation|training)
    kind="$cmd"; target="${1:?need the dataset directory (generation) or the sweep directory (training)}"; shift
    port=$([ "$kind" = generation ] && echo 60193 || echo 60194); host=127.0.0.1
    while [ $# -gt 0 ]; do case "$1" in --port) port="$2"; shift 2;; --public) host=0.0.0.0; shift;; *) echo "unknown option $1"; exit 2;; esac; done
    running "$kind" && { echo "$kind server already running on port $(cat "$(portf "$kind")") (pid $(cat "$(pidf "$kind")")); stop it first"; exit 1; }
    if ss -ltn 2>/dev/null | grep -q ":$port "; then echo "port $port is already in use; pick another with --port (this is a shared server)"; exit 1; fi
    PYTHONPATH="$REPO/src" nohup python3 -m physiworld_uav.visualization "$kind" $([ "$kind" = generation ] && echo "--dataset" || echo "--sweep") "$target" \
        --host "$host" --port "$port" > "logs/report_$kind.log" 2>&1 &
    echo $! > "$(pidf "$kind")"; echo "$port" > "$(portf "$kind")"; sleep 3
    if running "$kind"; then echo "$kind server started: http://$host:$port/$([ "$kind" = training ] && echo live.html)  (pid $(cat "$(pidf "$kind")"), log logs/report_$kind.log)"
    else echo "failed to start; see logs/report_$kind.log"; tail -5 "logs/report_$kind.log"; exit 1; fi;;
  status)
    for k in generation training; do
      if running "$k"; then echo "$k: running on port $(cat "$(portf "$k")") (pid $(cat "$(pidf "$k")"))"; else echo "$k: not started by this script"; fi
    done
    ss -ltnp 2>/dev/null | grep -E ":6019[0-9] " | awk '{print "listening: " $4}';;
  stop)
    if [ $# -ge 1 ]; then stop_one "$1"; else stop_one generation; stop_one training; fi;;
  *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2;;
esac
