#!/usr/bin/env bash
# Start a dataset generation AND its live status web page, both detached from your terminal.
# Run this on the server HOST (not inside the container). Closing your SSH session does not stop either.
#
#   scripts/start_generation.sh <experiment> [--port N] [--public]    start (e.g. experiment name: pilot)
#   scripts/start_generation.sh <experiment> --status                 show what is running and the progress
#   scripts/start_generation.sh <experiment> --stop                   stop the generation and the web page
#
# --public   serve on 0.0.0.0 so anyone on the campus network can open it (no login, plain HTTP).
#            Without it the page listens on 127.0.0.1 only and you reach it through an SSH tunnel.
# --port N   page port (default: first free port in 8777-8799)
#
# What it starts:
#   1. generation:  scripts/generate_until_complete.sh experiment=<name> inside the container
#                   (restarts with --resume after a crash, skips episodes already written)
#   2. status page: physiworld_uav.visualization.dataset_report --serve  (follows index.json live)
# Logs (all in <repo>/logs/, timestamped, git-ignored):
#   <exp>_<time>_launcher.log   this script       <exp>_<time>_generation.log   the generator
#   <exp>_<time>_server.log     the web page      <exp>_server.pid / .port      for --status and --stop
set -euo pipefail
# Phone notifications: URL comes from the environment, else from ~/.physiworld_notify (private file, never committed).
if [ -z "${PHYSIWORLD_NOTIFY_URL:-}" ] && [ -r "$HOME/.physiworld_notify" ]; then
  PHYSIWORLD_NOTIFY_URL="$(cat "$HOME/.physiworld_notify")"
fi
export PHYSIWORLD_NOTIFY_URL="${PHYSIWORLD_NOTIFY_URL:-}"
NOTIFY_FORMAT="${NOTIFY_FORMAT:-}"   # text | slack | discord | telegram (auto-detected from the URL if empty)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="${CONTAINER:-isaac-sim}"
CROOT=/workspace/PhysiWorldUAV                     # repo path inside the container
LOG_DIR="$ROOT/logs"
LAUNCH_LOG=""

log() { local lvl="$1"; shift; local line; line="$(date '+%Y-%m-%d %H:%M:%S') [$lvl] $*"
        echo "$line"; if [ -n "$LAUNCH_LOG" ]; then echo "$line" >> "$LAUNCH_LOG"; fi; }
die() { log ERROR "$*"; exit 1; }
usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

MODE=start; EXP=""; PUBLIC=0; PORT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --public) PUBLIC=1 ;;
    --port)   PORT="${2:-}"; [ -n "$PORT" ] || die "--port needs a number"; shift ;;
    --stop)   MODE=stop ;;
    --status) MODE=status ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1 (see --help)" ;;
    *) [ -z "$EXP" ] || die "unexpected argument: $1"; EXP="$1" ;;
  esac
  shift
done
[ -n "$EXP" ] || { usage; exit 2; }

PID_FILE="$LOG_DIR/${EXP}_server.pid"; PORT_FILE="$LOG_DIR/${EXP}_server.port"; NOTIFY_PID="$LOG_DIR/${EXP}_notify.pid"
notify_pid() { [ -f "$NOTIFY_PID" ] && kill -0 "$(cat "$NOTIFY_PID")" 2>/dev/null && cat "$NOTIFY_PID" || true; }
server_pid() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null && cat "$PID_FILE" || true; }
gen_running() { docker exec "$CONTAINER" pgrep -f "generate_until_complete.sh|experiments.generate run" >/dev/null 2>&1; }
port_busy() { ss -ltnH "sport = :$1" 2>/dev/null | grep -q .; }

dataset_dirs() {                                   # sets DS_CONT and DS_HOST from the experiment config
  local code out
  code="from physiworld_uav.config import compose, to_plain; from physiworld_uav.experiments.generate import dataset_dir; print('DATASET=' + str(dataset_dir(to_plain(compose(['experiment=$EXP'])))))"
  out="$(docker exec "$CONTAINER" bash -c "cd $CROOT && source omnidrones_env/bin/activate && export PYTHONPATH=\$PWD/src:\$PYTHONPATH && python -c \"$code\"" 2>&1)" \
    || die "could not read the experiment config for '$EXP': $(printf '%s' "$out" | tail -3)"
  DS_CONT="$(printf '%s\n' "$out" | sed -n 's/^DATASET=//p' | tail -1)"
  [ -n "$DS_CONT" ] || die "no dataset path for experiment '$EXP' (is the name right?)"
  DS_HOST="${DS_CONT/#$CROOT/$ROOT}"
}

summary() {                                        # one-line progress from index.json
  python3 - "$DS_HOST/index.json" <<'PY'
import json, sys, time, os
p = sys.argv[1]
if not os.path.exists(p):
    print("index.json: not written yet"); raise SystemExit
d = json.load(open(p)); age = time.time() - os.path.getmtime(p)
state = "running" if d.get("in_progress") else ("complete" if d.get("complete") else "finished with problems")
early = sum(1 for e in d["episodes"] if e.get("status") != "completed")
print(f"progress: {d['n_written']}/{d['n_planned']} written, {len(d.get('failures') or [])} failed, {early} ended early; "
      f"state: {state}; index updated {age:.0f}s ago")
PY
}

# ---------------------------------------------------------------- status / stop
if [ "$MODE" = status ]; then
  dataset_dirs
  echo "experiment : $EXP"; echo "dataset    : $DS_HOST"
  if gen_running; then echo "generation : RUNNING"; else echo "generation : not running"; fi
  if [ -n "$(server_pid)" ]; then echo "status page: RUNNING (pid $(cat "$PID_FILE"), port $(cat "$PORT_FILE" 2>/dev/null || echo ?))"
  else echo "status page: not running"; fi
  if [ -n "$(notify_pid)" ]; then echo "notifier   : RUNNING (pid $(cat "$NOTIFY_PID"))"; else echo "notifier   : not running"; fi
  summary; exit 0
fi
if [ "$MODE" = stop ]; then
  mkdir -p "$LOG_DIR"; LAUNCH_LOG="$LOG_DIR/${EXP}_$(date +%Y%m%d_%H%M%S)_launcher.log"
  if [ -n "$(server_pid)" ]; then kill "$(cat "$PID_FILE")" && log INFO "status page stopped"; rm -f "$PID_FILE" "$PORT_FILE"
  else log INFO "status page was not running"; fi
  if [ -n "$(notify_pid)" ]; then kill "$(cat "$NOTIFY_PID")" && log INFO "notifier stopped"; rm -f "$NOTIFY_PID"; fi
  if gen_running; then
    docker exec "$CONTAINER" pkill -f generate_until_complete.sh || true     # wrapper first, so it does not restart the generator
    docker exec "$CONTAINER" pkill -f "experiments.generate run" || true
    log INFO "generation stopped (episodes already written are kept; restart with this script to resume)"
  else log INFO "generation was not running"; fi
  exit 0
fi

# ---------------------------------------------------------------- start
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"; BASE="${EXP}_${STAMP}"
LAUNCH_LOG="$LOG_DIR/${BASE}_launcher.log"
GEN_LOG="$LOG_DIR/${BASE}_generation.log"; SRV_LOG="$LOG_DIR/${BASE}_server.log"
log INFO "starting '$EXP' (logs in $LOG_DIR)"

[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || echo false)" = true ] \
  || die "container '$CONTAINER' is not running (docker start $CONTAINER)"
if gen_running; then die "a generation is already running; check with: $0 $EXP --status   or stop it with: $0 $EXP --stop"; fi
if [ -n "$(server_pid)" ]; then die "a status page for '$EXP' is already running (pid $(cat "$PID_FILE")); stop it with: $0 $EXP --stop"; fi

dataset_dirs
log INFO "dataset directory: $DS_HOST"

log INFO "starting generation in container '$CONTAINER' -> $GEN_LOG"
docker exec -d "$CONTAINER" bash -c "cd $CROOT && source omnidrones_env/bin/activate && export PYTHONPATH=\$PWD/src:\$PYTHONPATH && scripts/generate_until_complete.sh experiment=$EXP > logs/$(basename "$GEN_LOG") 2>&1"
sleep 5
if ! gen_running; then
  log ERROR "generation exited right away; last log lines:"; tail -n 15 "$GEN_LOG" 2>/dev/null | tee -a "$LAUNCH_LOG" || true; exit 1
fi
log INFO "generation is running"

log INFO "waiting for the plan (manifest.jsonl) so the status page has something to show"
for _ in $(seq 1 90); do
  [ -s "$DS_HOST/manifest.jsonl" ] && break
  gen_running || die "generation stopped before writing the plan; see $GEN_LOG"
  sleep 2
done
[ -s "$DS_HOST/manifest.jsonl" ] || die "no manifest after 180 s; see $GEN_LOG"
log INFO "plan found: $(wc -l < "$DS_HOST/manifest.jsonl") episodes"

if [ -z "$PORT" ]; then
  for p in $(seq 8777 8799); do port_busy "$p" || { PORT="$p"; break; }; done
  [ -n "$PORT" ] || die "no free port in 8777-8799; pass --port N"
elif port_busy "$PORT"; then die "port $PORT is already in use; pick another with --port"; fi
HOST_BIND=127.0.0.1; [ "$PUBLIC" = 1 ] && HOST_BIND=0.0.0.0

log INFO "starting the status page on $HOST_BIND:$PORT -> $SRV_LOG"
setsid nohup env PYTHONPATH="$ROOT/src" python3 -m physiworld_uav.visualization.dataset_report \
  --dataset "$DS_HOST" --serve --host "$HOST_BIND" --port "$PORT" > "$SRV_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"; echo "$PORT" > "$PORT_FILE"
sleep 2
kill -0 "$(cat "$PID_FILE")" 2>/dev/null || { log ERROR "status page failed to start:"; tail -n 10 "$SRV_LOG" | tee -a "$LAUNCH_LOG"; exit 1; }
curl -fsS -o /dev/null "http://127.0.0.1:$PORT/" || die "status page is up but did not answer on port $PORT; see $SRV_LOG"
log INFO "status page is up (pid $(cat "$PID_FILE"))"

NOTIFY_LOG="$LOG_DIR/${BASE}_notify.log"
if [ -n "$PHYSIWORLD_NOTIFY_URL" ]; then
  if [ -z "$NOTIFY_FORMAT" ]; then
    case "$PHYSIWORLD_NOTIFY_URL" in *api.telegram.org*) NOTIFY_FORMAT=telegram ;; *hooks.slack.com*) NOTIFY_FORMAT=slack ;;
      *discord.com*) NOTIFY_FORMAT=discord ;; *) NOTIFY_FORMAT=text ;; esac
  fi
  setsid nohup env PYTHONPATH="$ROOT/src" python3 -m physiworld_uav.experiments.notify \
    --dataset "$DS_HOST" --format "$NOTIFY_FORMAT" > "$NOTIFY_LOG" 2>&1 < /dev/null &
  echo $! > "$NOTIFY_PID"; sleep 2
  if kill -0 "$(cat "$NOTIFY_PID")" 2>/dev/null; then log INFO "notifier is running (format $NOTIFY_FORMAT) -> $NOTIFY_LOG"
  else log WARN "notifier failed to start; see $NOTIFY_LOG"; rm -f "$NOTIFY_PID"; fi
else
  log INFO "no notification URL (PHYSIWORLD_NOTIFY_URL or ~/.physiworld_notify); phone alerts are off"
fi

SERVER_IP="$(hostname -I | awk '{print $1}')"
{
  echo
  echo "  generation : running   log: $GEN_LOG"
  echo "  status page: running   log: $SRV_LOG"
  if [ "$PUBLIC" = 1 ]; then
    echo "  open       : http://$SERVER_IP:$PORT/   (anyone on the campus network can see this page)"
  else
    echo "  open       : ssh -L $PORT:localhost:$PORT <user>@$SERVER_IP   then   http://localhost:$PORT/"
  fi
  echo "  later      : $0 $EXP --status    |    $0 $EXP --stop    |    tail -f $GEN_LOG"
  echo
} | tee -a "$LAUNCH_LOG"
