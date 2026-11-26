#!/usr/bin/env bash
# One-time setup of phone notifications through ntfy (run on the host). Creates a random topic in logs/ntfy.conf (git-ignored) and prints how
# to subscribe. Usage:  scripts/setup_ntfy.sh [--server https://ntfy.sh] [--topic NAME] [--reset]
# Install the free "ntfy" app (Android: Play Store / F-Droid, iOS: App Store), tap "+", enter the topic (and the server if not ntfy.sh).
# The topic is the only secret on the public server: do not share it. Notices carry no project, dataset or user names (see notify.py).
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO" || exit 1; mkdir -p logs
server="https://ntfy.sh"; topic=""; reset=0
while [ $# -gt 0 ]; do case "$1" in --server) server="$2"; shift 2;; --topic) topic="$2"; shift 2;; --reset) reset=1; shift;; *) echo "unknown option $1"; exit 2;; esac; done
if [ -f logs/ntfy.conf ] && [ "$reset" = 0 ]; then echo "already set up (logs/ntfy.conf); use --reset for a new topic"; else
  [ -n "$topic" ] || topic="$(python3 -c 'import secrets;print(secrets.token_urlsafe(14).replace("_","x").replace("-","y"))')"
  printf 'topic=%s\nserver=%s\n' "$topic" "$server" > logs/ntfy.conf; chmod 600 logs/ntfy.conf; echo "created logs/ntfy.conf"
fi
. <(sed 's/^/NTFY_/;s/^NTFY_topic/NTFY_TOPIC/;s/^NTFY_server/NTFY_SERVER/' logs/ntfy.conf)
echo "server: $NTFY_SERVER"; echo "topic : $NTFY_TOPIC"
echo "In the ntfy app: + -> topic '$NTFY_TOPIC'$([ "$NTFY_SERVER" != https://ntfy.sh ] && echo " (custom server $NTFY_SERVER)")."
PYTHONPATH="$REPO/src" python3 -m physiworld_uav.notify --test
