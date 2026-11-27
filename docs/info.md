# Start a dataset generation, watch it, get phone alerts

Run everything below on the server **host** (not inside the container). Closing your SSH session does not stop the generation, the status page or the notifier.

## 1. One-time setup: phone notifications (optional)

The notification URL is a secret. Keep it in a private file, never in the repo or in chat.

### ntfy (simplest, no account)
1. Install the **ntfy** app on your phone.
2. Make a hard-to-guess topic name (it acts as the password):
   ```bash
   echo "physiworld-$(openssl rand -hex 8)"
   ```
3. In the app, tap **+** and subscribe to that topic (server `ntfy.sh`).
4. Save the URL and test it:
   ```bash
   umask 077
   echo 'https://ntfy.sh/<your-topic>' > ~/.physiworld_notify
   curl -d "test from the lab server" "$(cat ~/.physiworld_notify)"
   ```
   Your phone should show the message within a few seconds.

### Telegram
1. In Telegram, chat with `@BotFather`, send `/newbot`, and copy the token.
2. Open your new bot and press **Start**.
3. Get your chat id from `https://api.telegram.org/bot<TOKEN>/getUpdates` (`"chat":{"id": <number>`).
4. Save the URL and test it:
   ```bash
   umask 077
   echo 'https://api.telegram.org/bot<TOKEN>/sendMessage?chat_id=<CHAT_ID>' > ~/.physiworld_notify
   curl -s "$(cat ~/.physiworld_notify)" -H 'Content-Type: application/json' -d '{"text":"test from the lab server"}'
   ```

Both services are untested with this project's watcher against the real servers, so run the test above first.

If `~/.physiworld_notify` does not exist (and `PHYSIWORLD_NOTIFY_URL` is not set), the launcher still works and simply skips phone alerts.

## 2. Start

```bash
cd /data2/chethan/mtech/PhysiWorldUAV
scripts/start_generation.sh pilot                 # local-only status page (SSH tunnel)
scripts/start_generation.sh pilot --public        # status page open to the campus network (no login, plain HTTP)
scripts/start_generation.sh pilot --port 8780     # choose the page port (default: first free in 8777-8799)
```

`pilot` is the experiment name (`config/experiment/`); use `smoke` for a quick test.

It starts, all detached:

| What | How | Log |
|---|---|---|
| Generation | `scripts/generate_until_complete.sh` in the container; restarts with `--resume` after a crash, skips episodes already written | `logs/<exp>_<time>_generation.log` |
| Status page | `physiworld_uav.visualization.dataset_report --serve`, follows `index.json` live | `logs/<exp>_<time>_server.log` |
| Notifier (only if a URL exists) | `physiworld_uav.experiments.notify` | `logs/<exp>_<time>_notify.log` |
| The launcher itself | | `logs/<exp>_<time>_launcher.log` |

At the end it prints how to open the page.

Notifier format is auto-detected from the URL (Telegram, Slack, Discord, otherwise plain text). Override with `NOTIFY_FORMAT=text scripts/start_generation.sh pilot`.

Messages sent: start, every 10 % of progress, each failed episode, a stall warning (no `index.json` update for 120 s), a periodic "still running", and finish.

## 3. Open the status page

* **Local-only (default):** from your laptop
  ```bash
  ssh -L <port>:localhost:<port> <user>@<server-ip>
  ```
  then open `http://localhost:<port>/`.
* **`--public`:** open `http://<server-ip>:<port>/` from anywhere on the campus network (also over the college Wi-Fi from the hostel).

The page refreshes every 5 s while running and warns if `index.json` has not changed for 120 s.

## 4. Check, stop, resume

```bash
scripts/start_generation.sh pilot --status        # what is running, progress, notifier state
scripts/start_generation.sh pilot --stop          # stops generation, status page and notifier
tail -f logs/pilot_*_generation.log               # live generator log
```

* Episodes already written are kept. To continue after `--stop`, a crash or a server reboot, run the same start command again: it resumes and skips finished episodes.
* A reboot does **not** restart anything on its own (container restart policy is `no`); start it again after a reboot, with `docker start isaac-sim` first if the container is down.
* A failing episode is logged with its traceback under `failures` in `index.json`, and the run continues.
* The start command refuses to run if a generation is already running.

## 5. Checking from another session

Log in again and either run `--status`, or ask Claude to read `data/datasets/<name>/index.json` and the logs in `logs/`.

## 6. When it finishes

`index.json` shows `complete: true` and the validator result. Validate again or build the report if you want:

```bash
docker exec isaac-sim bash -c 'cd /workspace/PhysiWorldUAV && source omnidrones_env/bin/activate && export PYTHONPATH=$PWD/src:$PYTHONPATH && python -m physiworld_uav.experiments.generate validate --dataset data/datasets/<name>'
PYTHONPATH=src python3 -m physiworld_uav.visualization.dataset_report --dataset data/datasets/<name>   # report.html
```
