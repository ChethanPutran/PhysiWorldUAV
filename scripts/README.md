scripts/start_generation.sh now starts the generation and the status web page together, with timestamped logs. I tested it on the smoke experiment, and everything worked. I did not run a real multi-hour generation with it. The first real run will be the real proof.

Usage (run on the server host, not inside the container)
scripts/start_generation.sh pilot                  # start both; page is local-only (SSH tunnel)
scripts/start_generation.sh pilot --public         # page open to the campus network (no login)
scripts/start_generation.sh pilot --port 8790      # choose the page port
scripts/start_generation.sh pilot --status         # what is running, plus progress
scripts/start_generation.sh pilot --stop           # stop both; written episodes are kept

What it does
1. Pre-checks. It checks that the container is up and that no generation or pagedy running. It also works out the dataset folder from the experiment config.
2. Generation. It starts generate_until_complete.sh detached inside the container, so closing SSH doesn't stop it. It then checks the process is still alive after 5 seconds.
3. Plan. It waits until the plan (manifest.jsonl) exists, so the page has someth
4. Status page. It starts the page detached on the first free port from 8777 to 8799, or the port you pass, and checks that it answers.
5. Summary. It prints the log paths and the address to open, which is an SSH tun with --public.

Logs are in logs/, which is now git-ignored. Each start writes these files:
- <exp>_<time>_launcher.log
- <exp>_<time>_generation.log
- <exp>_<time>_server.log

There are also <exp>_server.pid and <exp>_server.port files that --status and --stop use. Every message has a timestamp, and failures print the last log lines before exiting.

Test results
- Both processes started, and the page answered on / and /index.json.
- --status showed what was running, and --stop stopped both.
- smoke reports "finished with problems". That is stale Run 3 data: its validati fresh frames and the drone parameters weren't read back from PhysX. Both werefixed in Run 4. It isn't a launcher problem.
- The test refreshed that smoke dataset's manifest, provenance and index files,
- For an incomplete dataset the wrapper retries once, sees no new episodes, and stops. That is why --status showed the generation as running for a short while.

