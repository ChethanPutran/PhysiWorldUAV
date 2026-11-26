import json
import re

from physiworld_uav.visualization import dataset_report as dr


def _row(i, part="id_cl", split="train", cf=None):
    return {"episode_id": f"e{i}", "group_id": f"g{i // 2}", "part": part, "split_role": split,
            "data_type": "closed_loop", "scenario_template": "hover", "duration_s": 5.0, "n_steps": 1000,
            "controller": {"mode": "position"}, "counterfactual": cf, "replay_of": None,
            "domain": {"domain_id": "d0", "region": "id", "factors": {"mass": 1.02}, "drone": {"mass_kg": 0.73}},
            "wind": {"category": "id", "profile": "steady", "labels": {"gust_class": "none", "temporal_class": "constant"},
                     "stats": {"mean_horizontal_speed": 1.5, "peak_speed": 2.0}},
            "reference": {"family": "hover"}, "excitation": None, "experiment_group": "A", "ood_category": None}


def _write(tmp_path, rows, index=None):
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    if index is not None:
        (tmp_path / "index.json").write_text(json.dumps(index))


def test_report_from_manifest_only(tmp_path):
    _write(tmp_path, [_row(0), _row(1, cf={"is_baseline": True, "diff": {}})])
    html = dr.build_report(tmp_path)
    assert "__DATA__" not in html and "__META__" not in html
    rows = json.loads(re.search(r"const ROWS = (\[.*?\]);\nconst META", html, re.S).group(1))
    assert [r["id"] for r in rows] == ["e0", "e1"] and rows[1]["cf"] == "baseline" and rows[0]["written"] is None


def test_report_uses_index_and_survives_missing_fields(tmp_path):
    bad = _row(2)
    bad["wind"] = {}                                   # no wind block at all
    bad["reference"] = None
    _write(tmp_path, [_row(0), bad],
           {"dataset_id": "t", "dataset_version": "1", "n_planned": 2, "n_written": 1, "complete": False,
            "validation": {"ok": False, "hard": 1, "soft": 0}, "episodes": [{"episode_id": "e0"}]})
    html = dr.build_report(tmp_path)
    meta = json.loads(re.search(r"const META = (\{.*?\});", html, re.S).group(1))
    assert meta["has_index"] and meta["n_written"] == 1 and meta["hard"] == 1
    rows = json.loads(re.search(r"const ROWS = (\[.*?\]);\nconst META", html, re.S).group(1))
    assert rows[0]["written"] is True and rows[1]["written"] is False and rows[1]["wm"] == 0.0


def test_cli_writes_report(tmp_path):
    _write(tmp_path, [_row(0)])
    out = dr.main(["--dataset", str(tmp_path)])
    assert out.exists() and out.stat().st_size > 1000


def test_version_falls_back_to_directory_name(tmp_path):
    d = tmp_path / "physiworld_pilot_v0.5.0"
    d.mkdir()
    _write(d, [_row(0)])
    html = dr.build_report(d)
    meta = json.loads(re.search(r"const META = (\{.*?\});", html, re.S).group(1))
    assert meta["dataset_version"] == "0.5.0"
    assert '<meta charset="utf-8">' in html


def test_live_server_serves_report_and_index(tmp_path):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    _write(tmp_path, [_row(0), _row(1)],
           {"dataset_id": "t", "dataset_version": "1", "n_planned": 2, "n_written": 1, "complete": False,
            "in_progress": True, "validation": {"ok": None, "hard": None, "soft": None},
            "episodes": [{"episode_id": "e0", "status": "completed", "n_steps": 10}], "failures": []})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dr.make_handler(tmp_path))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert '"live":true' in page and '"status":"completed"' in page
        r = urllib.request.urlopen(base + "/index.json")
        assert json.loads(r.read())["in_progress"] is True and float(r.headers["X-Index-Age-S"]) >= 0
        with __import__("pytest").raises(urllib.error.HTTPError) as e:     # nothing else is exposed
            urllib.request.urlopen(base + "/manifest.jsonl")
        assert e.value.code == 404
    finally:
        srv.shutdown()


def _idx(n, total=10, fails=0, in_progress=True, complete=False, early=0):
    eps = [{"episode_id": f"e{i}", "status": "terminated_ground" if i < early else "completed"} for i in range(n)]
    return {"dataset_id": "t", "dataset_version": "1", "n_planned": total, "n_written": n,
            "failures": [{"episode_id": f"f{i}"} for i in range(fails)], "in_progress": in_progress,
            "complete": complete, "generation_wall_s": 100.0, "episodes": eps,
            "validation": {"ok": True, "hard": 0, "soft": 0} if not in_progress else {"ok": None}}


def test_notify_decisions_cover_start_progress_failure_stall_and_finish():
    from physiworld_uav.experiments import notify as nt
    st = nt.WatchState(last_heartbeat=0.0)
    assert nt.decide(st, None, 0, now=0) == []
    m = nt.decide(st, _idx(1), 5, now=10)                          # start; 10 % bucket reached
    assert any("running" in x for x in m) and any("Progress" in x for x in m)
    assert nt.decide(st, _idx(1), 5, now=20) == []                  # nothing new -> silent
    assert any("failed" in x and "f0" in x for x in nt.decide(st, _idx(2, fails=1), 5, now=30))
    assert any("No update" in x for x in nt.decide(st, _idx(2, fails=1), 2000, now=40))
    assert nt.decide(st, _idx(2, fails=1), 2000, now=50) == []      # one stall alert only
    fin = nt.decide(st, _idx(10, in_progress=False, complete=True, early=2), 1, now=60)
    assert len(fin) == 1 and "complete" in fin[0] and "2 ended early" in fin[0]
    assert nt.decide(st, _idx(10, in_progress=False, complete=True), 1, now=70) == []
    assert "/" not in "".join(fin).replace("10/10", "")            # no paths in messages


def test_notify_posts_text_and_json(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from physiworld_uav.experiments import notify as nt
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append((self.headers["Content-Type"], self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/t"
    try:
        nt.send(url, "hello", "text")
        nt.send(url, "hello", "slack")
        nt.send(url, "hello", "discord")
    finally:
        srv.shutdown()
    assert got[0][1] == b"hello" and json.loads(got[1][1]) == {"text": "hello"} and json.loads(got[2][1]) == {"content": "hello"}


def test_live_index_is_assembled_from_worker_files_during_a_parallel_generation(tmp_path):
    import json
    from physiworld_uav.visualization.dataset_report import combined_index
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps({"episode_id": f"e{i}"}) for i in range(5)) + "\n")
    assert combined_index(tmp_path) is None                                              # no plan yet
    (tmp_path / "provenance.json").write_text("{}")
    assert json.loads(combined_index(tmp_path)[0])["n_written"] == 0                     # planned, nothing written: an empty record, not None
    (tmp_path / "index.worker00.json").write_text(json.dumps({"episodes": ["e0", "e2"], "failures": [], "in_progress": True, "generation_wall_s": 30.0}))
    (tmp_path / "index.worker01.json").write_text(json.dumps({"episodes": ["e1"], "failures": [], "in_progress": True, "generation_wall_s": 20.0}))
    body, age = combined_index(tmp_path)
    idx = json.loads(body)
    assert idx["n_written"] == 3 and idx["n_planned"] == 5 and idx["in_progress"] is True and not idx["complete"]
    assert {e["episode_id"] for e in idx["episodes"]} == {"e0", "e1", "e2"} and age >= 0
    (tmp_path / "index.json").write_text(json.dumps({"complete": True, "in_progress": False, "n_written": 5, "n_planned": 5, "episodes": []}))
    assert json.loads(combined_index(tmp_path)[0])["complete"] is True                  # the merged index wins once the run is done


def test_live_states_follow_the_generators_assignment_and_order():
    import random
    from physiworld_uav.experiments.generate import assign_workers
    from physiworld_uav.visualization.dataset_report import _assign_workers, live_states
    rng = random.Random(3)
    rows = []
    for g in range(12):
        n = rng.choice([1, 3])
        for i in range(n):
            rows.append({"episode_id": f"g{g}_e{i}", "group_id": f"g{g}", "n_steps": rng.choice([1000, 2000, 2400]),
                         "replay_of": f"g{g}_e0" if (n == 3 and i == 2) else None})
    assert _assign_workers(rows, 4) == assign_workers(rows, 4)                          # the copy equals the generator's rule
    st = live_states(rows, set(), set(), {0: 4, 1: 4, 2: 4, 3: 4})
    assert len(st["running"]) == 4 and len(set(st["running"]) | set(st["queued"])) == len(rows)
    assert all(r not in st["queued"] for r in st["running"])
    done = set(st["running"])
    st2 = live_states(rows, done, set(), {0: 4, 1: 4, 2: 4})                             # worker 3 is gone: its episodes are neither running nor queued
    assert all(w["queued"] == 0 and w["current"] is None for w in st2["workers"] if w["worker"] == 3)
    assert live_states(rows, set(), set(), {})["running"] == []                          # no live worker: everything is pending


def test_resource_stats_reports_disk_and_workers_without_gpus(tmp_path):
    import json
    from physiworld_uav.visualization.dataset_report import resource_stats
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps({"episode_id": f"e{i}", "experiment_id": "x"}) for i in range(20)) + "\n")
    (tmp_path / "index.worker00.json").write_text(json.dumps({"episodes": [f"e{i}" for i in range(12)], "failures": [], "in_progress": True}))
    (tmp_path / "shard_w00_000.h5").write_bytes(b"0" * 1_200_000)
    st = resource_stats(tmp_path)
    assert st["total"] == 20 and st["written"] == 12
    assert abs(st["disk"]["shards_gb"] - 0.0012) < 1e-9 and abs(st["disk"]["per_episode_mb"] - 0.1) < 1e-9
    assert abs(st["disk"]["projected_total_gb"] * 1e9 - (0.1e6 * 20 + st["disk"]["other_mb"] * 1e6)) < 1e3 and st["disk"]["volume_free_gb"] > 0
    assert st["host"]["cores"] and st["ours"]["workers_alive"] == 0                       # no generator process for experiment "x"
    assert st["workers"][0]["worker"] == 0 and st["workers"][0]["episodes"] == 12 and st["workers"][0]["alive"] is False


def test_episode_progress_is_exact_with_a_progress_file_and_estimated_without(tmp_path):
    import json
    import os
    import time
    from physiworld_uav.visualization.dataset_report import _episode_progress
    now = time.time()
    (tmp_path / "index.worker00.json").write_text(json.dumps({"episodes": ["a", "b"]}))
    os.utime(tmp_path / "index.worker00.json", (now - 60, now - 60))                       # last episode finished 60 s ago
    est = _episode_progress(tmp_path, 0, 123, 2, 60.0, 10, now)                             # 60 episodes per hour -> typical 60 s
    assert est["ep_exact"] is False and abs(est["ep_elapsed_s"] - 60) < 1 and abs(est["ep_typical_s"] - 60) < 1e-6 and est["ep_pct"] == 99.0   # capped below 100
    (tmp_path / "progress.worker00.json").write_text(json.dumps({"episode_id": "c", "step": 1000, "n_steps": 2000, "t_start": now - 30, "updated": now - 1,
                                                                 "mean_episode_s": 80.0}))
    ex = _episode_progress(tmp_path, 0, 123, 2, 60.0, 10, now)
    assert ex["ep_exact"] is True and ex["ep_id"] == "c" and ex["ep_step"] == 100 and ex["ep_steps"] == 200
    assert abs(ex["ep_pct"] - 50.0) < 1e-9 and abs(ex["ep_elapsed_s"] - 30) < 1 and ex["ep_typical_s"] == 80.0
    (tmp_path / "progress.worker00.json").write_text(json.dumps({"episode_id": "c", "step": 1000, "n_steps": 2000, "t_start": now - 300, "updated": now - 100}))
    assert _episode_progress(tmp_path, 0, 123, 2, 60.0, 10, now)["ep_exact"] is False       # stale progress file: not trusted while the worker is alive


def test_cli_dispatch_and_help(capsys):
    import pytest
    from physiworld_uav.visualization.cli import COMMANDS, main
    main(["--help"])
    assert "generation" in capsys.readouterr().out and set(COMMANDS) == {"generation", "report", "progress", "training", "plan", "wind"}
    with pytest.raises(SystemExit):
        main(["nonsense"])


def test_training_dashboard_data_counts_jobs_and_reads_metrics(tmp_path):
    import json
    from physiworld_uav.visualization.live_training import PAGE, structured
    sweep, exp = tmp_path / "sw", tmp_path / "exp"
    sweep.mkdir(); (exp / "fx_v06_b3_s1").mkdir(parents=True)
    (sweep / "status.json").write_text(json.dumps({
        "fx_v06_b3_s1": {"state": "running", "model": "b3", "gpu": "0", "t0": 1.0},
        "fx_v06_b4_s0": {"state": "queued", "model": "b4"},
        "fx_v06_b2_s0": {"state": "done", "model": "b2", "gpu": "1", "t0": 1.0, "wall_s": 600.0}}))
    row = {"epoch": 3, "horizon": 5, "epoch_s": 8.0, "windows_per_s": 3000.0, "metric": 0.05, "best_global": 0.04, "train": {"roll": 0.1, "total": 1.0},
           "val": {"pos_rmse_m": {"1": 0.001}}}
    (exp / "fx_v06_b3_s1" / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    d, _ = structured(sweep, exp, {})
    assert d["counts"] == {"running": 1, "queued": 1, "done": 1, "failed": 0} and d["total"] == 3 and d["eta_s"] is not None and d["eta_s"] > 0
    j = {x["name"]: x for x in d["jobs"]}
    assert j["fx_v06_b3_s1"]["epoch"] == 1 and j["fx_v06_b3_s1"]["val"] == 0.05 and j["fx_v06_b3_s1"]["best"] == 0.04 and j["fx_v06_b4_s0"]["epoch"] == 0
    assert "id=\"view-dash\"" in PAGE and "max-aspect-ratio" in PAGE and "id=\"view-curves\"" in PAGE         # dashboard, phone layout and the curves tab are in the page


def test_access_list_allows_campus_and_tailscale_only():
    from physiworld_uav.visualization.access import allowed, parse_networks
    for ok in ("127.0.0.1", "::1", "10.24.36.33", "10.1.2.3", "10.255.255.254", "100.115.191.34", "::ffff:10.24.240.21"):
        assert allowed(ok), ok
    for bad in ("8.8.8.8", "192.168.1.5", "172.16.0.9", "100.63.0.1", "11.0.0.1", "not-an-ip", ""):
        assert not allowed(bad), bad
    assert allowed("192.168.1.5", parse_networks(["192.168.1.0/24"])) and not allowed("10.1.1.1", ["192.168.1.0/24"])


def test_report_server_answers_403_to_clients_outside_the_allowed_networks(tmp_path):
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer
    from physiworld_uav.visualization.dataset_report import make_handler
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"episode_id": "e0", "sim": {"control_decimation": 4}}) + "\n")
    (tmp_path / "provenance.json").write_text("{}")
    for allow, expect in ((None, 200), (["10.0.0.0/8"], 403)):                      # the test client is 127.0.0.1: allowed by default, not by 10.0.0.0/8
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(tmp_path, allow))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            code = urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/index.json", timeout=5).status
        except urllib.error.HTTPError as e:
            code = e.code
        finally:
            srv.shutdown(); srv.server_close()
        assert code == expect, (allow, code)
