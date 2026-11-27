"""Phone notifications (ntfy): configuration, the request that is sent, silence when not set up, no exceptions."""
from __future__ import annotations

import urllib.error

from physiworld_uav import notify as N


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _clear(monkeypatch, tmp_path):
    for k in ("NTFY_TOPIC", "NTFY_SERVER", "NTFY_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(N, "_conf_file", lambda: tmp_path / "ntfy.conf")


def test_without_a_topic_nothing_is_sent(monkeypatch, tmp_path):
    _clear(monkeypatch, tmp_path)
    called = []
    monkeypatch.setattr(N.urllib.request, "urlopen", lambda *a, **k: called.append(a) or _Resp())
    assert N.config()["topic"] == "" and N.notify("x") is False and not called and N.main(["--test"]) == 2


def test_environment_topic_builds_the_request(monkeypatch, tmp_path):
    _clear(monkeypatch, tmp_path)
    monkeypatch.setenv("NTFY_TOPIC", "abc123"); monkeypatch.setenv("NTFY_SERVER", "https://ntfy.example/")
    sent = {}

    def fake(req, timeout=None):
        sent.update(url=req.full_url, body=req.data, headers={k.lower(): v for k, v in req.header_items()}, method=req.get_method())
        return _Resp()
    monkeypatch.setattr(N.urllib.request, "urlopen", fake)
    assert N.notify("Generation finished: 10/10 episodes", title="Data generation ✓", priority="high", tags="white_check_mark") is True
    assert sent["url"] == "https://ntfy.example/abc123" and sent["method"] == "POST" and sent["body"] == b"Generation finished: 10/10 episodes"
    assert sent["headers"]["priority"] == "high" and sent["headers"]["tags"] == "white_check_mark" and sent["headers"]["title"].isascii()


def test_file_configuration_and_default_server(monkeypatch, tmp_path):
    _clear(monkeypatch, tmp_path)
    (tmp_path / "ntfy.conf").write_text("# comment\ntopic=fromfile\ntoken=tk_1\n")
    c = N.config()
    assert c == {"topic": "fromfile", "server": "https://ntfy.sh", "token": "tk_1"}
    monkeypatch.setenv("NTFY_TOPIC", "env_wins")
    assert N.config()["topic"] == "env_wins"


def test_errors_never_raise(monkeypatch, tmp_path):
    _clear(monkeypatch, tmp_path)
    monkeypatch.setenv("NTFY_TOPIC", "t")
    def boom(*a, **k):
        raise urllib.error.URLError("no network")
    monkeypatch.setattr(N.urllib.request, "urlopen", boom)
    assert N.notify("x") is False
    monkeypatch.setattr(N.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("anything")))
    assert N.notify("x") is False
