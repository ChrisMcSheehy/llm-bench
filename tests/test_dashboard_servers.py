"""The dashboard's launch controls, including what they must refuse to do."""
from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

from llmbench import launcher
from llmbench.dashboard.app import app
from tests.test_launcher import _FAKE_SERVER

client = TestClient(app)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "fake_ok.py").write_text(_FAKE_SERVER, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    profiles = tmp_path / "servers.yaml"
    profiles.write_text(
        "servers:\n"
        "  demo:\n"
        f"    binary: {sys.executable}\n"
        "    model: fake_ok\n"
        '    args: ["-ngl", "99"]\n',
        encoding="utf-8")
    monkeypatch.setenv("LLMBENCH_SERVERS", str(profiles))
    yield tmp_path
    launcher.stop_by_name("demo")


def test_the_profiles_are_listed_with_their_arguments(workspace):
    body = client.get("/api/servers").json()
    assert [p["name"] for p in body] == ["demo"]
    assert body[0]["args"] == ["-ngl", "99"]
    assert body[0]["running"] is False


def test_a_profile_can_be_started_and_stopped(workspace):
    started = client.post("/api/servers/demo/start")
    assert started.status_code == 200, started.text
    base_url = started.json()["base_url"]
    assert launcher.is_listening(int(base_url.rsplit(":", 1)[1]))

    listed = client.get("/api/servers").json()[0]
    assert listed["running"] is True

    stopped = client.post("/api/servers/demo/stop")
    assert stopped.status_code == 200
    assert not launcher.is_listening(int(base_url.rsplit(":", 1)[1])), "still up"


def test_starting_an_unknown_profile_is_refused_and_starts_nothing(workspace):
    """The security boundary: the browser may name a profile, never describe one.

    Asserts that nothing was launched, not merely that the status code was 404 — a 404
    with a process running behind it would be the bug this guards.
    """
    before = dict(launcher.running())
    response = client.post("/api/servers/not-a-profile/start")
    assert response.status_code == 404
    assert launcher.running() == before, "something was started anyway"


def test_the_api_offers_no_way_to_supply_a_binary(workspace):
    """A posted binary and arguments must not be honoured under any key name."""
    before = dict(launcher.running())
    response = client.post(
        "/api/servers/evil/start",
        json={"binary": sys.executable, "model": "fake_ok", "args": []})
    assert response.status_code == 404
    assert launcher.running() == before


def test_starting_a_running_server_twice_does_not_start_a_second_one(workspace):
    first = client.post("/api/servers/demo/start").json()
    second = client.post("/api/servers/demo/start").json()
    assert first["base_url"] == second["base_url"]


def test_a_failing_profile_returns_the_servers_own_output(tmp_path, monkeypatch):
    profiles = tmp_path / "servers.yaml"
    profiles.write_text(
        f"servers:\n  broken:\n    binary: {tmp_path / 'nope.exe'}\n    model: m\n",
        encoding="utf-8")
    monkeypatch.setenv("LLMBENCH_SERVERS", str(profiles))
    response = client.post("/api/servers/broken/start")
    assert response.status_code == 500
    assert "could not run" in response.json()["detail"]


def test_the_page_is_served(workspace):
    assert client.get("/servers").status_code == 200


# --- the machine behind each figure (Phase 4, D1) -----------------------------------

def test_the_pooled_endpoint_separates_quality_from_speed(tmp_path, monkeypatch):
    """Quality pools across machines; speed is reported per machine."""
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "pooled.db"))
    from datetime import datetime, timezone

    from llmbench.models import HostFingerprint, Metric, ModelFingerprint, RunResult
    from llmbench.store import Store

    store = Store(str(tmp_path / "pooled.db"))
    fp = ModelFingerprint(engine="llama.cpp", base_url="u", model_id="m")
    for run_id, name, speed in (("r1", "RTX 4090", 120.0), ("r2", "RX 6600", 30.0)):
        host = HostFingerprint(
            os="Linux", arch="x86_64", cpu_count=8,
            devices=[{"id": "D0", "backend": "CUDA", "name": name,
                      "total_mib": 8192, "free_mib": 8000}])
        store.start_run(
            RunResult(run_id=run_id, fingerprint=fp, suite="t",
                      started_at=datetime.now(timezone.utc)),
            host_hash=store.upsert_host(host))
        store.add_metrics(run_id, [
            Metric(evaluator="speed", name="decode_tps", value=speed),
            Metric(evaluator="needle", name="score_mean", value=0.9)])
    store.close()

    body = TestClient(app).get("/api/pooled").json()
    quality = [q for q in body["quality"] if q["name"] == "score_mean"]
    speed = [s for s in body["speed"] if s["name"] == "decode_tps"]
    assert len(quality) == 1, "quality should pool to one row per configuration"
    assert len(speed) == 2, "speed was pooled across two machines"
    assert sorted(round(s["value"]) for s in speed) == [30, 120]


def test_a_run_carries_the_machine_it_ran_on(tmp_path, monkeypatch):
    """So nobody compares two speed figures without seeing they are two machines."""
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "runs.db"))
    from datetime import datetime, timezone

    from llmbench.models import HostFingerprint, ModelFingerprint, RunResult
    from llmbench.store import Store

    store = Store(str(tmp_path / "runs.db"))
    host = HostFingerprint(os="Linux", arch="x86_64", cpu_count=8)
    store.start_run(
        RunResult(run_id="r1",
                  fingerprint=ModelFingerprint(engine="llama.cpp", base_url="u",
                                               model_id="m"),
                  suite="t", started_at=datetime.now(timezone.utc)),
        host_hash=store.upsert_host(host))
    store.close()

    rows = TestClient(app).get("/api/runs").json()
    assert rows[0]["host_hash"], "the run reports no machine"
    assert "Linux" in rows[0]["host_label"]
