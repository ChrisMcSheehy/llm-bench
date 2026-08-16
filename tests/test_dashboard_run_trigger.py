"""The dashboard may name a suite to run. It may never supply one.

Design B6, restating decision L1 for a second verb and for the same reason. L1 keeps the
browser from choosing which binary runs: the launch-profiles file is an allowlist and the
dashboard may ask for a profile *by name*. A run trigger that accepted a suite **body**
would hand all of that straight back, because a suite names targets and a target is an
address and an argument list.

So the whole of what the browser may post is two names, and these are the tests that say
so. `test_dashboard_servers.py` asserts the same property for binaries.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llmbench.dashboard import app as dashboard
from llmbench.dashboard.app import app
from llmbench.resources import available_suites, resolve_suite


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "r.db"))
    monkeypatch.setenv("LLMBENCH_SERVERS", str(tmp_path / "servers.yaml"))
    dashboard._RUN_STATE.update(status="idle", suite=None, server=None, error=None,
                                runs=[])
    return TestClient(app)


# ---- what may be named ------------------------------------------------------

def test_the_bundled_suite_is_offered_by_name(client):
    names = client.get("/api/suites").json()
    assert "default" in names, names


def test_a_named_suite_resolves_to_a_file_that_already_exists():
    path = resolve_suite("default")
    assert path.exists() and path.suffix == ".yaml"
    assert path in set(available_suites().values())


# ---- what may not ------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "../../../etc/passwd",
    "..\\..\\windows\\system32\\config",
    "/etc/passwd",
    "default.yaml",          # a filename is not a name; the extension is ours to add
    "default/../../secrets",
    "",
])
def test_a_path_is_not_a_name(client, name):
    """Every one of these is refused before anything touches the filesystem."""
    response = client.post("/api/run", json={"suite": name})
    assert response.status_code == 400, f"{name!r} was accepted: {response.text}"


def test_a_suite_body_cannot_be_posted(client):
    """The whole point. A suite names targets, and a target is an address and an argument
    list — so accepting a body here would be a way to run arbitrary programs on whatever
    machine is hosting the dashboard."""
    response = client.post("/api/run", json={
        "suite": "default",
        "targets": [{"engine": "llama.cpp", "base_url": "http://evil"}],
    })

    assert response.status_code == 422, (
        f"an unexpected field was tolerated rather than refused: {response.text}")


def test_an_unexpected_field_is_refused_rather_than_ignored(client):
    """Silently dropping a field is how a request that meant to smuggle something gets a
    success back and nobody finds out what was ignored."""
    response = client.post("/api/run", json={"suite": "default", "binary": "/bin/sh"})
    assert response.status_code == 422, response.text


def test_a_suite_that_does_not_exist_is_refused_and_says_what_does(client):
    response = client.post("/api/run", json={"suite": "nonesuch"})
    assert response.status_code == 400
    assert "default" in response.text, "the error did not say what is available"


def test_an_unknown_launch_profile_is_refused(client):
    """The profile side of the same rule: named, never supplied."""
    response = client.post("/api/run", json={"suite": "default", "server": "nonesuch"})
    assert response.status_code == 404, response.text


# ---- one at a time -----------------------------------------------------------

def test_a_second_run_is_refused_while_one_is_in_flight(client):
    """Two servers sharing one graphics card contend for it and corrupt the speed
    figures, which is why the sweep never starts two at once either."""
    dashboard._RUN_STATE.update(status="running", suite="default")

    response = client.post("/api/run", json={"suite": "default"})

    assert response.status_code == 409, response.text


def test_the_state_endpoint_reports_what_is_happening(client):
    dashboard._RUN_STATE.update(status="running", suite="default", server="vulkan")
    state = client.get("/api/run/active").json()

    assert state["status"] == "running"
    assert state["suite"] == "default"
    assert state["server"] == "vulkan"


def test_a_failed_run_is_reported_rather_than_left_looking_busy(client):
    """Nothing awaits the background task, so an exception would vanish into the event
    loop and the dashboard would show "running" for ever."""
    import asyncio

    from llmbench.resources import data_path

    asyncio.run(dashboard._execute(data_path("suites", "does-not-exist.yaml"), None))

    assert dashboard._RUN_STATE["status"] == "error"
    assert dashboard._RUN_STATE["error"], "the failure was swallowed"
