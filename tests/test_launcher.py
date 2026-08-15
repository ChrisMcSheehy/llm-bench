"""Starting and stopping a real child process.

These tests launch a fake server — a short Python module that answers /props the way
llama.cpp does — rather than llama-server itself. The bench runs on three operating
systems in CI on machines with no graphics card and no multi-gigabyte model, so a test
needing the real binary could not run there.

The fake is launched through exactly the same code path and the same argument shape as
the real thing: `<binary> -m <model> --host 127.0.0.1 --port N`, which for Python means
`python -m <module>`. It is a real spawn, a real port, a real readiness poll and a real
termination. What it cannot cover is llama.cpp's own behaviour, which is checked by hand
once (see the plan's L6).
"""
from __future__ import annotations

import subprocess
import sys
import urllib.request

import pytest

from llmbench import launcher
from llmbench.launcher import (
    LaunchError, Profile, RunningServer, free_port, launched, start, stop,
)

# Answers /props on the port it is given, like llama-server does.
_FAKE_SERVER = '''
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"build_info": "b9999-facefeed", "total_slots": 1}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass

HTTPServer(("127.0.0.1", port), H).serve_forever()
'''

# Exits at once, having complained — like a model that will not load.
_FAKE_FAILURE = '''
import sys
print("error: unable to load model 'nonsense.gguf'")
print("GGML_ASSERT(ggml_can_mul_mat(a, b)) failed")
sys.exit(1)
'''

# Starts, never listens. Stands in for a server wedged during load.
_FAKE_HANG = '''
import time
time.sleep(600)
'''


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    """Write the fake servers as importable modules and put them on the import path.

    The launcher passes the environment through unmodified, so setting PYTHONPATH here
    reaches the child.
    """
    for name, source in (("fake_ok", _FAKE_SERVER), ("fake_dead", _FAKE_FAILURE),
                         ("fake_hang", _FAKE_HANG)):
        (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return tmp_path


@pytest.fixture
def spawned(monkeypatch):
    """Record every process the launcher creates, so cleanup can be asserted.

    This wraps the real Popen rather than replacing it — the child is a genuine process.
    Without this, a test for "the timeout path leaves nothing running" has no handle to
    check and would have to assert something that cannot fail.
    """
    created: list[subprocess.Popen] = []
    real = subprocess.Popen

    def recording(*args, **kwargs):
        p = real(*args, **kwargs)
        created.append(p)
        return p

    monkeypatch.setattr(launcher.subprocess, "Popen", recording)
    return created


@pytest.fixture
def logs(tmp_path):
    return str(tmp_path / "logs")


def _profile(name: str, module: str, port=None) -> Profile:
    return Profile(name=name, binary=sys.executable, model=module, args=[], port=port)


def _alive(server: RunningServer) -> bool:
    return server.process.poll() is None


def test_start_returns_only_once_the_server_answers(fakes, logs):
    """A process that exists is not a server that works."""
    server = start(_profile("ok", "fake_ok"), log_dir=logs, timeout=60)
    try:
        with urllib.request.urlopen(f"{server.base_url}/props", timeout=5) as r:
            assert r.status == 200
    finally:
        stop(server)


def test_a_server_that_dies_reports_its_own_output(fakes, logs):
    """The whole reason for capturing the log: the useful text is the child's, not ours."""
    with pytest.raises(LaunchError) as excinfo:
        start(_profile("dead", "fake_dead"), log_dir=logs, timeout=60)
    message = str(excinfo.value)
    assert "unable to load model" in message, f"the child's output was dropped: {message}"
    assert "GGML_ASSERT" in message


def test_a_server_that_never_answers_times_out_and_is_cleaned_up(fakes, logs, spawned):
    """The timeout path must not leak the process it gave up on."""
    with pytest.raises(LaunchError, match="no answer"):
        start(_profile("hang", "fake_hang"), log_dir=logs, timeout=2)

    assert spawned, "no process was started, so this test proved nothing"
    for process in spawned:
        assert process.poll() is not None, "a server survived the failed launch"


def test_stop_actually_ends_the_process(fakes, logs):
    """Asserts the process is gone, not that stop() returned."""
    server = start(_profile("stopme", "fake_ok"), log_dir=logs, timeout=60)
    assert _alive(server)
    stop(server)
    assert not _alive(server), "the process survived stop()"


def test_stopping_twice_is_harmless(fakes, logs):
    server = start(_profile("twice", "fake_ok"), log_dir=logs, timeout=60)
    stop(server)
    stop(server)
    assert not _alive(server)


def test_the_context_manager_stops_the_server_when_the_body_raises(fakes, logs):
    """A failing run must not leak a server holding graphics memory."""
    holder = {}
    with pytest.raises(ZeroDivisionError):
        with launched(_profile("ctx", "fake_ok"), log_dir=logs, timeout=60) as server:
            holder["server"] = server
            assert _alive(server)
            1 / 0
    assert not _alive(holder["server"]), "the server outlived the failed body"


def test_two_servers_started_together_get_different_ports(fakes, logs):
    a = start(_profile("a", "fake_ok"), log_dir=logs, timeout=60)
    try:
        b = start(_profile("b", "fake_ok"), log_dir=logs, timeout=60)
        try:
            assert a.port != b.port
        finally:
            stop(b)
    finally:
        stop(a)


def test_an_explicit_port_is_honoured(fakes, logs):
    port = free_port()
    server = start(_profile("fixed", "fake_ok", port=port), log_dir=logs, timeout=60)
    try:
        assert server.port == port
        assert server.base_url.endswith(str(port))
    finally:
        stop(server)


def test_a_binary_that_does_not_exist_is_a_launch_error(tmp_path, logs):
    profile = Profile(name="ghost", binary=str(tmp_path / "nope.exe"), model="m")
    with pytest.raises(LaunchError, match="could not run"):
        start(profile, log_dir=logs, timeout=5)


def test_the_launch_arguments_are_reported_back(fakes, logs):
    """What the bench supplied is what completes the identity later."""
    profile = _profile("args", "fake_ok")
    profile.args = ["-ngl", "99"]
    server = start(profile, log_dir=logs, timeout=60)
    try:
        assert server.launch_args == ["-ngl", "99"]
    finally:
        stop(server)


def test_the_argument_order_puts_the_profile_last(fakes):
    """The profile's own arguments come after ours, so a user can override host or port."""
    argv = launcher.build_argv(_profile("o", "fake_ok"), 1234)
    profile_args = ["-ngl", "99"]
    argv_with = launcher.build_argv(
        Profile(name="o", binary="b", model="m", args=profile_args), 1234)
    assert argv[:3] == [sys.executable, "-m", "fake_ok"]
    assert argv_with[-2:] == profile_args
