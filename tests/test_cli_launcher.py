"""The launcher's command-line surface, driven end to end against a fake server."""
from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from llmbench import launcher
from llmbench.cli import app
from tests.test_launcher import _FAKE_SERVER

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A profiles file naming a fake server, plus an isolated place to record state."""
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
    return tmp_path


def test_servers_lists_the_profiles(workspace):
    result = runner.invoke(app, ["servers"])
    assert result.exit_code == 0, result.output
    assert "demo" in result.output
    assert "-ngl 99" in result.output


def test_servers_explains_itself_when_there_are_none(tmp_path, monkeypatch):
    """A first-time user must be told where to put the file, not shown an empty table."""
    monkeypatch.setenv("LLMBENCH_SERVERS", str(tmp_path / "none.yaml"))
    result = runner.invoke(app, ["servers"])
    assert result.exit_code == 0
    # rich wraps a long path across lines, splitting it mid-token, so the newlines are
    # removed before looking for it.
    flattened = result.output.replace("\n", "")
    assert "none.yaml" in flattened, "the user is not told where the file goes"
    assert "binary:" in result.output, "no example was offered"


def test_launch_then_stop(workspace):
    launch = runner.invoke(app, ["launch", "demo"])
    assert launch.exit_code == 0, launch.output
    assert "listening on" in launch.output

    record = launcher.running().get("demo")
    assert record, "the running server was not recorded"
    assert launcher.is_listening(int(record["port"]))

    stopped = runner.invoke(app, ["stop", "demo"])
    assert stopped.exit_code == 0, stopped.output
    assert not launcher.is_listening(int(record["port"])), "the server is still up"
    assert "demo" not in launcher.running(), "the record outlived the server"


def test_launching_an_unknown_profile_fails_and_starts_nothing(workspace):
    result = runner.invoke(app, ["launch", "not-a-profile"])
    assert result.exit_code != 0
    assert "No profile" in result.output
    assert launcher.running() == {}


def test_stopping_something_that_is_not_running_is_not_an_error(workspace):
    result = runner.invoke(app, ["stop", "demo"])
    assert result.exit_code == 0
    assert "not running" in result.output


def test_a_profile_whose_binary_is_missing_reports_why(tmp_path, monkeypatch):
    profiles = tmp_path / "servers.yaml"
    profiles.write_text(
        f"servers:\n  broken:\n    binary: {tmp_path / 'nope.exe'}\n    model: m\n",
        encoding="utf-8")
    monkeypatch.setenv("LLMBENCH_SERVERS", str(profiles))
    result = runner.invoke(app, ["launch", "broken"])
    assert result.exit_code != 0
    assert "could not run" in result.output
