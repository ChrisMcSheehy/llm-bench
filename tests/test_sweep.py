"""Running one suite across several builds in turn.

The point of the sweep: testing a llama.cpp pull request means running the same suite
against several binaries and comparing. Each build reports its own commit, so the runs
file as separate configurations rather than averaging together.

The suite used here configures no evaluators, so each run is detect-and-record. That is
deliberate: this file is about the sweep's own logic — launch, detect, store, stop, next
— and running real evaluators would need a real model and prove nothing extra about it.
"""
from __future__ import annotations

import sqlite3
import sys

import pytest
from typer.testing import CliRunner

from llmbench import launcher
from llmbench.cli import app
from tests.test_launcher import _FAKE_SERVER

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Two profiles differing only in -ngl, plus an empty suite and a private database."""
    (tmp_path / "fake_ok.py").write_text(_FAKE_SERVER, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    profiles = tmp_path / "servers.yaml"
    profiles.write_text(
        "servers:\n"
        "  build-a:\n"
        f"    binary: {sys.executable}\n"
        "    model: fake_ok\n"
        '    args: ["-ngl", "0"]\n'
        "  build-b:\n"
        f"    binary: {sys.executable}\n"
        "    model: fake_ok\n"
        '    args: ["-ngl", "99"]\n',
        encoding="utf-8")
    monkeypatch.setenv("LLMBENCH_SERVERS", str(profiles))

    db = tmp_path / "results.db"
    monkeypatch.setenv("LLMBENCH_DB", str(db))

    suite = tmp_path / "suite.yaml"
    suite.write_text("name: sweep-test\nevaluators: {}\n", encoding="utf-8")
    return {"suite": str(suite), "db": db, "tmp": tmp_path}


def _fingerprints(db):
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT hash, n_gpu_layers, launch_settings_observed, label FROM fingerprint"
    ).fetchall()
    conn.close()
    return rows


def test_a_sweep_files_each_build_separately(workspace):
    """Design criterion 3, from the database side."""
    result = runner.invoke(app, ["run", workspace["suite"],
                                 "--server", "build-a", "--server", "build-b"])
    assert result.exit_code == 0, result.output
    assert "Completed 2 run(s)" in result.output

    rows = _fingerprints(workspace["db"])
    assert len(rows) == 2, f"expected two configurations, got {len(rows)}"
    assert {r["n_gpu_layers"] for r in rows} == {"0", "99"}
    assert all(r["launch_settings_observed"] == 1 for r in rows), \
        "a launched server should never be filed as having unobserved settings"
    assert len({r["hash"] for r in rows}) == 2


def test_a_sweep_leaves_no_server_running(workspace):
    runner.invoke(app, ["run", workspace["suite"],
                        "--server", "build-a", "--server", "build-b"])
    for name in ("build-a", "build-b"):
        record = launcher.running().get(name)
        assert not record, f"{name} was left recorded as running"


def test_an_unknown_profile_fails_before_anything_is_started(workspace):
    """A typo in the last name must not cost a full suite run against the first."""
    result = runner.invoke(app, ["run", workspace["suite"],
                                 "--server", "build-a", "--server", "typo"])
    assert result.exit_code != 0
    assert "No such profile" in result.output
    assert not workspace["db"].exists() or not _fingerprints(workspace["db"]), \
        "a run happened despite the bad profile name"


def test_a_build_that_will_not_start_does_not_abandon_the_rest(workspace):
    """The normal case when the pull request under test is broken."""
    profiles = workspace["tmp"] / "servers.yaml"
    profiles.write_text(
        "servers:\n"
        "  broken:\n"
        f"    binary: {workspace['tmp'] / 'nope.exe'}\n"
        "    model: fake_ok\n"
        "  build-b:\n"
        f"    binary: {sys.executable}\n"
        "    model: fake_ok\n"
        '    args: ["-ngl", "99"]\n',
        encoding="utf-8")

    result = runner.invoke(app, ["run", workspace["suite"],
                                 "--server", "broken", "--server", "build-b"])
    assert result.exit_code == 0, result.output
    assert "did not start" in result.output
    assert "Completed 1 run(s)" in result.output, "the healthy build was skipped"
    assert len(_fingerprints(workspace["db"])) == 1


def test_running_without_server_still_needs_targets_in_the_suite(workspace):
    """The ordinary path is unchanged: a suite with no targets is still an error."""
    result = runner.invoke(app, ["run", workspace["suite"]])
    assert result.exit_code != 0
