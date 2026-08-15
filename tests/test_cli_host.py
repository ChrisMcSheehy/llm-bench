"""Showing the machine, and declaring what cannot be read from it."""
from __future__ import annotations

from typer.testing import CliRunner

from llmbench.cli import app

runner = CliRunner()


def test_host_show_prints_the_platform(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "h.db"))
    monkeypatch.setenv("LLMBENCH_HOST_DECLARED", str(tmp_path / "declared.json"))
    result = runner.invoke(app, ["host"])
    assert result.exit_code == 0, result.output
    assert "os" in result.output.lower()


def test_a_declared_processor_is_stored_and_shown(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "h.db"))
    monkeypatch.setenv("LLMBENCH_HOST_DECLARED", str(tmp_path / "declared.json"))
    set_result = runner.invoke(app, ["host", "--set-cpu-model", "AMD Ryzen 7 7800X3D"])
    assert set_result.exit_code == 0, set_result.output

    shown = runner.invoke(app, ["host"])
    assert "7800X3D" in shown.output


def test_a_declaration_is_marked_as_declared_rather_than_observed(tmp_path, monkeypatch):
    """A claim about hardware is not a reading of it, and the record must say which."""
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "h.db"))
    monkeypatch.setenv("LLMBENCH_HOST_DECLARED", str(tmp_path / "declared.json"))
    runner.invoke(app, ["host", "--set-cpu-model", "AMD Ryzen 7 7800X3D"])
    shown = runner.invoke(app, ["host"])
    assert "declared" in shown.output.lower()


def test_a_declaration_survives_a_second_unrelated_declaration(tmp_path, monkeypatch):
    """Declaring one thing must not silently discard another already recorded."""
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "h.db"))
    monkeypatch.setenv("LLMBENCH_HOST_DECLARED", str(tmp_path / "declared.json"))
    runner.invoke(app, ["host", "--set-cpu-model", "AMD Ryzen 7 7800X3D"])
    runner.invoke(app, ["host", "--set-note", "the quiet one under the desk"])
    shown = runner.invoke(app, ["host"])
    assert "7800X3D" in shown.output, "the earlier declaration was lost"


def test_the_hash_is_the_same_on_two_consecutive_calls(tmp_path, monkeypatch):
    """A machine that has not changed must not look like two machines."""
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "h.db"))
    monkeypatch.setenv("LLMBENCH_HOST_DECLARED", str(tmp_path / "declared.json"))
    a = runner.invoke(app, ["host"]).output
    b = runner.invoke(app, ["host"]).output
    assert a == b
