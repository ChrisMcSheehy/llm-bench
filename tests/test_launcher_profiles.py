"""Launch profiles: the file that says which servers llmbench is allowed to start."""
from __future__ import annotations

import pytest

from llmbench.launcher import Profile, load_profiles, profiles_path

_GOOD = """
servers:
  vulkan:
    binary: C:/builds/llama-server.exe
    model:  C:/models/qwen.gguf
    args:   ["-ngl", "99", "-c", "16384"]
  turboquant:
    binary: /usr/local/bin/llama-server
    model:  /models/qwen.gguf
    port:   8123
"""


def _write(tmp_path, text: str, monkeypatch):
    p = tmp_path / "servers.yaml"
    p.write_text(text, encoding="utf-8")
    monkeypatch.setenv("LLMBENCH_SERVERS", str(p))
    return p


def test_profiles_are_keyed_by_name(tmp_path, monkeypatch):
    _write(tmp_path, _GOOD, monkeypatch)
    profiles = load_profiles()
    assert set(profiles) == {"vulkan", "turboquant"}
    assert isinstance(profiles["vulkan"], Profile)


def test_the_arguments_are_kept_in_order(tmp_path, monkeypatch):
    """Order matters to llama.cpp, and a set or a dict would lose it."""
    _write(tmp_path, _GOOD, monkeypatch)
    assert load_profiles()["vulkan"].args == ["-ngl", "99", "-c", "16384"]


def test_a_profile_without_arguments_gets_an_empty_list(tmp_path, monkeypatch):
    _write(tmp_path, _GOOD, monkeypatch)
    assert load_profiles()["turboquant"].args == []


def test_an_explicit_port_is_kept_and_absence_means_choose_one(tmp_path, monkeypatch):
    _write(tmp_path, _GOOD, monkeypatch)
    profiles = load_profiles()
    assert profiles["turboquant"].port == 8123
    assert profiles["vulkan"].port is None


def test_a_missing_file_is_no_profiles_rather_than_an_error(tmp_path, monkeypatch):
    """Most users have never created one. That is not a fault condition."""
    monkeypatch.setenv("LLMBENCH_SERVERS", str(tmp_path / "absent.yaml"))
    assert load_profiles() == {}


@pytest.mark.parametrize("missing", ["binary", "model"])
def test_a_profile_missing_a_required_field_names_itself(tmp_path, monkeypatch, missing):
    """The error has to say which profile is wrong, or the user must bisect the file."""
    fields = {"binary": "b", "model": "m"}
    del fields[missing]
    body = "\n".join(f"    {k}: {v}" for k, v in fields.items())
    _write(tmp_path, f"servers:\n  broken:\n{body}\n", monkeypatch)
    with pytest.raises(ValueError, match="broken"):
        load_profiles()


def test_arguments_given_as_a_string_are_refused(tmp_path, monkeypatch):
    """Splitting a string on spaces would silently mangle a quoted path."""
    _write(tmp_path, 'servers:\n  s:\n    binary: b\n    model: m\n    args: "-ngl 99"\n',
           monkeypatch)
    with pytest.raises(ValueError, match="args"):
        load_profiles()


def test_the_default_path_sits_beside_the_results_database(monkeypatch):
    monkeypatch.delenv("LLMBENCH_SERVERS", raising=False)
    assert profiles_path().name == "servers.yaml"
    assert profiles_path().parent.name == ".llmbench"
