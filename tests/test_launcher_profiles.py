"""Launch profiles: the file that says which servers llmbench is allowed to start."""
from __future__ import annotations

import pytest

from llmbench.launcher import Profile, load_profiles, profiles_path
from llmbench.targets.llamacpp import _parse_args

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


# ---- shared defaults and path variables (design B7) --------------------------
#
# The reference collection this was modelled on holds three hundred profiles in one file,
# and it is maintainable only because it has exactly these two features. Without them a
# set that size is the same eight lines copied three hundred times, and the copies drift.
#
# Both are resolved at load time, so what leaves the loader is a real path and a complete
# argument list. Nothing downstream ever sees a template - which matters most for the
# fingerprint, because a configuration must hash by what it *is*, not by how it was
# written down.

_SHARED = """
defaults:
  args: ["-fa", "on", "-ctk", "q8_0"]
  vars:
    models: C:/models
    builds: C:/builds

servers:
  inherits:
    binary: "{builds}/llama-b10441/llama-server.exe"
    model:  "{models}/Qwen3.6-35B-UD-Q4_K_M.gguf"
    args:   ["-ngl", "99"]
  states-it:
    binary: "{builds}/llama-b10441/llama-server.exe"
    model:  "{models}/Qwen3.6-35B-UD-Q4_K_M.gguf"
    args:   ["-fa", "on", "-ctk", "q8_0", "-ngl", "99"]
  bare:
    binary: /usr/local/bin/llama-server
    model:  /models/qwen.gguf
"""


def test_the_profiles_own_arguments_come_last(tmp_path, monkeypatch):
    """"The usual settings, except this" only works if the exception is applied last:
    llama.cpp lets a later flag override an earlier one."""
    _write(tmp_path, _SHARED, monkeypatch)
    assert load_profiles()["inherits"].args == ["-fa", "on", "-ctk", "q8_0", "-ngl", "99"]


def test_a_profile_with_no_arguments_of_its_own_still_gets_the_defaults(tmp_path,
                                                                       monkeypatch):
    _write(tmp_path, _SHARED, monkeypatch)
    assert load_profiles()["bare"].args == ["-fa", "on", "-ctk", "q8_0"]


def test_a_variable_is_substituted_in_paths(tmp_path, monkeypatch):
    _write(tmp_path, _SHARED, monkeypatch)
    profile = load_profiles()["inherits"]

    assert profile.model == "C:/models/Qwen3.6-35B-UD-Q4_K_M.gguf"
    assert profile.binary == "C:/builds/llama-b10441/llama-server.exe"


def test_nothing_downstream_ever_sees_a_template(tmp_path, monkeypatch):
    """Resolution happens here, once. A template reaching the launcher would be a path
    that fails at the operating system; a template reaching the fingerprint would be a
    configuration filed under how it was typed."""
    _write(tmp_path, _SHARED, monkeypatch)
    for profile in load_profiles().values():
        unresolved = [t for t in (profile.binary, profile.model, *profile.args) if "{" in t]
        assert not unresolved, unresolved


def test_inheriting_a_flag_and_stating_it_are_the_same_configuration(tmp_path,
                                                                     monkeypatch):
    """The point of resolving at load time. These two profiles are written differently and
    are the same deployment, so tidying a profile file must not fork a configuration's
    history.

    Sameness is asserted over the *settings*, not the literal argv, because that is what
    the identity is built from. A profile restating a default gets the flag twice - once
    inherited, once its own - and llama.cpp takes the last, exactly as `_parse_args` does.
    Deduplicating instead would need a table of which flags carry a value and which stand
    alone, and getting that table wrong drops a setting silently.
    """
    _write(tmp_path, _SHARED, monkeypatch)
    profiles = load_profiles()

    assert _parse_args(profiles["inherits"].args) == _parse_args(profiles["states-it"].args)


def test_a_profile_overrides_the_default_it_disagrees_with(tmp_path, monkeypatch):
    """The property the whole scheme rests on: "the usual settings, except this". The
    flag appears twice in the resolved argv and the profile's value is the one that
    counts — for the real command line and for the identity alike."""
    _write(tmp_path, """
defaults:
  args: ["-ngl", "99", "-fa", "on"]
servers:
  partial-offload:
    binary: /usr/local/bin/llama-server
    model:  /models/qwen.gguf
    args:   ["-ngl", "24"]
""", monkeypatch)

    args = load_profiles()["partial-offload"].args
    assert args == ["-ngl", "99", "-fa", "on", "-ngl", "24"]
    assert _parse_args(args)["ngl"] == "24", "the inherited default won"


def test_an_undefined_variable_is_an_error_naming_it(tmp_path, monkeypatch):
    """Left as a literal it would reach the launcher and be reported as a missing file,
    which blames the disk for a typo."""
    _write(tmp_path, """
defaults:
  vars: {models: C:/models}
servers:
  typo:
    binary: /usr/local/bin/llama-server
    model:  "{modles}/qwen.gguf"
""", monkeypatch)

    with pytest.raises(ValueError) as exc:
        load_profiles()

    assert "modles" in str(exc.value), str(exc.value)
    assert "typo" in str(exc.value), "the error did not say which profile"


def test_braces_that_are_not_variables_are_left_alone(tmp_path, monkeypatch):
    """A chat template passed as an argument is full of braces and is not a placeholder.
    Only `{identifier}` is treated as one."""
    _write(tmp_path, """
defaults:
  vars: {models: C:/models}
servers:
  templated:
    binary: /usr/local/bin/llama-server
    model:  "{models}/qwen.gguf"
    args:   ["--chat-template", "{% for m in messages %}{{ m.content }}{% endfor %}"]
""", monkeypatch)

    profile = load_profiles()["templated"]
    assert profile.model == "C:/models/qwen.gguf"
    assert profile.args[-1] == "{% for m in messages %}{{ m.content }}{% endfor %}"


def test_a_file_with_no_defaults_block_still_loads(tmp_path, monkeypatch):
    """Every profile file written before this existed has no `defaults`, and must keep
    working untouched."""
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
