"""The CLI must survive a stream that cannot represent its own output.

On Windows a redirected stream defaults to a legacy code page. Rich substitutes for some
characters it cannot encode — table borders and the middle dot in a fingerprint label
both degrade to '?' — but not for all of them: printing the sweep's box-drawing rule to a
cp1252 pipe raised UnicodeEncodeError and killed the run.

That selectiveness is the trap. "It printed fine when I piped it" proves nothing about
the next character the tool learns to print, which is why the fix is at the entry point
and why these tests drive a real subprocess: the defect is in how the process configures
its own streams at start-up, which an in-process CliRunner replaces and cannot see.

Both tests below were confirmed to fail without the fix.
"""
from __future__ import annotations

import os
import subprocess
import sys

# The rule the sweep prints between builds. Not decoration for its own sake — this is
# the exact string that crashed.
_SWEEP_RULE = "[bold]\u2500\u2500 build-a \u2500\u2500[/bold]"


def _python(code: str, encoding: str, extra_env: dict | None = None):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = encoding
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-c", code],
                          capture_output=True, env=env, timeout=120)


def test_the_sweep_rule_does_not_crash_on_a_legacy_code_page():
    """The exact failure: box-drawing through the CLI's console, on a cp1252 stream."""
    result = _python(
        "from llmbench.cli import console\n"
        f"console.print({_SWEEP_RULE!r})\n",
        encoding="cp1252")
    assert result.returncode == 0, (
        "printing the sweep rule crashed:\n"
        + result.stderr.decode("utf-8", "replace"))


def test_a_full_sweep_runs_with_output_redirected_to_a_legacy_code_page(tmp_path):
    """End to end, because the rule is only one of the strings a run prints."""
    from tests.test_launcher import _FAKE_SERVER

    (tmp_path / "fake_ok.py").write_text(_FAKE_SERVER, encoding="utf-8")
    (tmp_path / "servers.yaml").write_text(
        "servers:\n"
        "  build-a:\n"
        f"    binary: {sys.executable}\n"
        "    model: fake_ok\n"
        '    args: ["-ngl", "99"]\n',
        encoding="utf-8")
    (tmp_path / "suite.yaml").write_text(
        "name: encoding-test\nevaluators: {}\n", encoding="utf-8")

    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "cp1252",
        "PYTHONPATH": str(tmp_path),
        "LLMBENCH_SERVERS": str(tmp_path / "servers.yaml"),
        "LLMBENCH_DB": str(tmp_path / "results.db"),
    })
    result = subprocess.run(
        [sys.executable, "-m", "llmbench.cli", "run", str(tmp_path / "suite.yaml"),
         "--server", "build-a"],
        capture_output=True, env=env, timeout=300)

    assert result.returncode == 0, (
        "a sweep crashed with its output redirected:\n"
        + result.stderr.decode("utf-8", "replace")[-2000:])
    assert b"Completed 1 run" in result.stdout
