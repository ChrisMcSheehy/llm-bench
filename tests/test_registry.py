"""Discovery must find every test module, whatever was imported before it."""
from __future__ import annotations

import os
import subprocess
import sys


def test_importing_one_evaluator_does_not_hide_the_others():
    """A partially-populated registry must not read as a complete one.

    `get()` and `available()` used to treat a non-empty registry as proof that
    discovery had already run. Any code importing a single evaluator module first
    — a plugin, a test, the dashboard — therefore left the tool convinced that
    module was the only test in existence, and a suite naming any other failed
    with "no evaluator named ...".

    This runs in a fresh interpreter because import caching makes the state it
    checks impossible to recreate inside an already-loaded process.
    """
    probe = (
        "import llmbench.evaluators.coding\n"          # the partial import
        "from llmbench.registry import available\n"
        "print(' '.join(available()))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe],
                            capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    found = result.stdout.split()
    assert "coding" in found
    assert "mcqa" in found, f"discovery was suppressed; only found: {found}"


# ---- a test module published as a separate package (design E4) ---------------
#
# The difference between a tool and something other people can build on is whether
# extending it means editing it. These build a *real* distribution - a module plus the
# .dist-info that declares its entry point - and put it on the path of a fresh
# interpreter. The mechanism under test is importlib.metadata's own discovery, and a
# patched `entry_points()` would sit on both sides of that seam without touching it
# (LESSONS.md, unit-tests-either-side-of-a-seam-do-not-test-the-seam).

_PLUGIN = '''
from llmbench.evaluators.base import Evaluator
from llmbench.registry import register


@register
class DemoEval(Evaluator):
    name = "{name}"
    version = "1"

    async def evaluate(self, ctx):
        return []
'''


def _installed_plugin(tmp_path, source: str) -> subprocess.CompletedProcess:
    """Lay out a distribution in tmp_path, then ask a fresh interpreter what it finds."""
    (tmp_path / "llmbench_demo.py").write_text(source, encoding="utf-8")
    dist = tmp_path / "llmbench_demo-0.1.dist-info"
    dist.mkdir()
    dist.joinpath("METADATA").write_text(
        "Metadata-Version: 2.1\nName: llmbench-demo\nVersion: 0.1\n", encoding="utf-8")
    dist.joinpath("entry_points.txt").write_text(
        "[llmbench.evaluators]\ndemo = llmbench_demo\n", encoding="utf-8")

    return subprocess.run(
        [sys.executable, "-c",
         "from llmbench.registry import available; print(' '.join(available()))"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(tmp_path)})


def test_an_installed_package_can_add_a_test_module(tmp_path):
    result = _installed_plugin(tmp_path, _PLUGIN.format(name="demo"))

    assert result.returncode == 0, result.stderr
    found = result.stdout.split()
    assert "demo" in found, f"the installed test module was not discovered: {found}"
    assert "needle" in found, "loading a plugin lost the built-in modules"


def test_a_plugin_that_cannot_load_names_itself_and_stops_the_run(tmp_path):
    """Skipping it quietly would be worse. A test module that is silently absent looks
    exactly like one that was never installed, and the suite then runs fewer tests than
    it was asked for without saying so."""
    result = _installed_plugin(tmp_path, "raise RuntimeError('the plugin is broken')")

    assert result.returncode != 0, "a broken plugin was skipped rather than reported"
    assert "demo" in result.stderr, f"the culprit was not named: {result.stderr}"
    assert "the plugin is broken" in result.stderr, result.stderr


def test_a_plugin_cannot_quietly_take_a_built_in_name(tmp_path):
    """Two modules answering to `needle` would mean the suite silently ran the other one."""
    result = _installed_plugin(tmp_path, _PLUGIN.format(name="needle"))

    assert result.returncode != 0, "a plugin overwrote a built-in test module"
    assert "clash" in result.stderr, result.stderr
