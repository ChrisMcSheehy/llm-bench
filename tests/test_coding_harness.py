"""The coding harness must run correct code correctly, and stop what it times out.

Both tests drive `_run_tests` directly. That method is the whole harness — write
the code, run it under pytest in a subprocess, enforce the timeout — and going
through `evaluate` instead would need a fake model server while proving nothing
extra about the part under test.
"""
from __future__ import annotations

import asyncio
import time

from llmbench.evaluators.coding import CodingEvaluator
from llmbench.resources import data_path

# Launched by the fake solution below, and deliberately outlives its parent unless
# something kills the whole tree. It touches `heartbeat` before `started` so that
# `started` existing always implies `heartbeat` does too.
_WATCHER = """\
import pathlib, sys, time

heartbeat, started = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
heartbeat.write_text(str(time.time()), encoding="utf-8")
started.write_text("yes", encoding="utf-8")
while True:
    time.sleep(0.1)
    heartbeat.write_text(str(time.time()), encoding="utf-8")
"""

# Stands in for a model's answer. Blocking at import time is what hangs pytest
# inside the harness and triggers the timeout.
_SOLUTION = """\
import subprocess, sys, time

subprocess.Popen([sys.executable, {watcher!r}, {heartbeat!r}, {started!r}])
time.sleep(600)
"""

_HARNESS_TIMEOUT_S = 2  # pytest itself boots in ~0.5s, so the watcher gets well clear


def test_the_reference_solution_passes():
    """The harness runs correct code correctly.

    Without this, a harness broken badly enough to run nothing at all still looks
    like an ordinary run of failing problems. That is exactly what happened: a
    scrubbed environment missing SYSTEMROOT killed the subprocess during pytest
    startup on Windows, and every coding problem silently scored zero.
    """
    problem_dir = data_path("problems", "coding", "two_sum")
    solution = (problem_dir / "solution.py").read_text(encoding="utf-8")

    passed, detail = asyncio.run(CodingEvaluator()._run_tests(
        solution, {"_dir": problem_dir}, 30))

    assert passed, f"the project's own reference solution did not pass: {detail}"


def test_non_ascii_model_output_does_not_break_the_harness():
    """A model may emit any character; writing its answer must not depend on locale.

    With no explicit encoding, Python writes using the platform's locale — cp1252
    on a default Windows install, which cannot represent most of Unicode. The
    harness would then crash on the model's answer rather than grading it.
    """
    problem_dir = data_path("problems", "coding", "two_sum")
    reference = (problem_dir / "solution.py").read_text(encoding="utf-8")
    solution = "# £ 日本語 — models emit whatever they like\n" + reference

    passed, detail = asyncio.run(CodingEvaluator()._run_tests(
        solution, {"_dir": problem_dir}, 30))

    assert passed, f"non-ASCII model output broke the harness: {detail}"


def test_timeout_kills_the_whole_process_tree(tmp_path):
    """A timed-out run takes everything it spawned down with it.

    The harness executes model-generated code. If the timeout kills only the direct
    child, anything that child started keeps running unsupervised and untimed.
    """
    heartbeat = tmp_path / "heartbeat.txt"
    started = tmp_path / "started.txt"
    watcher = tmp_path / "watcher.py"
    watcher.write_text(_WATCHER, encoding="utf-8")

    problem_dir = tmp_path / "problem"
    problem_dir.mkdir()
    (problem_dir / "tests.py").write_text("import solution\n", encoding="utf-8")

    solution = _SOLUTION.format(watcher=str(watcher), heartbeat=str(heartbeat),
                                started=str(started))

    outcome = asyncio.run(CodingEvaluator()._run_tests(
        solution, {"_dir": problem_dir}, _HARNESS_TIMEOUT_S))

    assert outcome == (False, "timeout")
    assert started.exists(), (
        "the background process never started, so this test never exercised the "
        "kill path — raise _HARNESS_TIMEOUT_S")

    last_seen = heartbeat.read_text(encoding="utf-8")
    time.sleep(1.0)
    assert heartbeat.read_text(encoding="utf-8") == last_seen, (
        "the background process outlived the timeout that killed its parent")
