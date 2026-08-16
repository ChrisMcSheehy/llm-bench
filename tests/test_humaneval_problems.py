"""The converted HumanEval problem set is complete, well-formed and actually runnable.

Design B1. The 164 problems were converted once from OpenAI's HumanEval and committed,
so nothing downloads them at run time and nothing regenerates them. That makes the
committed output the thing worth checking: a conversion defect here is silent, because a
malformed problem looks exactly like a model failing to solve it.

The structural checks cover all 164 and cost nothing. Execution covers a sample, because
running every reference solution takes about eighty seconds of subprocess launches - it
was done once at conversion time, and all 164 passed.
"""
from __future__ import annotations

import asyncio

import pytest
import yaml

from llmbench.evaluators.coding import CodingEvaluator
from llmbench.resources import data_path

_ROOT = data_path("problems", "coding")
_CONVERTED = sorted(d for d in _ROOT.iterdir()
                    if d.is_dir() and d.name.startswith("humaneval_"))
#: Includes the three whose checks call a helper from the prompt rather than only the
#: entry point - the case that broke the first conversion and produced a NameError from
#: the test rather than from the answer.
_SAMPLE = ["humaneval_000", "humaneval_032", "humaneval_038", "humaneval_050",
           "humaneval_163"]


def test_the_whole_benchmark_is_present():
    """164 is the number HumanEval publishes. A short set would quietly report a pass
    rate over fewer problems than the label claims."""
    assert len(_CONVERTED) == 164, f"found {len(_CONVERTED)} converted problems"


@pytest.mark.parametrize("directory", _CONVERTED, ids=lambda d: d.name)
def test_every_problem_is_well_formed(directory):
    """Each problem is three files, and the entry point named in the metadata is the one
    the solution defines and the tests import."""
    meta = yaml.safe_load((directory / "problem.yaml").read_text(encoding="utf-8"))
    solution = (directory / "solution.py").read_text(encoding="utf-8")
    tests = (directory / "tests.py").read_text(encoding="utf-8")

    assert meta["id"] == directory.name
    assert meta["prompt"].strip(), "the model would be asked nothing"
    assert meta["source"].startswith("HumanEval/"), meta.get("source")

    entrypoint = meta["entrypoint"]
    assert f"def {entrypoint}" in solution, f"{entrypoint} is not defined in solution.py"
    assert f"from solution import {entrypoint}" in tests
    assert f"check({entrypoint})" in tests


@pytest.mark.parametrize("name", _SAMPLE)
def test_a_sampled_reference_solution_passes(name):
    """The harness runs the converted problems correctly.

    A conversion that produced tests nobody could pass would look exactly like a weak
    model, which is why this asserts the success condition rather than the absence of an
    error. Three of these five are the problems whose `check` calls a helper defined in
    the prompt; importing only the entry point left them raising NameError from the test.
    """
    directory = _ROOT / name
    passed, detail = asyncio.run(CodingEvaluator()._run_tests(
        (directory / "solution.py").read_text(encoding="utf-8"), {"_dir": directory}, 30))

    assert passed, f"{name}: the reference solution did not pass its own tests: {detail}"


def test_the_attribution_travels_with_the_problems():
    """MIT requires the notice to travel with the work, and a reader finding these
    problems in the package needs to know whose they are."""
    notice = (_ROOT / "HUMANEVAL.md").read_text(encoding="utf-8")
    assert "MIT" in notice
    assert "openai" in notice.lower()


def test_the_bespoke_problems_are_still_here():
    """The four written for this project sit alongside the imported set rather than
    being replaced by it."""
    names = {d.name for d in _ROOT.iterdir() if d.is_dir()}
    assert {"two_sum", "kadane", "rle_encode", "balanced_brackets"} <= names
