"""One correct copy of the scaffolding every test module used to write by hand.

Design E1, E2 and E5. Before this, each module transferred the six measurements from
the generation result onto its sample itself, and the copies had drifted: `needle` and
`coding` recorded the server's own prefill and decode speeds, `oneshot` recorded one of
them, and the other seven recorded neither. Nobody decided that — it is what copying
produces, and an audit is the only thing that finds it.

`test_no_evaluator_scores_a_response_that_never_answered` is the companion to this file:
it guards what run_case does with a response it must refuse to grade, this one guards
what it records when there is something to grade.
"""
from __future__ import annotations

import asyncio

import pytest

from llmbench.evaluators.base import Breakdown, EvalContext, Evaluator
from llmbench.models import ModelFingerprint, Sample
from llmbench.registry import available, get
from llmbench.resources import load_jsonl
from llmbench.targets.base import GenResult, Target

#: Every field a module used to copy by hand, and the three at the end are the ones the
#: copies kept losing.
_MEASUREMENTS = ("input_tokens", "output_tokens", "latency_ms",
                 "tok_per_sec", "server_prompt_tps", "server_gen_tps")


class _Timed(Target):
    """A model that answers instantly and reports every timing a real server reports.

    The text is deliberate rubbish that every grader can chew on without crashing: a
    letter for the multiple-choice grader, a code block for the coding one, an object for
    the JSON one. Whether it grades *well* is beside the point — this file is about what
    gets recorded, not what gets scored.
    """

    engine = "timed"

    async def detect(self) -> ModelFingerprint:
        return ModelFingerprint(engine=self.engine, base_url=self.base_url, model_id="m")

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def generate(self, messages, *, max_tokens=512, temperature=0.0, extra=None):
        return GenResult(
            text='A. {"name": "x", "arguments": {}}\n```python\ndef f():\n    return 1\n```',
            input_tokens=31, output_tokens=7, latency_ms=42.0, tok_per_sec=166.7,
            server_prompt_tps=900.5, server_gen_tps=88.25, finish_reason="stop")


#: Every evaluator that asks a model something, with a configuration small enough to run
#: here. `perplexity` is absent because it shells out to a binary and never generates.
#: `coding` runs with execute:false so no subprocess starts — the measurements it records
#: come from the model call, which happens either way.
_CONFIGS = {
    "needle": {"context_lengths": [2048], "depths": [50], "repeats": 1},
    "long_context": {"context_lengths": [2048], "queries_per_rung": 1},
    "mcqa": {"limit": 2},
    "math_qa": {"limit": 2},
    "ifeval": {},
    "structured": {},
    "text2sql": {},
    "coding": {"execute": False},
    "human": {"limit": 1},
    "oneshot": {"limit": 1},
    "speed": {"scenarios": ["decode"], "trials": 1, "warmup": False},
    "reassembly": {"context_lengths": [2048], "repeats": 1},
}


def _samples(name: str) -> list[Sample]:
    fp = ModelFingerprint(engine="timed", base_url="http://timed", model_id="m",
                          n_ctx=8192)
    ctx = EvalContext(target=_Timed("http://timed"), fingerprint=fp,
                      config=_CONFIGS[name])
    return asyncio.run(get(name)().evaluate(ctx))


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_every_evaluator_records_every_measurement(name):
    graded = [s for s in _samples(name) if s.error is None and s.skipped is None]
    assert graded, f"{name} produced nothing to inspect"

    missing = {field for s in graded for field in _MEASUREMENTS
               if getattr(s, field) is None}
    assert not missing, (
        f"{name} recorded no {sorted(missing)} — the shared path fills all six, so a "
        f"module missing one has stopped using it")


def test_the_evaluator_list_covers_everything_that_generates():
    """An evaluator added later must be added here, or this fails rather than quietly
    leaving the new one unchecked — the blindness an edit list always has."""
    missing = sorted({n for n in available() if n != "perplexity"} - set(_CONFIGS))
    assert not missing, f"evaluators missing from the measurement check: {missing}"


# ---- E5: one loader, and an error a person can act on ------------------------

def test_a_malformed_question_file_names_the_file_and_the_line(tmp_path):
    """The four hand-written loaders raised `Expecting value: line 1 column 1`, naming
    neither the file nor the row — for a file people edit by hand."""
    bad = tmp_path / "questions.jsonl"
    bad.write_text('{"id": 1}\n\n{"id": 2\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_jsonl(str(bad), "datasets", "mcqa.jsonl")

    message = str(exc.value)
    assert str(bad) in message, message
    assert "line 3" in message, (
        f"blank lines are skipped but still count towards the line number: {message}")


def test_a_line_of_the_wrong_shape_is_refused_where_it_is_written(tmp_path):
    """Valid JSON, wrong thing. A bare array on a line parses fine and then fails deep in
    an evaluator as `it["question"]` on a list, naming framework code rather than the row
    of the file that is wrong."""
    f = tmp_path / "q.jsonl"
    f.write_text('{"id": 1}\n["not", "an", "object"]\n', encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        load_jsonl(str(f), "datasets", "mcqa.jsonl")

    assert "line 2" in str(exc.value), str(exc.value)
    assert "list" in str(exc.value), "the error did not say what it found instead"


def test_the_loader_skips_blank_lines_and_honours_the_limit(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text('{"id": 1}\n\n{"id": 2}\n{"id": 3}\n', encoding="utf-8")
    assert load_jsonl(str(f), "datasets", "mcqa.jsonl") == [{"id": 1}, {"id": 2},
                                                            {"id": 3}]
    assert load_jsonl(str(f), "datasets", "mcqa.jsonl", limit=2) == [{"id": 1}, {"id": 2}]


# ---- E2: breakdowns declared rather than looped ------------------------------

class _Declared(Evaluator):
    name = "declared"
    breakdowns = [Breakdown("accuracy", ("subject",))]

    async def evaluate(self, ctx):        # never called; aggregate() is the subject
        return []


def _s(score: float | None, **dims) -> Sample:
    return Sample(evaluator="declared", case_id="c", score=score,
                  passed=None if score is None else score >= 0.5, dims=dims)


def _by_dim(metrics, name: str) -> dict:
    return {tuple(m.dims.values()): m for m in metrics if m.name == name}


def test_a_declared_breakdown_averages_within_each_category():
    rows = _by_dim(_Declared().aggregate([
        _s(1.0, subject="physics"), _s(0.0, subject="physics"),
        _s(1.0, subject="chemistry")]), "accuracy")

    assert rows[("physics",)].value == 0.5
    assert rows[("physics",)].n == 2, "the figure must say how many items it rests on"
    assert rows[("chemistry",)].value == 1.0


def test_a_response_that_never_arrived_is_not_averaged_in_as_a_zero():
    """The `s.score or 0.0` defect, now impossible to reintroduce module by module."""
    rows = _by_dim(_Declared().aggregate([
        _s(1.0, subject="physics"),
        Sample(evaluator="declared", case_id="s", skipped="no answer: cut off",
               dims={"subject": "physics"})]), "accuracy")

    assert rows[("physics",)].value == 1.0
    assert rows[("physics",)].n == 1


def test_a_sample_missing_the_dimension_is_left_out_rather_than_given_a_label():
    rows = _by_dim(_Declared().aggregate(
        [_s(1.0, subject="physics"), _s(0.0)]), "accuracy")

    assert list(rows) == [("physics",)], "a sample with no subject was filed under one"
    assert rows[("physics",)].value == 1.0


def test_a_breakdown_can_name_two_dimensions():
    """`long_context` splits recall by task *and* rung; one figure over both would hide
    which of its two sub-tasks broke."""
    from llmbench.evaluators.long_context import LongContextEvaluator

    rows = _by_dim(LongContextEvaluator().aggregate([
        Sample(evaluator="long_context", case_id="a", score=1.0,
               dims={"task": "multikey", "context_len": 2048}),
        Sample(evaluator="long_context", case_id="b", score=0.0,
               dims={"task": "vartrack", "context_len": 2048}),
    ]), "recall")

    assert rows[("multikey", 2048)].value == 1.0
    assert rows[("vartrack", 2048)].value == 0.0


def test_the_effective_context_agrees_with_the_recall_it_is_read_from():
    """It picks the largest rung still at the floor, and reads the same figures the
    table shows rather than re-summing the samples into a second opinion."""
    from llmbench.evaluators.needle import NeedleEvaluator

    metrics = NeedleEvaluator().aggregate([
        Sample(evaluator="needle", case_id="a", score=1.0, dims={"context_len": 2048}),
        Sample(evaluator="needle", case_id="b", score=0.0, dims={"context_len": 8192}),
    ])
    eff = [m for m in metrics if m.name == "effective_ctx"]
    recall = _by_dim(metrics, "recall")

    assert [m.value for m in eff] == [2048.0], "a rung at 0.0 recall was called usable"
    assert recall[(2048,)].value == 1.0 and recall[(8192,)].value == 0.0
