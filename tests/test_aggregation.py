"""What the default aggregator does with samples that were never attempted.

A skipped sample is neither a result nor a failure. Counting it as either produces a
number that describes a measurement nobody took.
"""
from __future__ import annotations

from llmbench.evaluators.base import Evaluator
from llmbench.models import Sample


class _Eval(Evaluator):
    name = "t"

    async def evaluate(self, ctx):        # never called; aggregate() is the subject
        return []


def _graded(score: float) -> Sample:
    return Sample(evaluator="t", case_id="g", score=score, passed=score >= 0.5)


def _named(metrics) -> dict[str, float]:
    return {m.name: m.value for m in metrics}


def test_a_skipped_sample_is_not_averaged_into_the_score():
    """Two perfect answers and an unattempted rung is 1.0, not 0.67 and not 0.0.

    A regression guard, not a red test: a skipped sample carries no score, so the
    `is not None` guards already excluded it before the `graded` filter was tightened.
    It is here so that a future skip which *does* carry a figure cannot slip into a
    mean unnoticed.
    """
    metrics = _named(_Eval().aggregate(
        [_graded(1.0), _graded(1.0), Sample(evaluator="t", case_id="s", skipped="why")]))
    assert metrics["score_mean"] == 1.0


def test_skipped_and_errored_are_counted_separately():
    metrics = _named(_Eval().aggregate([
        _graded(1.0),
        Sample(evaluator="t", case_id="e", error="boom"),
        Sample(evaluator="t", case_id="s", skipped="not attempted"),
    ]))
    assert metrics["error_count"] == 1.0
    assert metrics["skipped_count"] == 1.0


def test_a_skip_never_reads_as_a_failure():
    """The distinction D3 exists for: a modest machine's limit is not a broken run.

    Also a regression guard rather than a red test - error_count never counted skips,
    because before this phase there were none to count. It pins the distinction now
    that both states exist and could be conflated by a careless edit.
    """
    metrics = _named(_Eval().aggregate(
        [_graded(1.0), Sample(evaluator="t", case_id="s", skipped="not attempted")]))
    assert metrics["error_count"] == 0.0


def test_the_counts_survive_an_evaluator_that_graded_nothing_at_all():
    """Returning no metrics at all would leave the dashboard unable to say why."""
    metrics = _named(_Eval().aggregate([
        Sample(evaluator="t", case_id="e", error="boom"),
        Sample(evaluator="t", case_id="s", skipped="not attempted"),
    ]))
    assert metrics["error_count"] == 1.0
    assert metrics["skipped_count"] == 1.0


def test_the_mean_says_how_many_scores_it_averaged():
    metrics = {m.name: m for m in _Eval().aggregate(
        [_graded(1.0), _graded(0.0), _graded(1.0)])}
    assert metrics["score_mean"].n == 3


def test_the_counts_exclude_the_samples_that_were_not_graded():
    """Five samples, three of them measurements: the mean rests on three."""
    metrics = {m.name: m for m in _Eval().aggregate([
        _graded(1.0), _graded(0.0), _graded(1.0),
        Sample(evaluator="t", case_id="e", error="boom"),
        Sample(evaluator="t", case_id="s", skipped="not attempted"),
    ])}
    assert metrics["score_mean"].n == 3
    assert metrics["pass_rate"].n == 3


def test_a_count_of_failures_rests_on_every_sample_including_the_failures():
    """error_count is 1 out of 5, not 1 out of the 3 that worked."""
    metrics = {m.name: m for m in _Eval().aggregate([
        _graded(1.0), _graded(0.0), _graded(1.0),
        Sample(evaluator="t", case_id="e", error="boom"),
        Sample(evaluator="t", case_id="s", skipped="not attempted"),
    ])}
    assert metrics["error_count"].n == 5
    assert metrics["skipped_count"].n == 5


def test_the_throughput_mean_counts_only_the_samples_that_reported_a_speed():
    slow = Sample(evaluator="t", case_id="g", score=1.0, passed=True, tok_per_sec=40.0)
    metrics = {m.name: m for m in _Eval().aggregate([slow, _graded(1.0)])}
    assert metrics["tok_per_sec_mean"].n == 1, "the sample with no speed was counted"
