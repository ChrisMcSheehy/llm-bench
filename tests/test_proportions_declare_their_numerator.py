"""Which figures are proportions, decided by the code that computed them (design C1).

A name-based allowlist cannot answer this: `score_mean` is produced by the shared
aggregator for every module, and it is a proportion for mcqa (right or wrong) and a
continuous mean for reassembly (bit accuracy on a gradient). One name, two kinds.
"""
import pytest

from llmbench.evaluators.base import _successes
from llmbench.evaluators.mcqa import MCQAEvaluator
from llmbench.evaluators.reassembly import ReassemblyEvaluator
from llmbench.models import Sample


@pytest.mark.parametrize("values,expected", [
    ([1.0, 1.0, 1.0], 3),
    ([0.0, 0.0, 0.0], 0),
    ([1.0, 0.0, 1.0], 2),
    ([True, False, True], 2),
    ([1, 0, 1], 2),
])
def test_a_list_of_ones_and_zeros_is_a_count_of_successes(values, expected):
    assert _successes(values) == expected


@pytest.mark.parametrize("values", [
    [0.5, 1.0],       # a continuous mean - reassembly's bit_accuracy
    [2.0, 3.0],       # parts_found, which counts 0..3
    [0.9999, 1.0],    # near-perfect is not perfect
])
def test_anything_else_has_no_numerator(values):
    assert _successes(values) is None


def test_no_values_is_not_a_numerator_of_zero():
    """An empty list is an absence of measurement, not a measurement of none."""
    assert _successes([]) is None


def _graded(evaluator, scores):
    return [Sample(evaluator=evaluator, case_id=str(i), score=s,
                   passed=bool(s), answered=True)
            for i, s in enumerate(scores)]


def test_a_binary_evaluator_reports_its_numerator():
    metrics = {m.name: m for m in
               MCQAEvaluator().aggregate(_graded("mcqa", [1.0, 1.0, 0.0]))}
    assert metrics["score_mean"].successes == 2
    assert metrics["score_mean"].n == 3
    assert metrics["pass_rate"].successes == 2
    assert metrics["answer_rate"].successes == 3


def test_a_count_is_not_given_a_numerator():
    """error_count's value is a count and its n is the sample total. An interval
    built from those two would describe a rate while the number beside it is a
    count - a worse defect than no interval."""
    metrics = {m.name: m for m in MCQAEvaluator().aggregate(_graded("mcqa", [1.0, 0.0]))}
    assert metrics["error_count"].successes is None
    assert metrics["skipped_count"].successes is None


def test_a_continuous_score_is_not_given_a_numerator():
    metrics = {m.name: m for m in
               ReassemblyEvaluator().aggregate(_graded("reassembly", [0.25, 0.75]))}
    assert metrics["score_mean"].successes is None


def test_reassemblys_own_tiers_report_their_numerators():
    """The tiers loop is a second construction site inside reassembly, reached only
    when samples carry the meta it reads. The score_mean check above runs the shared
    aggregator and never touches it - which is how an unimported name survived there
    (LESSONS: verification-should-exceed-the-plans-minimum)."""
    samples = [
        Sample(evaluator="reassembly", case_id="a", score=1.0, answered=True,
               meta={"parts_found": 3, "order_correct": True, "exact_match": True}),
        Sample(evaluator="reassembly", case_id="b", score=0.0, answered=True,
               meta={"parts_found": 2, "order_correct": False, "exact_match": False}),
    ]
    metrics = {m.name: m for m in ReassemblyEvaluator().aggregate(samples)}

    # order_correct and exact_match are 0/1 and carry a numerator.
    assert metrics["order_correct"].successes == 1
    assert metrics["exact_match"].successes == 1
    # parts_found counts 0..3, so it is not a proportion of anything.
    assert metrics["parts_found"].successes is None
