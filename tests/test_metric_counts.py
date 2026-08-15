"""What the bespoke aggregators say their figures rest on.

Three of the trickiest, where the count is not simply "every graded sample". The
whole-suite check in selftest.py covers the other seven, because an edit list is
exactly as blind as the person who wrote it.
"""
from __future__ import annotations

from llmbench.evaluators.coding import CodingEvaluator
from llmbench.evaluators.needle import NeedleEvaluator
from llmbench.models import Sample


def _needle(length: int, depth: int, score: float) -> Sample:
    return Sample(evaluator="needle", case_id=f"{length}:{depth}", score=score,
                  passed=score >= 0.5, dims={"context_len": length, "depth_pct": depth})


def test_needle_recall_counts_the_cells_in_that_rung():
    """Recall at 2048 is averaged over the three depths probed there, not over six."""
    metrics = NeedleEvaluator().aggregate([
        _needle(2048, 0, 1.0), _needle(2048, 50, 1.0), _needle(2048, 100, 0.0),
        _needle(8192, 0, 1.0), _needle(8192, 50, 1.0), _needle(8192, 100, 1.0)])
    recall = [m for m in metrics if m.name == "recall"]
    assert [m.n for m in recall] == [3, 3], [(m.dims, m.n) for m in recall]


def test_effective_ctx_counts_rungs_and_not_samples():
    """It picks the largest rung, so the rungs are what it rests on."""
    metrics = NeedleEvaluator().aggregate([
        _needle(2048, 0, 1.0), _needle(2048, 50, 1.0), _needle(2048, 100, 0.0),
        _needle(8192, 0, 1.0), _needle(8192, 50, 1.0), _needle(8192, 100, 1.0)])
    eff = [m for m in metrics if m.name == "effective_ctx"]
    assert len(eff) == 1
    assert eff[0].n == 2, "two rungs were measured, so two rungs is what it chose from"


def test_coding_pass_at_1_counts_every_graded_attempt():
    """Two problems attempted twice each is four attempts, not two problems."""
    attempts = [Sample(evaluator="coding", case_id=f"{p}:{i}", passed=True,
                       dims={"problem": p})
                for p in ("two_sum", "kadane") for i in (0, 1)]
    metrics = {m.name: m for m in CodingEvaluator().aggregate(attempts) if not m.dims}
    assert metrics["pass@1"].n == 4
