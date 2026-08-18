"""The arithmetic behind every interval this bench prints.

The tables here are measured values recorded in DESIGN-statistical-confidence.md
(C2), not values this code produced. They are the regression net: if a refactor
changes what Wilson returns, these fail.
"""
import pytest

from llmbench.stats import wilson


def test_a_perfect_score_on_six_items_does_not_claim_certainty():
    """The whole reason Wilson was chosen over the textbook normal interval.

    The normal interval returns [1.000, 1.000] here - certainty, from six questions.
    """
    lo, hi = wilson(6, 6)
    assert lo == pytest.approx(0.610, abs=0.001)
    assert hi == pytest.approx(1.000, abs=0.001)
    assert lo < 1.0


def test_a_zero_score_on_six_items_does_not_claim_certainty():
    lo, hi = wilson(0, 6)
    assert lo == pytest.approx(0.000, abs=0.001)
    assert hi == pytest.approx(0.390, abs=0.001)
    assert hi > 0.0


@pytest.mark.parametrize("k,n,lo,hi", [
    (5, 6, 0.436, 0.970),
    (83, 100, 0.745, 0.891),
    (67, 100, 0.573, 0.754),
    (164, 164, 0.977, 1.000),
    (150, 164, 0.862, 0.948),
])
def test_it_reproduces_the_measured_table(k, n, lo, hi):
    got_lo, got_hi = wilson(k, n)
    assert got_lo == pytest.approx(lo, abs=0.001)
    assert got_hi == pytest.approx(hi, abs=0.001)


def test_no_interval_ever_leaves_the_unit_range():
    """Swept, not sampled at chosen points: the failure being guarded against is an
    estimator that misbehaves at an extreme nobody thought to write down."""
    for n in range(1, 60):
        for k in range(0, n + 1):
            lo, hi = wilson(k, n)
            assert 0.0 <= lo <= hi <= 1.0, f"{k}/{n} produced [{lo}, {hi}]"


def test_the_point_estimate_lies_inside_its_own_interval():
    for n in range(1, 60):
        for k in range(0, n + 1):
            lo, hi = wilson(k, n)
            assert lo <= k / n <= hi, f"{k}/{n} excluded its own rate"


def test_no_items_means_no_interval():
    """Not [0, 1], and not a crash. An interval over nothing is not a wide interval."""
    assert wilson(0, 0) is None


def test_it_refuses_a_numerator_larger_than_the_denominator():
    """A caller that gets this wrong has a defect; returning a plausible number
    would hide it (lesson: assert-the-success-condition-not-the-absence-of-error)."""
    with pytest.raises(ValueError):
        wilson(7, 6)


# ---- the paired comparison (design C4) ------------------------------------
#
# The interval above says how solid one figure is. These say whether the difference
# between two of them is one the questions can actually show.

from llmbench.stats import mcnemar_exact


@pytest.mark.parametrize("b,c,p", [
    (1, 0, 1.0000),
    (2, 0, 0.5000),
    (3, 0, 0.2500),
    (5, 0, 0.0625),
    (6, 1, 0.1250),
    (10, 2, 0.0386),
    (12, 3, 0.0352),
    (20, 5, 0.0041),
])
def test_mcnemar_reproduces_the_measured_table(b, c, p):
    """Measured values from DESIGN-statistical-confidence.md (C4), not values this
    code produced."""
    assert mcnemar_exact(b, c) == pytest.approx(p, abs=0.0001)


def test_winning_five_and_losing_none_is_still_not_significant():
    """The finding this phase exists to surface. A configuration that won five items
    outright and lost none cannot be separated from noise at the conventional
    threshold, and no other view of this data can say so."""
    assert mcnemar_exact(5, 0) > 0.05


def test_no_disagreement_is_no_evidence_of_a_difference():
    """Comparing a run with itself. Not a division by zero, and not significance."""
    assert mcnemar_exact(0, 0) == 1.0


def test_which_run_is_named_first_does_not_change_the_answer():
    for b, c in [(0, 0), (3, 1), (10, 2), (20, 5)]:
        assert mcnemar_exact(b, c) == pytest.approx(mcnemar_exact(c, b))


def test_a_probability_never_exceeds_one():
    """The doubling that makes the test two-sided can overshoot at small counts."""
    for b in range(0, 25):
        for c in range(0, 25):
            p = mcnemar_exact(b, c)
            assert 0.0 <= p <= 1.0, f"b={b} c={c} gave {p}"


def test_a_large_lopsided_split_is_significant():
    """The other end: enough disagreement, going one way, is a real difference."""
    assert mcnemar_exact(20, 5) < 0.01
