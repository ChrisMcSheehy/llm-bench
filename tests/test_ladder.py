"""Climbing a ladder of context lengths, and the two rules that stop the climb.

The clock is injected so the budget rules are exercised in microseconds rather than
by sleeping. Every test asserts which rungs ran and why the rest did not - asserting
merely that "something was skipped" would pass on a climber that skipped everything.
"""
from __future__ import annotations

import asyncio

from llmbench.evaluators._ladder import climb, context_ladder
from llmbench.models import Sample


def _ok(length: int) -> Sample:
    return Sample(evaluator="t", case_id=str(length), score=1.0,
                  dims={"context_len": length})


def _skip(length: int, reason: str) -> Sample:
    return Sample(evaluator="t", case_id=f"{length}:skipped", skipped=reason,
                  dims={"context_len": length})


class _Clock:
    """A clock that advances by whatever the rung is told to cost."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _run(rungs, *, costs=None, fails_at=None, budget=None):
    """Climb `rungs`, where each rung costs `costs[length]` seconds and may fail."""
    clock = _Clock()
    attempted: list[int] = []

    async def run_rung(length: int) -> list[Sample]:
        attempted.append(length)
        clock.now += (costs or {}).get(length, 1.0)
        if fails_at is not None and length == fails_at:
            return [Sample(evaluator="t", case_id=str(length), error="out of memory")]
        return [_ok(length)]

    samples = asyncio.run(climb(rungs, run_rung, _skip, budget, clock))
    return attempted, samples


def _skipped(samples) -> list[Sample]:
    return [s for s in samples if s.skipped]


def test_every_rung_runs_when_nothing_stops_the_climb():
    attempted, samples = _run([2048, 8192, 32768])
    assert attempted == [2048, 8192, 32768]
    assert _skipped(samples) == []


def test_the_rungs_are_climbed_from_the_bottom_however_they_are_given():
    """D3: approaching the limit from below finds it cheaply."""
    attempted, _ = _run([32768, 2048, 8192])
    assert attempted == [2048, 8192, 32768]


def test_a_failed_rung_stops_the_climb():
    attempted, _ = _run([2048, 8192, 32768], fails_at=8192)
    assert attempted == [2048, 8192], "the climb continued past a failure"


def test_the_rungs_above_a_failure_say_which_rung_failed_and_why():
    _, samples = _run([2048, 8192, 32768], fails_at=8192)
    skipped = _skipped(samples)
    assert [s.dims["context_len"] for s in skipped] == [32768]
    assert "8192" in skipped[0].skipped
    assert "out of memory" in skipped[0].skipped


def test_a_failure_is_not_recorded_as_a_skip_on_its_own_rung():
    """The rung that failed carries an error. Only the ones never tried are skips."""
    _, samples = _run([2048, 8192], fails_at=8192)
    failed = [s for s in samples if s.error]
    assert len(failed) == 1
    assert failed[0].skipped is None


# Measured against a real llama-server on 2026-08-05 (build b10144-d73c1d6b2,
# gemma4-heretic Q8_0, Radeon RX 7900 XTX), so the numbers below can be re-derived rather
# than trusted - see docs/ironclad/PROBE-2026-08-04-ladder-timing.md:
#
#     rung    actual   projected from the rung below   ratio
#     4096       6.3s   -                              -
#     8192       7.4s   12.6s                          0.58   over-estimate
#    16384      17.9s   14.8s                          1.21
#    32768      51.2s   35.8s                          1.43
#
# The projection is a lower bound at the two upper steps and an over-estimate at the
# bottom one, because a rung costs fixed overhead plus a term proportional to length. That
# is an accepted weakness: the over-estimate happens three orders of magnitude below the
# 1800s budget, where it cannot cause a skip.
def test_the_budget_stops_the_climb_before_a_rung_that_will_not_fit():
    """80s at 8192 projects 320s at 32768, which does not fit the remaining budget."""
    attempted, samples = _run([2048, 8192, 32768],
                              costs={2048: 20.0, 8192: 80.0}, budget=200.0)
    assert attempted == [2048, 8192]
    skipped = _skipped(samples)
    assert [s.dims["context_len"] for s in skipped] == [32768]
    assert "budget" in skipped[0].skipped


def test_a_fast_ladder_is_never_stopped_by_the_budget():
    """The same rungs, cheap: the rule must not fire merely because a budget exists."""
    attempted, samples = _run([2048, 8192, 32768],
                              costs={2048: 0.5, 8192: 2.0}, budget=200.0)
    assert attempted == [2048, 8192, 32768]
    assert _skipped(samples) == []


def test_without_a_budget_the_climb_never_stops_for_time():
    attempted, _ = _run([2048, 8192, 32768], costs={2048: 900.0, 8192: 3600.0})
    assert attempted == [2048, 8192, 32768]


def test_the_ladder_itself_is_unchanged_by_this_move():
    """context_ladder moved file and nothing else (D3b)."""
    assert context_ladder(16384, [8192, 32768], 6) == [8192, 16384]
    assert context_ladder(1048576, [8192, 32768, 131072], 2) == [131072, 1048576]
