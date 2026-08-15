"""Climbing a ladder of context lengths, and knowing when to stop.

Shared by the two evaluators that walk such a ladder - `needle` and `long_context` -
so that both stop for the same reasons and a reader has one place to check what those
reasons are. A helper module rather than an evaluator: it registers nothing, in the
same way `_extract.py` does not.

Design D3. Start at the smallest rung and climb, because the only reliable way to find
an unknown machine's ceiling is to walk up to it, and approaching from below discovers
the limit cheaply instead of by paying for the most expensive attempt first. A rung
that is not attempted is recorded as skipped **with a reason**, which is a state of its
own and never an error: a modest machine's honest limit is not a broken run.
"""
from __future__ import annotations

import time
from typing import Awaitable, Callable, Optional

from llmbench.models import Sample


def context_ladder(n_ctx: int, knots: list[int], max_rungs: int) -> list[int]:
    """The rungs to attempt for a model with this context length.

    Moved here from `needle.py` unchanged (design D3b): what the ladder contains was
    never the problem, and the model's advertised maximum must stay on it or
    `effective_ctx` - the largest rung still answering correctly - cannot reach it.
    """
    rungs = sorted({k for k in knots if k <= n_ctx})
    if not rungs or rungs[-1] != n_ctx:
        rungs.append(n_ctx)
    rungs = sorted(set(rungs))
    if len(rungs) > max_rungs:                # keep the top rungs, always incl. n_ctx
        rungs = rungs[-max_rungs:]
    return rungs


async def climb(rungs: list[int],
                run_rung: Callable[[int], Awaitable[list[Sample]]],
                skipped_sample: Callable[[int, str], Sample],
                time_budget_s: Optional[float] = None,
                clock: Callable[[], float] = time.monotonic) -> list[Sample]:
    """Attempt each rung in ascending order until it stops being worth it.

    `run_rung(length)` produces every sample for one rung. `skipped_sample(length,
    reason)` builds the single row that stands for a rung nobody attempted - one per
    rung, not one per cell, because the meaningful count is how many rungs were left
    unclimbed. `clock` is injectable so the budget can be tested without sleeping.

    Two rules stop the climb, both checked only *between* rungs:

    1. Any sample in the completed rung carrying an error. The machine has refused;
       everything above will refuse harder.
    2. The rung just completed projects the next one past `time_budget_s`. The
       projection is linear - `seconds * (next_length / this_length)` - which is
       deliberately the optimistic estimate: processing a prompt costs at least linear
       time in its length, since attention adds a quadratic term on top. Stopping only
       when even the optimistic projection overshoots means this never skips a rung
       that would have fitted. It may allow one that overruns, which is the harmless
       direction.

    A rung already running is never abandoned, so the worst case is a single rung
    overrunning the budget rather than a truncated measurement. `time_budget_s=None`
    disables the second rule entirely.
    """
    ladder = sorted(r for r in rungs if r > 0)
    out: list[Sample] = []
    elapsed = 0.0
    stop: Optional[str] = None

    for i, length in enumerate(ladder):
        if stop:
            out.append(skipped_sample(length, stop))
            continue

        started = clock()
        samples = await run_rung(length)
        took = clock() - started
        elapsed += took
        out.extend(samples)

        failure = next((s.error for s in samples if s.error), None)
        if failure:
            stop = (f"not attempted: the {length}-token rung failed ({failure[:120]})")
            continue

        if time_budget_s is None or i + 1 >= len(ladder):
            continue
        nxt = ladder[i + 1]
        projected = took * (nxt / length)
        if elapsed + projected > time_budget_s:
            stop = (f"not attempted: the climb stopped after {length} tokens - the next "
                    f"rung projects to {projected:.0f}s on top of the {elapsed:.0f}s "
                    f"already spent, past the {time_budget_s:.0f}s budget")
    return out
