"""Interval arithmetic for figures this bench publishes.

Pure functions over counts. No I/O, no imports from the rest of the project, so this
is testable without a store, a server or a run (design C7).

Why this is hand-written rather than `scipy.stats`: scipy plus numpy is roughly a
hundred megabytes of dependency to avoid twenty lines of arithmetic, in a project whose
seven dependencies are all small and whose bundled-data rule exists so that it works
without a download.
"""
from __future__ import annotations

import math
from typing import Optional

#: Two-sided 95%. Hard-coded rather than a parameter: a bench that prints intervals at
#: mixed confidence levels in one table is unreadable, and nothing has asked for another.
#: `statistics.NormalDist().inv_cdf(0.975)`, evaluated once and pinned so the printed
#: figures cannot drift with a standard-library change.
Z_95 = 1.959963984540054


def wilson(successes: int, n: int, z: float = Z_95) -> Optional[tuple[float, float]]:
    """The Wilson score interval for `successes` out of `n`.

    Returns (low, high), or None when there is nothing to describe.

    Wilson rather than the normal ("Wald") interval because this bench's question sets
    are small and perfect scores are routine, which is exactly where the normal interval
    fails: on 6 of 6 it returns [1.000, 1.000], claiming certainty from six questions,
    and on 5 of 6 it returns an upper bound of 1.132, which is not a probability. Wilson
    does neither, at any n (design C2).

    `n == 0` returns None rather than [0, 1]. An interval over no data is not a wide
    interval; it is an absent one, and this project displays absent as a dash and never
    as a number (design D3, D7a).
    """
    if successes < 0 or n < 0:
        raise ValueError(f"counts cannot be negative: {successes} of {n}")
    if successes > n:
        raise ValueError(f"more successes than items: {successes} of {n}")
    if n == 0:
        return None

    p = successes / n
    denominator = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    low = (centre - spread) / denominator
    high = (centre + spread) / denominator
    # The two extremes have an exact algebraic answer that floating point misses by a
    # few parts in 10^17: at k=0 the centre and the spread are equal, so the lower bound
    # is exactly 0; at k=n the numerator and denominator are equal, so the upper bound is
    # exactly 1. Left to the arithmetic they come out as 5e-17 and 0.99999999999999989,
    # which puts the observed rate a whisker outside its own interval - invisible at two
    # decimal places and wrong. Set rather than clamped, because clamping keeps the
    # residue on whichever side it lands.
    low = 0.0 if successes == 0 else max(0.0, low)
    high = 1.0 if successes == n else min(1.0, high)
    return (low, high)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on `b` and `c` discordant outcomes.

    `b` is the number of items the first configuration got right and the second got
    wrong; `c` is the reverse. Returns the probability of seeing a split at least this
    lopsided if the two were equally good.

    **Only the discordant items are counted, and that is the whole point.** Items both
    configurations answered correctly, and items both got wrong, say nothing about which
    is better — they are facts about the question. All the information about a difference
    lives where the two disagreed, and comparing two independent intervals instead throws
    that pairing away (design C4).

    The exact binomial form rather than the chi-square approximation, because the
    approximation needs the discordant count to be large and this project's question sets
    produce a handful. Under the null hypothesis the split is a fair coin, so this is the
    two-tailed binomial probability of `min(b, c)` or fewer heads in `b + c` tosses.

    `b == c == 0` returns 1.0: two runs that never disagreed give no evidence of a
    difference. That is emphatically not evidence that they are the same, which is why
    every caller prints the counts beside the verdict (design C5).
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts cannot be negative: b={b} c={c}")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n)
    # Doubling a one-tailed probability overshoots when the split is near even, and 1.06
    # is not a probability. Capped rather than left to print.
    return min(1.0, 2 * tail)
