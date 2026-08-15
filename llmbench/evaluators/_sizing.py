"""Building a prompt of approximately a given token length.

Shared by every evaluator that needs a prompt of a stated size - `needle` and
`long_context` walk a ladder of them, `speed` measures how fast one is ingested. A
helper module rather than an evaluator: it registers nothing, like `_extract.py` and
`_ladder.py`.

The point is the calibration. Asking the server to tokenise a million-token prompt to
find out how long it is costs the same round-trip as the measurement itself, and doing
that per rung makes the sizing more expensive than the test. So the ratio of characters
to tokens is measured **once** against a short probe, and every prompt after that is
built to a character budget.

That makes the size approximate, and the approximation is never reported as a fact: the
server tells us the real prompt-token count in its own response, and that is what the
sample records.
"""
from __future__ import annotations


async def chars_per_token(ctx, probe: str) -> float:
    """Characters per token for this backend, measured once against `probe`.

    `max(1, ...)` guards a backend that reports zero tokens for a non-empty string -
    a division by zero here would abort a whole run over a sizing detail.
    """
    tokens = await ctx.count_tokens(probe)
    return len(probe) / max(1, tokens)
