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

import random

#: Invented places and plain adjectives. The filler has to read like ordinary prose,
#: because a model processes repeated nonsense very differently from text, and it has to
#: be *invented* so that nothing planted in it can be answered from training data instead
#: of from the context.
CITIES = [
    "Brindlemoor", "Vantag", "Ashford Cross", "Kelbourne", "Marrowdeep",
    "Highcinder", "Ostravel", "Dunmarch", "Fenwillow", "Corveth",
]
WORDS = ["amber", "quartz", "cobalt", "verdant", "saffron", "cinder", "harbor",
         "lantern", "meridian", "thistle", "fathom", "gable", "orchard"]

FILLER_TEMPLATES = [
    "The council of {c} recorded that the {w1} shipments arrived before the {w2} season.",
    "In the district of {c}, {w1} merchants traded quietly along the {w2} road.",
    "Records from {c} note an unusually mild winter, with {w1} frost and {w2} rain.",
    "The librarian of {c} catalogued {w1} manuscripts beside the {w2} archives.",
    "Travellers passing through {c} spoke of {w1} lanterns and {w2} bridges.",
    "A survey of {c} listed {w1} orchards, {w2} mills, and several old wells.",
]

#: A probe of ordinary prose for the calibration below. Repeated so that one tokenize
#: call sees enough text for the ratio to settle.
CALIBRATION_PROBE = " ".join(
    t.format(c="Brindlemoor", w1="amber", w2="cobalt") for t in FILLER_TEMPLATES) * 8


def build_filler(rng: random.Random, char_budget: int) -> str:
    """Prose of roughly `char_budget` characters, drawn deterministically from `rng`.

    Shared by every evaluator that has to bury something in a long document, so that the
    haystack is the same kind of text in each and a result from one is comparable with a
    result from another. Overshoots the budget by at most one sentence, which matters far
    less than cutting a sentence in half would.
    """
    parts, size = [], 0
    while size < char_budget:
        template = rng.choice(FILLER_TEMPLATES)
        sentence = template.format(c=rng.choice(CITIES), w1=rng.choice(WORDS),
                                   w2=rng.choice(WORDS))
        parts.append(sentence)
        size += len(sentence) + 1
    return " ".join(parts)


async def chars_per_token(ctx, probe: str) -> float:
    """Characters per token for this backend, measured once against `probe`.

    `max(1, ...)` guards a backend that reports zero tokens for a non-empty string -
    a division by zero here would abort a whole run over a sizing detail.
    """
    tokens = await ctx.count_tokens(probe)
    return len(probe) / max(1, tokens)
