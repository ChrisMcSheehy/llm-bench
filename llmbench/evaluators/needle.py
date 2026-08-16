"""Needle-in-a-haystack across a KV-cache ladder.

For a model with detected n_ctx, we test retrieval at a ladder of context
lengths (e.g. 128k / 256k / 512k / 1000k for a 1M model) crossed with insertion
depths (0-100%). Each cell is one graded sample; the aggregate is the classic
length x depth recall heatmap plus a headline recall.

Sizing note: we calibrate chars-per-token once via the backend tokenizer, build
the haystack by character budget, then record the *actual* prompt-token count
the server reports. That keeps us to one tokenize round-trip regardless of how
big the context ladder is — important when a rung is a million tokens.
"""
from __future__ import annotations

import random
import re
from typing import Any

from llmbench.evaluators._ladder import climb, context_ladder
from llmbench.evaluators._sizing import (
    CALIBRATION_PROBE, CITIES, WORDS, build_filler, chars_per_token,
)
from llmbench.evaluators.base import Breakdown, EvalContext, Evaluator, Verdict, View
from llmbench.models import Metric, Sample
from llmbench.registry import register

#: The recall a rung must still reach to count as usable context. Two thirds, so a rung
#: that fails a third of its probes is not reported as one the model can work at.
_RECALL_FLOOR = 0.66

_DEFAULTS = {
    "context_lengths": None,          # explicit list; overrides the ladder
    "context_fractions": None,        # explicit fractions of n_ctx
    "ladder_knots": [8192, 32768, 131072, 262144, 524288, 1048576],
    "max_rungs": 6,
    "depths": [0, 25, 50, 75, 100],   # insertion depth, %
    "answer_tokens": 32,
    "overhead_tokens": 256,           # instructions + question + slack
    "seed": 7,
    "repeats": 1,                     # samples per (length, depth) cell
    # Thirty minutes per ladder. The climb stops when the rung just completed projects
    # the next one past what is left of this; null disables the rule (design D3c).
    # Sized to refuse the rung that would run for hours, not to trim an ordinary ladder.
    "time_budget_s": 1800,
}


@register
class NeedleEvaluator(Evaluator):
    name = "needle"
    version = "1"
    default_config = _DEFAULTS
    breakdowns = [Breakdown("recall", ("context_len",))]
    views = [View("heatmap", "recall by context length and insertion depth",
                  x="context_len", y="depth_pct")]

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        rng = random.Random(cfg["seed"])
        n_ctx = ctx.fingerprint.n_ctx or 8192

        lengths = self._lengths(cfg, n_ctx)
        cpt = await chars_per_token(ctx, CALIBRATION_PROBE)

        async def rung(length: int) -> list[Sample]:
            budget_tokens = max(512, length - cfg["overhead_tokens"] - cfg["answer_tokens"])
            return [await self._one(ctx, cfg, rng, length, budget_tokens, depth, cpt, rep)
                    for depth in cfg["depths"] for rep in range(cfg["repeats"])]

        # Climb from the bottom and stop when the machine refuses or the clock says the
        # next rung will not fit (design D3). Rungs above that are recorded as skipped
        # with a reason rather than attempted and failed.
        return await climb(lengths, rung, self._skipped, cfg["time_budget_s"])

    def _skipped(self, length: int, reason: str) -> Sample:
        """One row for a rung nobody climbed, whatever the depths would have been.

        The meaningful count is how many rungs were left unclimbed, and a row per cell
        would report the same fact five times.
        """
        return Sample(evaluator=self.name, case_id=f"{length}:skipped",
                      group=str(length), dims={"context_len": length}, skipped=reason)

    def _lengths(self, cfg: dict[str, Any], n_ctx: int) -> list[int]:
        if cfg["context_lengths"]:
            # Sorted rather than trusted: the climb goes upward, and a list written out
            # of order would otherwise start near the top.
            return sorted(n for n in cfg["context_lengths"] if n <= n_ctx) or [n_ctx]
        if cfg["context_fractions"]:
            return sorted({max(512, int(n_ctx * f)) for f in cfg["context_fractions"]})
        return context_ladder(n_ctx, cfg["ladder_knots"], cfg["max_rungs"])

    async def _one(self, ctx, cfg, rng, length, budget_tokens, depth, cpt, rep) -> Sample:
        city = rng.choice(CITIES)
        code = f"{rng.choice(WORDS).upper()}-{rng.randint(1000, 9999)}"
        needle = (f" IMPORTANT: The special access code for the city of {city} "
                  f"is {code}. Remember it. ")

        char_budget = int(budget_tokens * cpt)
        filler = build_filler(rng, char_budget)
        cut = int(len(filler) * (depth / 100.0))
        # snap to a word boundary so we don't bisect a token nastily
        space = filler.rfind(" ", 0, cut) if cut else 0
        cut = space if space > 0 else cut
        haystack = filler[:cut] + needle + filler[cut:]

        system = ("You are given a long document. Somewhere in it is a special "
                  "access code for a city. Read carefully.")
        user = (f"{haystack}\n\nQuestion: What is the special access code for the "
                f"city of {city}? Reply with only the code, nothing else.")

        def grade(res) -> Verdict:
            found = code.lower() in re.sub(r"\s+", " ", res.text).lower()
            return Verdict(score=1.0 if found else 0.0, passed=found,
                           meta={"code": code, "answer": res.text[:200],
                                 "truncated": res.truncated})

        # A response that never reached an answer is not a failed retrieval, and scoring
        # it zero would report a measurement nobody took (design D3). run_case records it
        # as skipped and never calls the grader.
        return await self.run_case(
            ctx, case_id=f"{length}:{depth}:{rep}", group=str(length),
            dims={"context_len": length, "depth_pct": depth, "city": city, "rep": rep},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            grade=grade, max_tokens=cfg["answer_tokens"], temperature=0.0)

    def aggregate(self, samples: list[Sample]) -> list[Metric]:
        """Per-rung recall is declared above; only the effective context is bespoke."""
        metrics = super().aggregate(samples)
        # Read back from the recall figures rather than re-grouping the samples, so the
        # rung this picks and the rung the table shows cannot come from different sums.
        by_len = {m.dims["context_len"]: m.value for m in metrics
                  if m.name == "recall" and "context_len" in m.dims}
        good = [length for length, recall in by_len.items() if recall >= _RECALL_FLOOR]
        if good:
            # The rungs measured, not the samples: this figure chooses among rungs, and
            # a count of samples would suggest a precision it does not have.
            metrics.append(Metric(evaluator=self.name, name="effective_ctx",
                                  value=float(max(good)), unit="tokens",
                                  n=len(by_len)))
        return metrics
