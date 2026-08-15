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
from llmbench.evaluators.base import EvalContext, Evaluator
from llmbench.models import Metric, Sample
from llmbench.registry import register

_CITIES = [
    "Brindlemoor", "Vantag", "Ashford Cross", "Kelbourne", "Marrowdeep",
    "Highcinder", "Ostravel", "Dunmarch", "Fenwillow", "Corveth",
]
_WORDS = ["amber", "quartz", "cobalt", "verdant", "saffron", "cinder", "harbor",
          "lantern", "meridian", "thistle", "fathom", "gable", "orchard"]

_FILLER_TEMPLATES = [
    "The council of {c} recorded that the {w1} shipments arrived before the {w2} season.",
    "In the district of {c}, {w1} merchants traded quietly along the {w2} road.",
    "Records from {c} note an unusually mild winter, with {w1} frost and {w2} rain.",
    "The librarian of {c} catalogued {w1} manuscripts beside the {w2} archives.",
    "Travellers passing through {c} spoke of {w1} lanterns and {w2} bridges.",
    "A survey of {c} listed {w1} orchards, {w2} mills, and several old wells.",
]

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

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        rng = random.Random(cfg["seed"])
        n_ctx = ctx.fingerprint.n_ctx or 8192

        lengths = self._lengths(cfg, n_ctx)
        cpt = await self._chars_per_token(ctx)

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

    async def _chars_per_token(self, ctx: EvalContext) -> float:
        block = " ".join(
            t.format(c="Brindlemoor", w1="amber", w2="cobalt") for t in _FILLER_TEMPLATES
        ) * 8
        toks = await ctx.count_tokens(block)
        return len(block) / max(1, toks)

    def _build_filler(self, rng: random.Random, char_budget: int) -> str:
        parts, size = [], 0
        while size < char_budget:
            t = rng.choice(_FILLER_TEMPLATES)
            s = t.format(c=rng.choice(_CITIES), w1=rng.choice(_WORDS), w2=rng.choice(_WORDS))
            parts.append(s)
            size += len(s) + 1
        return " ".join(parts)

    async def _one(self, ctx, cfg, rng, length, budget_tokens, depth, cpt, rep) -> Sample:
        city = rng.choice(_CITIES)
        code = f"{rng.choice(_WORDS).upper()}-{rng.randint(1000, 9999)}"
        needle = (f" IMPORTANT: The special access code for the city of {city} "
                  f"is {code}. Remember it. ")

        char_budget = int(budget_tokens * cpt)
        filler = self._build_filler(rng, char_budget)
        cut = int(len(filler) * (depth / 100.0))
        # snap to a word boundary so we don't bisect a token nastily
        space = filler.rfind(" ", 0, cut) if cut else 0
        cut = space if space > 0 else cut
        haystack = filler[:cut] + needle + filler[cut:]

        system = ("You are given a long document. Somewhere in it is a special "
                  "access code for a city. Read carefully.")
        user = (f"{haystack}\n\nQuestion: What is the special access code for the "
                f"city of {city}? Reply with only the code, nothing else.")

        dims = {"context_len": length, "depth_pct": depth, "city": city, "rep": rep}
        try:
            res = await ctx.generate(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                max_tokens=cfg["answer_tokens"], temperature=0.0,
            )
        except Exception as e:  # network / OOM / context overflow
            return Sample(evaluator=self.name, case_id=f"{length}:{depth}:{rep}",
                          group=str(length), dims=dims, error=repr(e))

        # A response that never reached an answer is not a failed retrieval, and scoring
        # it zero would report a measurement nobody took (design D3).
        if res.unusable_reason:
            return Sample(evaluator=self.name, case_id=f"{length}:{depth}:{rep}",
                          group=str(length), dims=dims, skipped=res.unusable_reason)

        found = code.lower() in re.sub(r"\s+", " ", res.text).lower()
        return Sample(
            evaluator=self.name, case_id=f"{length}:{depth}:{rep}", group=str(length),
            dims=dims, score=1.0 if found else 0.0, passed=found,
            input_tokens=res.input_tokens, output_tokens=res.output_tokens,
            latency_ms=res.latency_ms, tok_per_sec=res.tok_per_sec,
            server_prompt_tps=res.server_prompt_tps, server_gen_tps=res.server_gen_tps,
            meta={"code": code, "answer": res.text[:200], "truncated": res.truncated},
        )

    def aggregate(self, samples: list[Sample]) -> list[Metric]:
        metrics = super().aggregate(samples)
        graded = [s for s in samples if s.error is None and s.score is not None]
        if not graded:
            return metrics
        # Per-rung recall.
        by_len: dict[int, list[float]] = {}
        for s in graded:
            by_len.setdefault(s.dims["context_len"], []).append(s.score)
        for length, scores in sorted(by_len.items()):
            metrics.append(Metric(evaluator=self.name, name="recall",
                                  value=round(sum(scores) / len(scores), 4),
                                  n=len(scores), dims={"context_len": length}))
        # Effective context: largest rung still at/above threshold.
        thr = 0.66
        good = [ln for ln, sc in by_len.items() if sum(sc) / len(sc) >= thr]
        if good:
            # The rungs measured, not the samples: this figure chooses among rungs, and
            # a count of samples would suggest a precision it does not have.
            metrics.append(Metric(evaluator=self.name, name="effective_ctx",
                                  value=float(max(good)), unit="tokens",
                                  n=len(by_len)))
        return metrics
