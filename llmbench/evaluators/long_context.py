"""Long-context stress tests beyond single-needle (RULER / MRCR flavoured).

Two synthetic sub-tasks, both scaling across the same context ladder as the
needle test:

  multikey  — K distinct "code for <word> is <CODE>" facts are planted among
              filler; the model must return the code for one queried word while
              K-1 near-identical distractors sit in context. This is where
              KV-cache quantisation (TurboQuant) tends to break down first.
  vartrack  — a chain of variable assignments (v0 = N; v1 = v0; ...) is spread
              through the context; the model must report the final value. Tests
              multi-hop tracking across distance, not just verbatim retrieval.

Both are graded deterministically. Set context lengths explicitly or let them
auto-derive from detected n_ctx.
"""
from __future__ import annotations

import random
import re
from typing import Any

from llmbench.evaluators._extract import extract_final_number, numbers_equal
from llmbench.evaluators.base import Breakdown, EvalContext, Evaluator, Verdict
from llmbench.evaluators._ladder import climb, context_ladder
from llmbench.models import Sample
from llmbench.registry import register

_WORDS = ["amber", "quartz", "cobalt", "verdant", "saffron", "cinder", "harbor",
          "lantern", "meridian", "thistle", "fathom", "gable", "orchard", "vellum"]
_FILLER = ("The archivists of the northern reaches kept meticulous ledgers of "
           "grain, cloth, and lamp oil through every quiet season. ")

_DEFAULTS = {
    "context_lengths": None,
    "ladder_knots": [8192, 32768, 131072, 262144],
    "max_rungs": 4,
    "tasks": ["multikey", "vartrack"],
    "n_keys": 8,            # multikey: distractor count
    "n_hops": 12,           # vartrack: chain length
    "queries_per_rung": 3,
    # Thirty minutes per ladder, as in `needle`. See design D3c.
    "time_budget_s": 1800,
    "answer_tokens": 24,
    "overhead_tokens": 256,
    "seed": 11,
}


@register
class LongContextEvaluator(Evaluator):
    name = "long_context"
    version = "1"
    default_config = _DEFAULTS
    # Recall per (task, rung): the two sub-tasks degrade at different distances, and one
    # figure covering both would hide which of them broke.
    breakdowns = [Breakdown("recall", ("task", "context_len"))]

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        rng = random.Random(cfg["seed"])
        n_ctx = ctx.fingerprint.n_ctx or 8192
        lengths = (sorted(cfg["context_lengths"]) if cfg["context_lengths"] else
                   context_ladder(n_ctx, cfg["ladder_knots"], cfg["max_rungs"]))
        cpt = await self._cpt(ctx)

        async def rung(length: int) -> list[Sample]:
            budget = max(512, length - cfg["overhead_tokens"] - cfg["answer_tokens"])
            char_budget = int(budget * cpt)
            out: list[Sample] = []
            for q in range(cfg["queries_per_rung"]):
                if "multikey" in cfg["tasks"]:
                    out.append(await self._multikey(ctx, cfg, rng, length, char_budget, q))
                if "vartrack" in cfg["tasks"]:
                    out.append(await self._vartrack(ctx, cfg, rng, length, char_budget, q))
            return out

        # The same climb as `needle`, for the same reason (design D3).
        return await climb(lengths, rung, self._skipped, cfg["time_budget_s"])

    def _skipped(self, length: int, reason: str) -> Sample:
        """One row for a rung nobody climbed, whichever tasks it would have run."""
        return Sample(evaluator=self.name, case_id=f"{length}:skipped",
                      dims={"context_len": length}, skipped=reason)

    async def _cpt(self, ctx: EvalContext) -> float:
        toks = await ctx.count_tokens(_FILLER * 10)
        return len(_FILLER * 10) / max(1, toks)

    def _weave(self, rng, char_budget: int, inserts: list[str]) -> str:
        """Spread `inserts` roughly evenly through filler of ~char_budget."""
        n = len(inserts)
        seg = max(1, char_budget // (n + 1))
        chunks, used, i = [], 0, 0
        while used < char_budget:
            chunks.append(_FILLER)
            used += len(_FILLER)
            if i < n and used >= seg * (i + 1):
                chunks.append(inserts[i])
                i += 1
        for j in range(i, n):          # flush any remaining inserts
            chunks.append(inserts[j])
        return " ".join(chunks)

    async def _multikey(self, ctx, cfg, rng, length, char_budget, qi) -> Sample:
        words = rng.sample(_WORDS, cfg["n_keys"])
        codes = {w: f"{rng.choice(_WORDS).upper()}-{rng.randint(1000, 9999)}" for w in words}
        inserts = [f" Note: the code for {w} is {c}. " for w, c in codes.items()]
        rng.shuffle(inserts)
        target = rng.choice(words)
        body = self._weave(rng, char_budget, inserts)
        prompt = (f"{body}\n\nQuestion: what is the code for {target}? "
                  f"Reply with only the code.")
        def grade(res) -> Verdict:
            ok = codes[target].lower() in re.sub(r"\s+", " ", res.text).lower()
            return Verdict(score=1.0 if ok else 0.0, passed=ok,
                           meta={"gold": codes[target], "answer": res.text[:120]})

        return await self.run_case(
            ctx, case_id=f"mk:{length}:{qi}", group="multikey",
            dims={"context_len": length, "task": "multikey"},
            messages=[{"role": "user", "content": prompt}], grade=grade,
            max_tokens=cfg["answer_tokens"], temperature=0.0)

    async def _vartrack(self, ctx, cfg, rng, length, char_budget, qi) -> Sample:
        start = rng.randint(10000, 99999)
        hops = cfg["n_hops"]
        names = [f"v{n}" for n in range(hops)]
        inserts = [f" Let {names[0]} = {start}. "]
        for k in range(1, hops):
            inserts.append(f" Let {names[k]} = {names[k-1]}. ")
        rng.shuffle(inserts)
        body = self._weave(rng, char_budget, inserts)
        prompt = (f"{body}\n\nThe variables are chained by equality. "
                  f"What is the numeric value of {names[-1]}? Reply with only the number.")
        def grade(res) -> Verdict:
            ok = numbers_equal(extract_final_number(res.text), str(start))
            return Verdict(score=1.0 if ok else 0.0, passed=ok,
                           meta={"gold": start, "answer": res.text[:120]})

        return await self.run_case(
            ctx, case_id=f"vt:{length}:{qi}", group="vartrack",
            dims={"context_len": length, "task": "vartrack"},
            messages=[{"role": "user", "content": prompt}], grade=grade,
            max_tokens=cfg["answer_tokens"], temperature=0.0)
