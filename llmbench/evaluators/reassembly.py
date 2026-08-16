"""Find three fragments across a long document and put them back together (design B5).

The third user of `_ladder.py`, which `ARCHITECTURE.md` anticipated: *"A third ladder
evaluator adds no rules."*

**What is genuinely new here, stated honestly, because the overlap is larger than it
looks.** `long_context` already plants K near-identical facts among distractors and
already follows a chain of assignments across distance. So "multi-hop retrieval under
distractors" is *not* new and is not the reason this exists. Three things are absent:

1. **Retrieving several items and emitting all of them at once.** `multikey` plants K
   facts and asks for one; `vartrack` follows a chain and returns one value. Neither asks
   the model to hold several retrieved items simultaneously and reproduce them together,
   which is a capacity question rather than a retrieval question.
2. **Assembly as a separately scored step.** The order can be wrong while the retrieval
   was perfect, and no existing evaluator can express that difference.
3. **Resolution below pass/fail.** `long_context` and `needle` both grade 1.0 or 0.0. At
   the rung where a configuration begins to degrade, a step function says a line was
   crossed and nothing about how far. The central question of this project is *how much*
   quality a compression setting costs, and every long-context instrument answers it in
   whole numbers.

Put plainly: the existing evaluators tell you *that* a configuration broke at 256k. This
one is built to tell you *how badly*.

**The key is hexadecimal**, and that is load-bearing. Hex maps exactly four bits per
character, which is what makes a bit-level comparison meaningful rather than approximate.
Base64 carries visually confusable characters (`l`/`I`/`1`, `O`/`0`) and mixed case, so it
would measure transcription luck as much as recall.

**The key is generated from a seed, never taken from anything published.** A real public
key may sit in the model's training data, and a model reciting a memorised key would score
perfectly while retrieving nothing. Generating it makes it certainly unseen; seeding it
makes it reproducible, which every comparison in this project depends on. A distinct key
per cell also means a server reusing a cache between calls cannot carry an answer forward.

**A returned key of the wrong length reports bit accuracy as unknown, never as a number.**
Comparing bits between strings of different lengths measures alignment, not recall, and
would produce a plausible-looking ~50% out of what is actually a structural failure.

**Deliberately not in version one: decoy fragments.** They belong here eventually, for the
same reason the agency suite scores focus. But a v1 without them means a low score has
exactly one possible cause, and that is worth more than breadth while the evaluator is new.
"""
from __future__ import annotations

import random
import re
from typing import Any, Optional

from llmbench.evaluators._ladder import climb, context_ladder
from llmbench.evaluators._sizing import CALIBRATION_PROBE, build_filler, chars_per_token
from llmbench.evaluators.base import Breakdown, EvalContext, Evaluator, Verdict, View
from llmbench.models import Metric, Sample
from llmbench.registry import register

_LABELS = ("ALPHA", "BETA", "GAMMA")
_HEX = "0123456789abcdef"

_DEFAULTS = {
    "context_lengths": None,
    "ladder_knots": [8192, 32768, 131072, 262144],
    "max_rungs": 4,
    # 48 hex characters is 192 bits in three parts of 16. Long enough that bit accuracy
    # has resolution, short enough that a model has room to reproduce it.
    "key_chars": 48,
    "depths": [10, 50, 90],           # where the three parts sit, in % of the document
    "repeats": 1,
    # Generous, and stated here rather than buried, because a reassembled key plus any
    # reasoning is several hundred tokens of high-entropy text. The `unusable_response`
    # machinery correctly refuses to score a truncated answer - but a benchmark that
    # routinely truncates is measuring its own configuration rather than the model.
    "answer_tokens": 512,
    "overhead_tokens": 512,
    "time_budget_s": 1800,
    "seed": 13,
}


def split_key(key: str, parts: int = 3) -> list[str]:
    """The key in equal parts. Equal so that no part is a smaller target than another."""
    size = len(key) // parts
    return [key[i * size:(i + 1) * size] for i in range(parts)]


def bit_accuracy(expected: str, got: str) -> Optional[float]:
    """Fraction of matching bits, or None when the lengths differ.

    None rather than a number is the whole point. Padding or truncating to compare would
    always produce a figure, and the figure would describe alignment rather than recall -
    a structural failure dressed up as roughly half marks.
    """
    if len(expected) != len(got):
        return None
    try:
        difference = int(expected, 16) ^ int(got, 16)
    except ValueError:
        return None                       # not hexadecimal at all
    total = len(expected) * 4             # hex is exactly four bits per character
    return (total - bin(difference).count("1")) / total


@register
class ReassemblyEvaluator(Evaluator):
    name = "reassembly"
    version = "1"
    default_config = _DEFAULTS
    #: Bit accuracy per rung — the gradient this evaluator exists to produce. `recall` is
    #: the name the other two ladder evaluators use, so the dashboard and the pooling
    #: rules already know it.
    breakdowns = [Breakdown("recall", ("context_len",))]
    #: A line, because bit accuracy is a gradient - the shape of the decline is the
    #: finding, and bars would hide it behind their own spacing.
    views = [View("line", "bit accuracy by context length", x="context_len")]

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        rng = random.Random(cfg["seed"])
        n_ctx = ctx.fingerprint.n_ctx or 8192
        lengths = (sorted(cfg["context_lengths"]) if cfg["context_lengths"] else
                   context_ladder(n_ctx, cfg["ladder_knots"], cfg["max_rungs"]))
        cpt = await chars_per_token(ctx, CALIBRATION_PROBE)

        async def rung(length: int) -> list[Sample]:
            budget = max(512, length - cfg["overhead_tokens"] - cfg["answer_tokens"])
            return [await self._one(ctx, cfg, rng, length, int(budget * cpt), rep)
                    for rep in range(cfg["repeats"])]

        # The same climb as `needle` and `long_context`, for the same reason (design D3).
        return await climb(lengths, rung, self._skipped, cfg["time_budget_s"])

    def _skipped(self, length: int, reason: str) -> Sample:
        return Sample(evaluator=self.name, case_id=f"{length}:skipped",
                      group=str(length), dims={"context_len": length}, skipped=reason)

    def _document(self, rng: random.Random, char_budget: int, parts: list[str],
                  depths: list[int]) -> str:
        """Filler with each labelled part placed at its depth, deepest first.

        Inserting from the end backwards keeps every earlier offset valid; inserting
        forwards would shift each subsequent position by the length already added, and
        the parts would drift towards the front by an amount nobody chose.
        """
        text = build_filler(rng, char_budget)
        for label, part, depth in sorted(zip(_LABELS, parts, depths),
                                         key=lambda t: t[2], reverse=True):
            cut = int(len(text) * (depth / 100.0))
            space = text.rfind(" ", 0, cut) if cut else 0
            cut = space if space > 0 else cut
            fragment = f" FRAGMENT {label}: {part} . "
            text = text[:cut] + fragment + text[cut:]
        return text

    async def _one(self, ctx, cfg, rng, length, char_budget, rep) -> Sample:
        # A distinct key per cell: one key reused across rungs could be carried forward by
        # a server reusing its cache between calls, and the run would measure that instead.
        key = "".join(rng.choice(_HEX) for _ in range(cfg["key_chars"]))
        parts = split_key(key)
        document = self._document(rng, char_budget, parts, cfg["depths"])

        system = ("You are given a long document containing three labelled fragments of a "
                  "single hexadecimal key.")
        user = (f"{document}\n\n"
                f"The document contains FRAGMENT ALPHA, FRAGMENT BETA and FRAGMENT GAMMA. "
                f"Find all three and reply with the complete key, being ALPHA followed by "
                f"BETA followed by GAMMA, joined together with no spaces and nothing else.")

        def grade(res) -> Verdict:
            answer = self._extract(res.text, len(key))
            found = sum(1 for p in parts if p.lower() in res.text.lower())
            accuracy = bit_accuracy(key, answer) if answer else None
            exact = answer == key
            return Verdict(
                # The score IS the bit accuracy, so the per-rung breakdown is the gradient
                # rather than another step function. Unknown stays None: a wrong-length
                # answer is a real failure, counted by `exact_match` below and by
                # `pass_rate`, but it carries no bit measurement to average.
                score=accuracy,
                passed=exact,
                meta={"parts_found": found, "order_correct": self._ordered(res.text, parts),
                      "bit_accuracy": accuracy, "exact_match": exact,
                      "expected": key, "answer": answer or res.text[:120]})

        return await self.run_case(
            ctx, case_id=f"{length}:{rep}", group=str(length),
            dims={"context_len": length, "rep": rep},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            grade=grade, max_tokens=cfg["answer_tokens"], temperature=0.0)

    def _extract(self, text: str, expected_len: int) -> str:
        """The longest run of hex characters the model offered.

        Longest rather than first because a reasoning model narrates - "ALPHA is 1a2b..."
        - and the assembled key is the longest such run. Deliberately not trimmed to the
        expected length: a wrong length is a finding, and trimming would hide it.
        """
        runs = re.findall(rf"[{_HEX}]{{4,}}", text.lower())
        return max(runs, key=len) if runs else ""

    def _ordered(self, text: str, parts: list[str]) -> bool:
        """Whether the three parts appear in the answer in the right order.

        Isolates assembly from retrieval: order can be wrong while every part was found,
        and that is a different finding from having missed one.
        """
        lowered = text.lower()
        positions = [lowered.find(p.lower()) for p in parts]
        return all(p >= 0 for p in positions) and positions == sorted(positions)

    def aggregate(self, samples: list[Sample]) -> list[Metric]:
        """The three tiers below bit accuracy, each resting on its own count.

        A low bit accuracy can then be explained rather than merely observed: "found two
        of three" and "found all three and mistyped one character" are the same
        `exact_match` of zero and completely different findings about the configuration.
        """
        metrics = super().aggregate(samples)
        graded = [s for s in samples if s.error is None and s.skipped is None]
        if not graded:
            return metrics

        tiers: list[tuple[str, Any, Optional[str]]] = [
            ("parts_found", lambda s: s.meta.get("parts_found"), "of 3"),
            ("order_correct", lambda s: _as_number(s.meta.get("order_correct")), None),
            ("exact_match", lambda s: _as_number(s.meta.get("exact_match")), None),
        ]
        for name, read, unit in tiers:
            values = [v for v in (read(s) for s in graded) if v is not None]
            if values:
                metrics.append(Metric(evaluator=self.name, name=name,
                                      value=round(sum(values) / len(values), 4),
                                      unit=unit, n=len(values)))
        return metrics


def _as_number(value: Optional[bool]) -> Optional[float]:
    return None if value is None else float(value)
