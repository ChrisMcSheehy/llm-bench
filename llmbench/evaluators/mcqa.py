"""Multiple-choice QA — one harness for the whole family.

MMLU, ARC-C/E, HellaSwag, TruthfulQA (MC), GPQA, CommonsenseQA, OpenBookQA,
WMDP — they're all "question + lettered options -> pick the letter". Point
`data_file` at a JSONL of items and set `subject` labels; grading is exact
letter match, so it's fully deterministic and quant-sensitive.

JSONL item schema (one per line):
    {"id": "...", "question": "...", "options": ["...", "..."],
     "answer": "B", "subject": "physics"}      # answer is a letter or 0-based index

A tiny original sample set is bundled so the module runs with no download; drop
in real datasets (converted to this schema) for real numbers.
"""
from __future__ import annotations

import string

from llmbench.evaluators._extract import extract_choice
from llmbench.evaluators.base import Breakdown, EvalContext, Evaluator, Verdict, View
from llmbench.models import Sample
from llmbench.registry import register
from llmbench.resources import load_jsonl

_DEFAULTS = {
    "data_file": None,          # None = the bundled sample set
    "limit": None,
    "max_tokens": 8,          # letter-only; keep tight
    "temperature": 0.0,
    "cot": False,             # if True, allow reasoning then a final letter
}


@register
class MCQAEvaluator(Evaluator):
    name = "mcqa"
    version = "1"
    default_config = _DEFAULTS
    breakdowns = [Breakdown("accuracy", ("subject",))]
    views = [View("bar", "accuracy by subject", x="subject")]

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        items = load_jsonl(cfg["data_file"], "datasets", "mcqa.jsonl", limit=cfg["limit"])
        out = []
        for it in items:
            out.append(await self._one(ctx, cfg, it))
        return out

    async def _one(self, ctx, cfg, it) -> Sample:
        opts = it["options"]
        letters = string.ascii_uppercase[: len(opts)]
        gold = it["answer"]
        gold = letters[gold] if isinstance(gold, int) else str(gold).strip().upper()
        block = "\n".join(f"{l}. {o}" for l, o in zip(letters, opts))
        instr = ("Answer with the single letter of the correct option."
                 if not cfg["cot"] else
                 "Think briefly, then end with 'Answer: <letter>'.")
        prompt = f"{it['question']}\n\n{block}\n\n{instr}"
        subject = it.get("subject", "all")

        def grade(res) -> Verdict:
            pred = extract_choice(res.text, len(opts))
            ok = pred == gold
            return Verdict(score=1.0 if ok else 0.0, passed=ok,
                           meta={"pred": pred, "gold": gold, "answer": res.text[:120]})

        return await self.run_case(
            ctx, case_id=str(it["id"]), group=subject, dims={"subject": subject},
            messages=[{"role": "user", "content": prompt}], grade=grade,
            max_tokens=cfg["max_tokens"] if not cfg["cot"] else 512,
            temperature=cfg["temperature"])
