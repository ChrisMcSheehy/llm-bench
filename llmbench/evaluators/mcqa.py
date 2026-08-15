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

import json
import string
from typing import Any

from llmbench.evaluators._extract import extract_choice
from llmbench.evaluators.base import EvalContext, Evaluator
from llmbench.models import Metric, Sample
from llmbench.registry import register
from llmbench.resources import resolve_data_file

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

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        items = self._load(cfg)
        out = []
        for it in items:
            out.append(await self._one(ctx, cfg, it))
        return out

    def _load(self, cfg) -> list[dict]:
        p = resolve_data_file(cfg["data_file"], "datasets", "mcqa.jsonl")
        items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        if cfg["limit"]:
            items = items[: cfg["limit"]]
        return items

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
        dims = {"subject": it.get("subject", "all")}
        try:
            res = await ctx.generate([{"role": "user", "content": prompt}],
                                     max_tokens=cfg["max_tokens"] if not cfg["cot"] else 512,
                                     temperature=cfg["temperature"])
        except Exception as e:
            return Sample(evaluator=self.name, case_id=str(it["id"]), group=dims["subject"],
                          dims=dims, error=repr(e))
        if res.unusable_reason:
            return Sample(evaluator=self.name, case_id=str(it["id"]), group=dims["subject"],
                          dims=dims, skipped=res.unusable_reason)
        pred = extract_choice(res.text, len(opts))
        ok = pred == gold
        return Sample(evaluator=self.name, case_id=str(it["id"]), group=dims["subject"],
                      dims=dims, score=1.0 if ok else 0.0, passed=ok,
                      input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                      latency_ms=res.latency_ms, tok_per_sec=res.tok_per_sec,
                      meta={"pred": pred, "gold": gold, "answer": res.text[:120]})

    def aggregate(self, samples: list[Sample]) -> list[Metric]:
        metrics = super().aggregate(samples)
        graded = [s for s in samples if s.error is None and s.passed is not None]
        by_sub: dict[str, list[bool]] = {}
        for s in graded:
            by_sub.setdefault(s.dims.get("subject", "all"), []).append(bool(s.passed))
        for sub, v in sorted(by_sub.items()):
            metrics.append(Metric(evaluator=self.name, name="accuracy",
                                  value=round(sum(v) / len(v), 4), n=len(v),
                                  dims={"subject": sub}))
        return metrics
