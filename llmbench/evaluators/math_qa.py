"""Math word problems — GSM8K / MATH style, numeric-answer grading.

The model may reason freely; we extract the final answer via \\boxed{...},
'#### N', or the last number, and compare numerically. Deterministic and
strongly quant-sensitive (arithmetic reasoning degrades early under aggressive
quantisation).

JSONL item schema:
    {"id": "...", "question": "...", "answer": "42", "type": "gsm8k"}
"""
from __future__ import annotations

from llmbench.evaluators._extract import extract_final_number, numbers_equal
from llmbench.evaluators.base import Breakdown, EvalContext, Evaluator, Verdict, View
from llmbench.models import Sample
from llmbench.registry import register
from llmbench.resources import load_jsonl

_DEFAULTS = {
    "data_file": None,          # None = the bundled sample set
    "limit": None,
    "max_tokens": 512,
    "temperature": 0.0,
    "cot": True,
}


@register
class MathEvaluator(Evaluator):
    name = "math_qa"
    version = "1"
    default_config = _DEFAULTS
    breakdowns = [Breakdown("accuracy", ("type",))]
    views = [View("bar", "accuracy by problem type", x="type")]

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        items = load_jsonl(cfg["data_file"], "datasets", "math.jsonl", limit=cfg["limit"])
        return [await self._one(ctx, cfg, it) for it in items]

    async def _one(self, ctx, cfg, it) -> Sample:
        suffix = ("\n\nReason step by step, then give the final answer as "
                  "\\boxed{answer}." if cfg["cot"] else
                  "\n\nReply with only the final number.")
        kind = it.get("type", "math")

        def grade(res) -> Verdict:
            pred = extract_final_number(res.text)
            ok = numbers_equal(pred, str(it["answer"]))
            return Verdict(score=1.0 if ok else 0.0, passed=ok,
                           meta={"pred": pred, "gold": str(it["answer"]),
                                 "answer": res.text[-160:]})

        return await self.run_case(
            ctx, case_id=str(it["id"]), group=kind, dims={"type": kind},
            messages=[{"role": "user", "content": it["question"] + suffix}], grade=grade,
            max_tokens=cfg["max_tokens"], temperature=cfg["temperature"])
