"""Math word problems — GSM8K / MATH style, numeric-answer grading.

The model may reason freely; we extract the final answer via \\boxed{...},
'#### N', or the last number, and compare numerically. Deterministic and
strongly quant-sensitive (arithmetic reasoning degrades early under aggressive
quantisation).

JSONL item schema:
    {"id": "...", "question": "...", "answer": "42", "type": "gsm8k"}
"""
from __future__ import annotations

import json

from llmbench.evaluators._extract import extract_final_number, numbers_equal
from llmbench.evaluators.base import EvalContext, Evaluator
from llmbench.models import Metric, Sample
from llmbench.registry import register
from llmbench.resources import resolve_data_file

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

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        p = resolve_data_file(cfg["data_file"], "datasets", "math.jsonl")
        items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        if cfg["limit"]:
            items = items[: cfg["limit"]]
        return [await self._one(ctx, cfg, it) for it in items]

    async def _one(self, ctx, cfg, it) -> Sample:
        suffix = ("\n\nReason step by step, then give the final answer as "
                  "\\boxed{answer}." if cfg["cot"] else
                  "\n\nReply with only the final number.")
        dims = {"type": it.get("type", "math")}
        try:
            res = await ctx.generate(
                [{"role": "user", "content": it["question"] + suffix}],
                max_tokens=cfg["max_tokens"], temperature=cfg["temperature"])
        except Exception as e:
            return Sample(evaluator=self.name, case_id=str(it["id"]), group=dims["type"],
                          dims=dims, error=repr(e))
        if res.unusable_reason:
            return Sample(evaluator=self.name, case_id=str(it["id"]), group=dims["type"],
                          dims=dims, skipped=res.unusable_reason)
        pred = extract_final_number(res.text)
        ok = numbers_equal(pred, str(it["answer"]))
        return Sample(evaluator=self.name, case_id=str(it["id"]), group=dims["type"],
                      dims=dims, score=1.0 if ok else 0.0, passed=ok,
                      input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                      latency_ms=res.latency_ms, tok_per_sec=res.tok_per_sec,
                      meta={"pred": pred, "gold": str(it["answer"]), "answer": res.text[-160:]})

    def aggregate(self, samples: list[Sample]) -> list[Metric]:
        metrics = super().aggregate(samples)
        graded = [s for s in samples if s.error is None and s.passed is not None]
        by_t: dict[str, list[bool]] = {}
        for s in graded:
            by_t.setdefault(s.dims.get("type", "math"), []).append(bool(s.passed))
        for t, v in sorted(by_t.items()):
            metrics.append(Metric(evaluator=self.name, name="accuracy",
                                  value=round(sum(v) / len(v), 4), n=len(v),
                                  dims={"type": t}))
        return metrics
