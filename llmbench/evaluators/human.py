"""Human evaluation — generation side.

Auto-graders can't judge writing quality, coherence, helpfulness, or tone. This
evaluator doesn't score anything: it generates one response per arena prompt and
stores it (as a sample with the text in meta) so the *dashboard arena* can serve
blind A/B comparisons between configs and compute an Elo leaderboard from your
votes.

Run a suite with `human` enabled across the configs you want to compare (each
becomes a distinct fingerprint), then open the dashboard and go to Arena.

Sampling defaults to temperature 0.7 — human comparisons should see the model's
natural voice, not greedy output.
"""
from __future__ import annotations

from llmbench.evaluators.base import EvalContext, Evaluator, Verdict
from llmbench.models import Metric, Sample
from llmbench.registry import register
from llmbench.resources import load_jsonl

_DEFAULTS = {
    "data_file": None,          # None = the bundled arena prompts
    "limit": None,
    "max_tokens": 700,
    "temperature": 0.7,
}


@register
class HumanEvalEvaluator(Evaluator):
    name = "human"
    version = "1"
    default_config = _DEFAULTS

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        items = load_jsonl(cfg["data_file"], "datasets", "arena_prompts.jsonl",
                           limit=cfg["limit"])

        out: list[Sample] = []
        for it in items:
            cat = it.get("category", "general")

            # The verdict carries no score and no pass: this module stores a response for
            # a person to judge, and a heuristic standing in for that judgement is exactly
            # what it exists to avoid.
            def grade(res, it=it, cat=cat) -> Verdict:
                return Verdict(meta={"prompt_id": it["id"], "prompt": it["prompt"],
                                     "category": cat, "response": res.text})

            out.append(await self.run_case(
                ctx, case_id=it["id"], group=cat,
                dims={"category": cat, "prompt_id": it["id"]},
                messages=[{"role": "user", "content": it["prompt"]}], grade=grade,
                max_tokens=cfg["max_tokens"], temperature=cfg["temperature"]))
        return out

    def aggregate(self, samples: list[Sample]) -> list[Metric]:
        # A response that never arrived is not a response to rate, so it is not counted
        # as one - the same distinction the default aggregator draws.
        graded = [s for s in samples if s.error is None and s.skipped is None]
        return [
            # "12 responses out of 12 prompts" - the count is the prompts attempted,
            # which is what makes the figure readable at all.
            Metric(evaluator=self.name, name="responses", value=float(len(graded)),
                   n=len(samples)),
            Metric(evaluator=self.name, name="error_count",
                   value=float(sum(1 for s in samples if s.error)), n=len(samples)),
        ]
