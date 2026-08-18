"""Instruction following with programmatically verifiable constraints.

IFEval / COLLIE style: each task pairs a prompt with constraints that can be
checked in code (no judge). We report instruction-level accuracy (fraction of
individual constraints met) and prompt-level accuracy (all constraints in a
prompt met) — the two headline IFEval numbers. Format adherence is one of the
first things aggressive quantisation breaks, so this is a sensitive probe.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from llmbench.evaluators._extract import first_json
from llmbench.evaluators.base import EvalContext, Evaluator, Verdict
from llmbench.models import Metric, Sample
from llmbench.registry import register

Check = Callable[[str], bool]


def _bullets(n: int) -> Check:
    return lambda t: len([ln for ln in t.splitlines()
                          if ln.strip().startswith(("-", "*", "•"))]) == n


def _word_count(lo: int, hi: int) -> Check:
    return lambda t: lo <= len(re.findall(r"\b\w+\b", t)) <= hi


def _keyword_times(word: str, k: int) -> Check:
    return lambda t: len(re.findall(rf"\b{re.escape(word)}\b", t, re.I)) == k


def _ends_with(phrase: str) -> Check:
    return lambda t: t.strip().rstrip('"').endswith(phrase)


def _no_commas() -> Check:
    return lambda t: "," not in t


def _all_lower() -> Check:
    return lambda t: t.strip() == t.strip().lower() and any(c.isalpha() for c in t)


def _valid_json() -> Check:
    return lambda t: first_json(t) is not None


def _n_paragraphs(n: int) -> Check:
    return lambda t: len([p for p in re.split(r"\n\s*\n", t.strip()) if p.strip()]) == n


def _title_wrapped() -> Check:
    return lambda t: bool(re.search(r"<<[^>]+>>", t))


# (id, prompt, [(label, check), ...])
_TASKS = [
    ("bullets3",
     "List three benefits of unit testing. Reply with exactly three bullet points, each starting with '-'.",
     [("exactly 3 bullets", _bullets(3))]),
    ("wc_range",
     "Describe what a data warehouse is in between 20 and 40 words.",
     [("20-40 words", _word_count(20, 40))]),
    ("keyword",
     "Write two sentences about coffee. Use the word 'aroma' exactly twice.",
     [("'aroma' x2", _keyword_times("aroma", 2))]),
    ("endphrase",
     "Explain what an index does in a database, then end your reply with exactly: END OF ANSWER",
     [("ends with phrase", _ends_with("END OF ANSWER"))]),
    ("nocomma_lower",
     "Write a single lowercase sentence about the sea that contains no commas.",
     [("all lowercase", _all_lower()), ("no commas", _no_commas())]),
    ("json_obj",
     'Return a JSON object with keys "name" and "age" for a fictional person. Output only JSON.',
     [("valid json", _valid_json())]),
    ("paras2",
     "Write about renewable energy in exactly two paragraphs separated by a blank line.",
     [("exactly 2 paragraphs", _n_paragraphs(2))]),
    ("title",
     "Suggest a title for a talk on Snowflake and wrap the title in double angle brackets like <<Title>>.",
     [("title in <<>>", _title_wrapped())]),
]

_DEFAULTS = {"max_tokens": 400, "temperature": 0.0}


@register
class IFEvalEvaluator(Evaluator):
    name = "ifeval"
    version = "1"
    default_config = _DEFAULTS

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        out = []
        for tid, prompt, checks in _TASKS:
            def grade(res, checks=checks) -> Verdict:
                results = [(label, bool(chk(res.text))) for label, chk in checks]
                met = sum(1 for _, ok in results if ok)
                return Verdict(score=met / len(results), passed=met == len(results),
                               meta={"checks": dict(results), "answer": res.text[:200]})

            out.append(await self.run_case(
                ctx, case_id=tid, group=tid, dims={"n_constraints": len(checks)},
                messages=[{"role": "user", "content": prompt}], grade=grade,
                max_tokens=cfg["max_tokens"], temperature=cfg["temperature"]))
        return out

    def aggregate(self, samples: list[Sample]) -> list[Metric]:
        """IFEval's two published figures, which the default aggregator already computes.

        `instruction_acc` is the mean score — the fraction of individual constraints met —
        and `prompt_acc` is the pass rate, every constraint in a prompt met. They are
        emitted under IFEval's own names as well, because those are what a reader
        comparing against published IFEval numbers will look for.

        This used to compute both by hand and skipped `super()`, which meant it silently
        reported no `skipped_count` and, once it existed, no `answer_rate` — a module that
        opts out of the shared aggregator opts out of everything added to it later.
        """
        metrics = super().aggregate(samples)
        published = {"instruction_acc": "score_mean", "prompt_acc": "pass_rate"}
        by_name = {m.name: m for m in metrics}
        for alias, computed in published.items():
            source = by_name.get(computed)
            if source is not None:
                metrics.append(Metric(evaluator=self.name, name=alias,
                                      value=source.value, n=source.n,
                                      successes=source.successes))
        return metrics
