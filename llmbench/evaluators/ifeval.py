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
from llmbench.evaluators.base import EvalContext, Evaluator
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
            try:
                res = await ctx.generate([{"role": "user", "content": prompt}],
                                         max_tokens=cfg["max_tokens"],
                                         temperature=cfg["temperature"])
            except Exception as e:
                out.append(Sample(evaluator=self.name, case_id=tid, error=repr(e)))
                continue
            if res.unusable_reason:
                out.append(Sample(evaluator=self.name, case_id=tid,
                                  skipped=res.unusable_reason))
                continue
            results = [(label, bool(chk(res.text))) for label, chk in checks]
            met = sum(1 for _, ok in results if ok)
            all_ok = met == len(results)
            out.append(Sample(
                evaluator=self.name, case_id=tid, group=tid,
                dims={"n_constraints": len(results)},
                score=met / len(results), passed=all_ok,
                input_tokens=res.input_tokens, output_tokens=res.output_tokens,
                latency_ms=res.latency_ms, tok_per_sec=res.tok_per_sec,
                meta={"checks": dict(results), "answer": res.text[:200]}))
        return out

    def aggregate(self, samples: list[Sample]) -> list[Metric]:
        # Skips excluded as well as errors: a skipped sample carries score None, which
        # this sum would add to a float. The default aggregator already draws this
        # distinction; every override has to draw it too.
        graded = [s for s in samples if s.error is None and s.skipped is None]
        metrics = []
        if graded:
            inst = sum(s.score for s in graded) / len(graded)
            prompt = sum(1 for s in graded if s.passed) / len(graded)
            metrics.append(Metric(evaluator=self.name, name="instruction_acc",
                                  value=round(inst, 4), n=len(graded)))
            metrics.append(Metric(evaluator=self.name, name="prompt_acc",
                                  value=round(prompt, 4), n=len(graded)))
            # score_mean mirrors instruction_acc so the leaderboard picks it up.
            metrics.append(Metric(evaluator=self.name, name="score_mean",
                                  value=round(inst, 4), n=len(graded)))
        metrics.append(Metric(evaluator=self.name, name="error_count",
                              value=float(sum(1 for s in samples if s.error)),
                              n=len(samples)))
        return metrics
