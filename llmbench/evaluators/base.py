"""Evaluator base — the contract a test module fulfils.

A module implements one class:

    @register
    class MyEval(Evaluator):
        name = "my_eval"
        version = "1"
        default_config = {...}
        async def evaluate(self, ctx: EvalContext) -> list[Sample]:
            ...

That is the entire surface. The orchestrator handles discovery, running,
aggregation and persistence. Override `aggregate()` only if the default
mean-of-scores / pass-rate summary is not what you want.
"""
from __future__ import annotations

import abc
import inspect
import statistics
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union

from llmbench.models import Metric, ModelFingerprint, Sample
from llmbench.targets.base import GenResult, Target


@dataclass
class EvalContext:
    """Everything an evaluator is handed at run time."""

    target: Target
    fingerprint: ModelFingerprint
    config: dict[str, Any] = field(default_factory=dict)

    async def count_tokens(self, text: str) -> int:
        return await self.target.count_tokens(text)

    async def generate(self, messages, **kw):
        return await self.target.generate(messages, **kw)


@dataclass(frozen=True)
class Verdict:
    """How one response graded — the only part of a result a module has to supply.

    Everything else on a Sample (the six measurements, the failure and no-answer states)
    is the same for every module and is filled in by `Evaluator.run_case`.
    """

    score: Optional[float] = None
    passed: Optional[bool] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Breakdown:
    """A per-category figure, declared by a module instead of looped over by hand.

    Five modules carried a near-identical group-and-average loop, and the differences
    between them were accidents of copying rather than decisions (design E2).

    `metric` is part of the declaration rather than derived from the dimension, because
    the names are load-bearing: `store.py` reads them to decide which figures may be
    pooled across machines, so `accuracy` and `recall` cannot be collapsed into one
    generated name.
    """

    metric: str
    by: tuple[str, ...]
    #: Only `oneshot` differs, and only because it always has. Rounding is not what this
    #: decision is about, so the existing figures keep the shape they had.
    round_to: int = 4


#: A grader turns one response into a verdict. It may be an ordinary function or a
#: coroutine function: `coding` grades by running pytest in a subprocess and must be able
#: to await, while the other nine graders are pure and should not have to pretend to be
#: asynchronous to accommodate it.
Grader = Callable[[GenResult], Union[Verdict, Awaitable[Verdict]]]


class Evaluator(abc.ABC):
    name: str = ""
    version: str = "0"
    default_config: dict[str, Any] = {}
    #: Per-category figures this module wants. Produced by the default aggregator, so a
    #: module that overrides `aggregate` must call `super()` to get them.
    breakdowns: list[Breakdown] = []

    def resolve_config(self, override: dict[str, Any] | None) -> dict[str, Any]:
        cfg = dict(self.default_config)
        if override:
            cfg.update(override)
        return cfg

    @abc.abstractmethod
    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        ...

    async def run_case(self, ctx: EvalContext, *, case_id: str,
                       messages: list[dict[str, str]], grade: Grader,
                       group: Optional[str] = None,
                       dims: Optional[dict[str, Any]] = None,
                       **gen: Any) -> Sample:
        """Ask the model one question and return the finished Sample (design E1).

        Three things happen here that used to happen in every module, by hand:

        1. A failed call becomes a sample carrying the error, not an exception that ends
           the run.
        2. A response that never reached an answer becomes a *skipped* sample carrying
           the reason. It is not graded zero — see `GenResult.unusable_reason`.
        3. Every one of the six measurements is transferred onto the sample.

        Point 3 is why this exists. The transfer was copied into ten places and the
        copies had drifted: `needle` and `coding` recorded the server's own prefill and
        decode speeds, `oneshot` recorded one of them, and the other seven recorded
        neither — so the instruction-following and multiple-choice modules reported no
        server-side speed at all. Nobody decided that; it is what copying produces.

        `grade` is handed the whole `GenResult` rather than its text, because modules
        record more than the answer: `needle` wants `truncated`, `human` and `oneshot`
        store the full response for the dashboard to render.

        A grading exception is deliberately **not** caught. The try below covers the
        model call only, exactly as the hand-written copies did. A defect in our own
        grading code recorded as `error=` on a sample would file a bench bug as a model
        result, and the run would look merely disappointing rather than broken.
        """
        where = {"evaluator": self.name, "case_id": case_id, "group": group,
                 "dims": dims or {}}
        try:
            res = await ctx.generate(messages, **gen)
        except Exception as exc:                      # network, OOM, context overflow
            return Sample(**where, error=repr(exc))
        if res.unusable_reason:
            return Sample(**where, skipped=res.unusable_reason)

        verdict = grade(res)
        if inspect.isawaitable(verdict):
            verdict = await verdict
        return Sample(
            **where, score=verdict.score, passed=verdict.passed, meta=verdict.meta,
            input_tokens=res.input_tokens, output_tokens=res.output_tokens,
            latency_ms=res.latency_ms, tok_per_sec=res.tok_per_sec,
            server_prompt_tps=res.server_prompt_tps, server_gen_tps=res.server_gen_tps,
        )

    # Default aggregation: overall score + pass-rate + throughput + failure counts, plus
    # one figure per category for every `Breakdown` the module declared. Good enough for
    # most modules; override for bespoke metrics (e.g. needle's effective context).
    def aggregate(self, samples: list[Sample]) -> list[Metric]:
        # A skipped sample is neither: it is a rung this machine was never asked to
        # climb (design D3). Averaging it in would put a measurement where there is
        # none, and counting it as an error would call an honest limit a fault.
        graded = [s for s in samples if s.error is None and s.skipped is None]
        metrics: list[Metric] = []

        # Every figure below states the count it was computed from, in the same
        # expression that computes it (design D7a). The counts differ from each other on
        # purpose: a mean rests on the samples that carried a value, and a count of
        # failures rests on every sample there was.
        scores = [s.score for s in graded if s.score is not None]
        if scores:
            metrics.append(Metric(evaluator=self.name, name="score_mean",
                                  value=round(statistics.mean(scores), 4),
                                  n=len(scores)))
        passes = [s.passed for s in graded if s.passed is not None]
        if passes:
            metrics.append(Metric(evaluator=self.name, name="pass_rate",
                                  value=round(sum(passes) / len(passes), 4),
                                  n=len(passes)))
        tps = [s.tok_per_sec for s in graded if s.tok_per_sec]
        if tps:
            metrics.append(Metric(evaluator=self.name, name="tok_per_sec_mean",
                                  value=round(statistics.mean(tps), 2), unit="tok/s",
                                  n=len(tps)))

        # Reported unconditionally, including when nothing was graded. The old early
        # return meant an evaluator whose every sample failed produced no metrics at
        # all, so the one case most in need of explanation explained nothing.
        metrics.append(Metric(evaluator=self.name, name="error_count",
                              value=float(sum(1 for s in samples if s.error is not None)),
                              n=len(samples)))
        metrics.append(Metric(evaluator=self.name, name="skipped_count",
                              value=float(sum(1 for s in samples if s.skipped is not None)),
                              n=len(samples)))

        # Declared per-category figures, last, where the hand-written loops used to run.
        # Each rests on `score`: every module that wants a breakdown sets it on every
        # sample it grades, and a sample carrying no score carries no measurement to
        # average. A sample missing one of the named dimensions is left out rather than
        # filed under a made-up label.
        for spec in self.breakdowns:
            buckets: dict[tuple, list[float]] = {}
            for s in graded:
                if s.score is None or any(k not in s.dims for k in spec.by):
                    continue
                buckets.setdefault(tuple(s.dims[k] for k in spec.by), []).append(s.score)
            for key, values in sorted(buckets.items()):
                metrics.append(Metric(
                    evaluator=self.name, name=spec.metric,
                    value=round(statistics.mean(values), spec.round_to),
                    n=len(values), dims=dict(zip(spec.by, key))))
        return metrics
