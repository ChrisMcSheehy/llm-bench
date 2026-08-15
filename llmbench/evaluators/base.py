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
import statistics
from dataclasses import dataclass, field
from typing import Any

from llmbench.models import Metric, ModelFingerprint, Sample
from llmbench.targets.base import Target


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


class Evaluator(abc.ABC):
    name: str = ""
    version: str = "0"
    default_config: dict[str, Any] = {}

    def resolve_config(self, override: dict[str, Any] | None) -> dict[str, Any]:
        cfg = dict(self.default_config)
        if override:
            cfg.update(override)
        return cfg

    @abc.abstractmethod
    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        ...

    # Default aggregation: overall score + pass-rate + throughput, plus the same
    # broken down by every dimension key present on the samples. Good enough for
    # most modules; override for bespoke metrics (e.g. needle's heatmap cells).
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
        return metrics
