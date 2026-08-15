"""Perplexity — exact quant-fidelity via the native llama-perplexity binary.

The HTTP server only returns logprobs for tokens it *generates*, not
teacher-forced logprobs over a fixed text, so true perplexity can't be computed
over the API. But you already build llama.cpp, so this evaluator shells out to
the `llama-perplexity` binary against the *same GGUF* the server reported
(model_path from /props) over a bundled text file, and parses the final PPL.

This is the rawest measure of what a quant costs: run it for F16 and each quant
of the same model and compare. Opt-in — it does nothing unless `binary` is set
and the model path is resolvable locally.

Config:
    binary: path to llama-perplexity (or on PATH)
    text_file: corpus to score (defaults to bundled datasets/ppl_corpus.txt)
    model_path: override; otherwise taken from detection
    ctx: context size for the PPL run
    extra_args: list of extra CLI args (e.g. ["-ngl","99","-fa","on"])
    timeout_s: subprocess timeout
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from llmbench.evaluators.base import EvalContext, Evaluator
from llmbench.models import Metric, Sample
from llmbench.registry import register
from llmbench.resources import resolve_data_file

_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)", re.I)

_DEFAULTS = {
    "binary": None,                         # e.g. "llama-perplexity" or full path
    "text_file": None,                      # None = the bundled corpus
    "model_path": None,
    "ctx": 4096,
    "extra_args": [],
    "timeout_s": 3600,
}


@register
class PerplexityEvaluator(Evaluator):
    name = "perplexity"
    version = "1"
    default_config = _DEFAULTS

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        if not cfg["binary"]:
            # Not configured is not broken. This is an opt-in module, and reporting it
            # as an error put a fault on every healthy run of the default suite.
            return [Sample(evaluator=self.name, case_id="ppl",
                           skipped="not attempted: set `binary` to a llama-perplexity "
                                   "executable to measure quant fidelity")]

        model_path = cfg["model_path"] or (ctx.fingerprint.raw.get("props", {})
                                           .get("model_path"))
        if not model_path or not Path(model_path).exists():
            return [Sample(evaluator=self.name, case_id="ppl",
                           error=f"model gguf not found locally: {model_path!r}")]

        text = resolve_data_file(cfg["text_file"], "datasets", "ppl_corpus.txt")

        args = [cfg["binary"], "-m", model_path, "-f", str(text),
                "-c", str(cfg["ctx"]), *map(str, cfg["extra_args"])]
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=cfg["timeout_s"])
        except asyncio.TimeoutError:
            return [Sample(evaluator=self.name, case_id="ppl", error="timeout")]
        except FileNotFoundError:
            return [Sample(evaluator=self.name, case_id="ppl",
                           error=f"binary not found: {cfg['binary']!r}")]

        ppl = self.parse_ppl(out.decode(errors="replace"))
        if ppl is None:
            return [Sample(evaluator=self.name, case_id="ppl",
                           error="could not parse PPL from output")]
        return [Sample(evaluator=self.name, case_id="ppl", group="ppl",
                       meta={"perplexity": ppl, "model_path": model_path})]

    @staticmethod
    def parse_ppl(text: str):
        m = _PPL_RE.search(text)
        return float(m.group(1)) if m else None

    def aggregate(self, samples: list[Sample]) -> list[Metric]:
        metrics = []
        for s in samples:
            if s.error is None and s.skipped is None and "perplexity" in s.meta:
                # One measurement over the whole configured corpus, so the count is 1.
                # What qualifies this figure is which corpus, and that travels in the
                # sample's meta rather than as a number of items.
                metrics.append(Metric(evaluator=self.name, name="perplexity",
                                      value=round(s.meta["perplexity"], 4), unit="ppl",
                                      n=1))
        metrics.append(Metric(evaluator=self.name, name="error_count",
                              value=float(sum(1 for s in samples if s.error)),
                              n=len(samples)))
        metrics.append(Metric(evaluator=self.name, name="skipped_count",
                              value=float(sum(1 for s in samples if s.skipped)),
                              n=len(samples)))
        return metrics
