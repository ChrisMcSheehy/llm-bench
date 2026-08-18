"""Speed, as the two figures it actually is (design B3).

A model server does two different jobs. It **reads** the prompt — one pass over every
token at once, limited by how fast the machine can compute — and then it **writes** the
answer, one token at a time, limited by how fast it can move the weights through memory.
Those two speeds have different bottlenecks, they respond to different settings, and a
single number blending them describes neither.

The bench used to publish exactly such a blend: output tokens divided by total wall-clock
time, which includes reading the prompt, then averaged across every rung of a context
ladder — so the headline speed depended most on the one variable it hid. The right
figures were already arriving from llama.cpp's own `timings` block and nothing read them.

This module reports them separately, at a few deliberately-chosen prompt sizes:

    decode        ~64 in, 256 out    writing, uncontaminated by reading
    prefill 512   ~512 in, 1 out     reading, short
    prefill 2k    ~2048 in, 1 out    reading, medium
    prefill 8k    ~8192 in, 1 out    reading, long
    combined 4k   ~4096 in, 128 out  a realistic mixed workload

**Prefill scenarios generate exactly one token and report no writing speed at all.** One
token's worth of timing is a sub-millisecond measurement of a process that has barely
started, and publishing it would be noise wearing a number's clothes.

**One warm-up run per scenario, discarded, then three measured trials, reported as the
median.** First-run effects are real — caches are cold, the server may still be settling —
and a mean over three trials carries a single outlier straight into the headline, where a
median does not. The warm-up is not recorded as a sample because it is not a measurement;
this docstring is the record that it happened.

**The prompt sizes are approximate and never reported as exact.** Prompts are built to a
character budget from a ratio measured once (see `_sizing.py`), so a scenario named
`prefill_512` is about 512 tokens, not exactly. What the sample records is the count the
server itself reported, which is the true one.

Figures come from the server's own `predicted_per_second` and `prompt_per_second`. Where
a backend does not report them — anything that is not llama.cpp — this module reports
**nothing** for that figure rather than falling back to a wall-clock number that would be
the very blend this design exists to remove. Wall-clock is still recorded on every sample
as the cross-check.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from llmbench.evaluators._sizing import chars_per_token
from llmbench.evaluators.base import EvalContext, Evaluator, Verdict
from llmbench.models import Metric, Sample
from llmbench.registry import register

#: Ordinary prose, so the tokenizer sees the kind of text a real prompt contains. Nonsense
#: or repeated single characters tokenise very differently and would measure the wrong
#: thing.
_FILLER = ("The harbour ledger for that season records grain, sailcloth and lamp oil "
           "arriving on the morning tide, with the quantities set down in a steady hand. ")

_QUESTION = "\n\nReply with the single word: acknowledged."


@dataclass(frozen=True)
class Scenario:
    """One prompt size and answer length, and whether writing speed is measurable at it."""

    name: str
    prompt_tokens: int
    max_tokens: int
    #: False where the answer is a single token: see the module docstring.
    measures_decode: bool


_SCENARIOS = [
    Scenario("decode", 64, 256, True),
    Scenario("prefill_512", 512, 1, False),
    Scenario("prefill_2k", 2048, 1, False),
    Scenario("prefill_8k", 8192, 1, False),
    Scenario("combined_4k", 4096, 128, True),
]

#: Which scenario each headline figure is taken from. A leaderboard row has space for one
#: reading speed and one writing speed, so these two say which measurement got the slot.
#: `decode` because it is the only scenario whose generation is uncontaminated by a long
#: prompt, and `prefill_2k` because it is the middle of the three ingestion sizes - large
#: enough to be dominated by the work rather than by the round-trip, small enough that
#: most machines can hold it.
_HEADLINE_DECODE = "decode"
_HEADLINE_PREFILL = "prefill_2k"

_DEFAULTS = {
    # Three is the smallest number of trials a median can be taken over. More costs
    # linear time for a slowly-improving estimate; this is a bench, not a laboratory.
    "trials": 3,
    "warmup": True,
    # None = every scenario the machine can hold.
    "scenarios": None,
}


@register
class SpeedEvaluator(Evaluator):
    name = "speed"
    version = "1"
    default_config = _DEFAULTS

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        wanted = cfg["scenarios"]
        scenarios = [s for s in _SCENARIOS if not wanted or s.name in wanted]
        n_ctx = ctx.fingerprint.n_ctx or 8192
        cpt = await chars_per_token(ctx, _FILLER * 10)

        out: list[Sample] = []
        for scenario in scenarios:
            needed = scenario.prompt_tokens + scenario.max_tokens
            if needed > n_ctx:
                # The same distinction the context ladder draws (design D3): a machine
                # that cannot hold an 8k prompt has an honest limit, not a failure, and
                # scoring it zero would report a measurement nobody took.
                out.append(Sample(
                    evaluator=self.name, case_id=f"{scenario.name}:skipped",
                    group=scenario.name, dims={"scenario": scenario.name},
                    skipped=(f"not attempted: {scenario.name} needs {needed} tokens of "
                             f"context and this model reports {n_ctx}")))
                continue

            prompt = self._prompt(scenario, cpt)
            if cfg["warmup"]:
                # Run and drop. A first trial measures a cold cache and a server that may
                # still be settling, which is a fact about the moment rather than the
                # configuration.
                await self._trial(ctx, scenario, prompt, trial=-1)
            for trial in range(cfg["trials"]):
                out.append(await self._trial(ctx, scenario, prompt, trial))
        return out

    def _prompt(self, scenario: Scenario, cpt: float) -> str:
        """Filler padded to roughly the scenario's prompt size, plus a short question."""
        budget = max(1, int(scenario.prompt_tokens * cpt) - len(_QUESTION))
        repeats = max(1, budget // len(_FILLER))
        return (_FILLER * repeats)[:budget] + _QUESTION

    async def _trial(self, ctx: EvalContext, scenario: Scenario, prompt: str,
                     trial: int) -> Sample:
        return await self.run_case(
            ctx, case_id=f"{scenario.name}:{trial}", group=scenario.name,
            dims={"scenario": scenario.name, "trial": trial},
            messages=[{"role": "user", "content": prompt}],
            # No verdict: this module measures, it does not grade. A score here would be
            # a quality figure, and quality figures pool across machines - which is the
            # one thing a speed measurement must never do.
            grade=lambda res: Verdict(meta={
                "scenario": scenario.name,
                "prompt_tokens_requested": scenario.prompt_tokens,
                "answer": res.text[:80],
            }),
            max_tokens=scenario.max_tokens, temperature=0.0)

    def aggregate(self, samples: list[Sample]) -> list[Metric]:
        """Median per scenario, and nothing at all where the server stayed silent.

        Calls `super()` for the failure counts and the answer rate, which every evaluator
        owes. It could not until the blended `tok_per_sec_mean` was removed from the
        default aggregator — emitting that here would have put the wrong speed back
        beside the right ones, which is the whole point of this module.
        """
        metrics: list[Metric] = super().aggregate(samples)
        measured = [s for s in samples if s.error is None and s.skipped is None]
        by_scenario = {s.name: s for s in _SCENARIOS}

        for name in sorted({s.dims.get("scenario") for s in measured} - {None}):
            trials = [s for s in measured if s.dims.get("scenario") == name]
            scenario = by_scenario.get(name)

            metrics.extend(self._median(
                "prefill_tps", trials, lambda s: s.server_prompt_tps, name))
            if scenario is None or scenario.measures_decode:
                metrics.extend(self._median(
                    "decode_tps", trials, lambda s: s.server_gen_tps, name))
                # The cross-check, kept separate and named for what it is. It includes
                # prompt processing, so it is always the pessimistic figure and must
                # never be read as the decode speed.
                metrics.extend(self._median(
                    "wallclock_tps", trials, lambda s: s.tok_per_sec, name))
            # What the server said the prompt really was, since the scenario name is a
            # target rather than a measurement.
            metrics.extend(self._median(
                "prompt_tokens", trials, lambda s: s.input_tokens, name, unit="tokens"))

        # The headline pair, without dimensions, because that is what a leaderboard row
        # can hold - the same two-level shape `structured` already uses, where a
        # dimensionless `pass_rate` sits above the per-task ones. Which scenario each came
        # from is fixed and documented rather than inferable from the figure, so the
        # constants above are the answer to "speed at what?".
        metrics.extend(self._headline(metrics, "decode_tps", _HEADLINE_DECODE))
        metrics.extend(self._headline(metrics, "prefill_tps", _HEADLINE_PREFILL))
        return metrics

    def _headline(self, metrics: list[Metric], name: str, scenario: str) -> list[Metric]:
        """Promote one scenario's figure to a dimensionless headline, or nothing.

        Nothing when that scenario was skipped or the backend published no timings - a
        leaderboard with a blank speed column is honest, and one showing a figure from
        whichever scenario happened to run is not.
        """
        source = next((m for m in metrics
                       if m.name == name and m.dims.get("scenario") == scenario), None)
        if source is None:
            return []
        return [Metric(evaluator=self.name, name=name, value=source.value,
                       unit=source.unit, n=source.n, successes=source.successes)]

    def _median(self, metric: str, trials: list[Sample], read, scenario: str,
                unit: str = "tok/s") -> list[Metric]:
        """One metric, or none at all when the backend reported nothing.

        Absent is the honest answer for a backend that does not publish its own timings;
        substituting a wall-clock figure would silently reintroduce the blend.
        """
        values = [v for v in (read(s) for s in trials) if v is not None]
        if not values:
            return []
        return [Metric(evaluator=self.name, name=metric,
                       value=round(statistics.median(values), 2), unit=unit,
                       n=len(values), dims={"scenario": scenario})]
