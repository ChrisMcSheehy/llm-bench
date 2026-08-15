"""Two speed figures, and the ways a single blended one used to be wrong.

Design B3. `targets/base.py` computes `tok_per_sec` as output tokens over total wall
time - which includes reading the prompt - and the old headline then averaged that across
every rung of a context ladder. So the published speed depended most heavily on the one
variable it hid. The correct pair was already arriving in llama.cpp's `timings` block and
nothing read it.
"""
from __future__ import annotations

import asyncio

from llmbench.evaluators.base import EvalContext
from llmbench.evaluators.speed import SpeedEvaluator
from llmbench.models import ModelFingerprint, Sample
from llmbench.store import QUALITY_METRICS
from llmbench.targets.base import GenResult, Target


class _Server(Target):
    """A server that reports its own timings, and counts what it was asked to do."""

    engine = "speedy"

    def __init__(self, prompt_tps=None, gen_tps=None, url="http://speedy"):
        super().__init__(url)
        self._prompt_tps = prompt_tps if prompt_tps is not None else [900.0]
        self._gen_tps = gen_tps if gen_tps is not None else [50.0]
        self.calls: list[int] = []          # prompt length of each call, in characters

    async def detect(self):
        return ModelFingerprint(engine=self.engine, base_url=self.base_url, model_id="m")

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def generate(self, messages, *, max_tokens=512, temperature=0.0, extra=None):
        self.calls.append(len(messages[-1]["content"]))
        i = len(self.calls) - 1
        pick = lambda seq: seq[i % len(seq)]
        return GenResult(
            text="acknowledged", input_tokens=len(messages[-1]["content"]) // 4,
            output_tokens=max_tokens, latency_ms=100.0, tok_per_sec=11.0,
            server_prompt_tps=pick(self._prompt_tps), server_gen_tps=pick(self._gen_tps),
            finish_reason="stop")


def _run(target: _Server, n_ctx: int = 32768, **config) -> list[Sample]:
    fp = ModelFingerprint(engine="speedy", base_url=target.base_url, model_id="m",
                          n_ctx=n_ctx)
    ctx = EvalContext(target=target, fingerprint=fp, config=config)
    return asyncio.run(SpeedEvaluator().evaluate(ctx))


def _metrics(samples, evaluator=None) -> dict:
    return {(m.name, m.dims.get("scenario")): m
            for m in (evaluator or SpeedEvaluator()).aggregate(samples)}


# ---- the two figures are two figures ----------------------------------------

def test_a_prefill_scenario_reports_no_writing_speed():
    """It generates one token. Timing a process that has barely started is noise, and
    publishing it beside a real figure invites it to be read as one."""
    samples = _run(_Server(), scenarios=["prefill_2k"], trials=3, warmup=False)
    m = _metrics(samples)

    assert ("prefill_tps", "prefill_2k") in m
    assert ("decode_tps", "prefill_2k") not in m, "a one-token decode figure was published"


def test_a_decode_scenario_reports_both_and_says_which_is_which():
    samples = _run(_Server(), scenarios=["decode"], trials=3, warmup=False)
    m = _metrics(samples)

    assert m[("decode_tps", "decode")].value == 50.0
    assert m[("prefill_tps", "decode")].value == 900.0
    # The blended wall-clock number still exists, under a name that says what it is.
    assert m[("wallclock_tps", "decode")].value == 11.0


def test_the_wall_clock_figure_is_never_called_the_decode_speed():
    """The defect in one line: 11 tok/s of wall clock and 50 tok/s of actual generation
    are both true, and only one of them answers "how fast does this model write"."""
    samples = _run(_Server(), scenarios=["decode"], trials=1, warmup=False)
    m = _metrics(samples)

    assert m[("decode_tps", "decode")].value != m[("wallclock_tps", "decode")].value


# ---- how the trials are combined --------------------------------------------

def test_an_outlier_trial_does_not_move_the_figure():
    """The reason for a median. One slow trial - a background process, a scheduler
    hiccup - drags a mean of three a long way and a median not at all."""
    samples = _run(_Server(gen_tps=[50.0, 52.0, 2.0]),
                   scenarios=["decode"], trials=3, warmup=False)
    m = _metrics(samples)

    assert m[("decode_tps", "decode")].value == 50.0, "the mean would be 34.67"
    assert m[("decode_tps", "decode")].n == 3, "the figure must say how many trials"


def test_the_warm_up_run_happens_and_is_not_measured():
    """A first trial measures a cold cache, which is a fact about the moment rather than
    about the configuration - but skipping it entirely would leave that in the figures."""
    warm = _Server()
    warmed = _run(warm, scenarios=["decode"], trials=3, warmup=True)

    cold = _Server()
    _run(cold, scenarios=["decode"], trials=3, warmup=False)

    assert len(warm.calls) == 4, "the warm-up run was never made"
    assert len([s for s in warmed if s.skipped is None]) == 3, "it was counted as a trial"
    assert len(cold.calls) == 3, "a warm-up ran when it was switched off"


# ---- what a silent backend gets ---------------------------------------------

def test_a_backend_that_reports_no_timings_gets_no_figure_rather_than_a_wrong_one():
    """Only llama.cpp publishes its own timings. Falling back to wall clock here would
    reintroduce the exact blend this evaluator exists to remove, under the right name."""
    samples = _run(_Server(prompt_tps=[None], gen_tps=[None]),
                   scenarios=["decode"], trials=3, warmup=False)
    m = _metrics(samples)

    assert ("decode_tps", "decode") not in m
    assert ("prefill_tps", "decode") not in m
    assert m[("wallclock_tps", "decode")].value == 11.0, "the cross-check should remain"


# ---- a machine that cannot hold the scenario --------------------------------

def test_a_scenario_too_big_for_the_context_is_skipped_with_a_reason():
    """The distinction design D3 draws on the context ladder, one module over: a limit
    is not a failure, and it is certainly not a speed of zero."""
    target = _Server()
    samples = _run(target, n_ctx=2048, scenarios=["prefill_8k"], trials=3, warmup=False)

    assert target.calls == [], "an 8k prompt was sent to a 2k-context model"
    assert len(samples) == 1, "one row for the scenario, not one per trial"
    assert samples[0].skipped and "2048" in samples[0].skipped, samples[0].skipped
    assert samples[0].score is None, "a skip must never carry a figure"


def test_the_skip_is_counted_and_not_treated_as_an_error():
    samples = _run(_Server(), n_ctx=2048, scenarios=["prefill_8k"], trials=3, warmup=False)
    m = _metrics(samples)

    assert m[("skipped_count", None)].value == 1.0
    assert m[("error_count", None)].value == 0.0


# ---- the figures must never pool across machines ----------------------------

def test_speed_figures_are_not_in_the_pool_across_machines_allowlist():
    """The whole point of separating host from model (design D1). Two machines running
    one configuration produce one quality figure and two speeds, and a speed that pooled
    would average a laptop with a desktop and present it as a property of the model.

    `store.QUALITY_METRICS` is an allowlist: anything absent groups by host, which is the
    safe direction. This test exists so that a later edit adding these "for completeness"
    has to argue with it first.
    """
    for name in ("prefill_tps", "decode_tps", "wallclock_tps", "prompt_tokens"):
        assert name not in QUALITY_METRICS, (
            f"{name} would be pooled across machines, hiding two computers in one figure")


# ---- the prompt size is a target, and the report says what it really was -----

def test_the_recorded_prompt_size_is_the_servers_count_not_the_target():
    """Prompts are padded using a ratio measured once, so the size is approximate. The
    scenario name is what was asked for; the metric is what actually arrived."""
    samples = _run(_Server(), scenarios=["prefill_2k"], trials=1, warmup=False)
    m = _metrics(samples)

    reported = m[("prompt_tokens", "prefill_2k")].value
    assert reported > 0
    assert samples[0].meta["prompt_tokens_requested"] == 2048
    assert abs(reported - 2048) < 2048 * 0.5, (
        f"padding produced {reported} tokens for a 2048-token target, which is not "
        f"'approximately' by any reading")
