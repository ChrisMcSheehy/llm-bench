"""A response that could not answer is not an answer of zero.

Found on 2026-08-05, the first time this bench met a real reasoning model: it put its
thinking in `reasoning_content`, spent the whole 32-token answer budget on it, and
returned `content: ""` with `finish_reason: "length"`. Every rung scored 0.00, which
reads as a model that cannot retrieve rather than a token budget that was too small.

The payload below is the real one, captured from llama-server b10144-d73c1d6b2.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from llmbench.evaluators.base import EvalContext
from llmbench.models import ModelFingerprint
from llmbench.registry import available, get
from llmbench.targets.base import GenResult, Target

# What the server really sent back. Trimmed only in the length of the reasoning text.
_CUT_OFF = {
    "choices": [{
        "message": {"role": "assistant", "content": "",
                    "reasoning_content": "*   Input: \"Say the word COBALT-7468\"..."},
        "finish_reason": "length",
    }],
    "usage": {"completion_tokens": 32, "prompt_tokens": 31, "total_tokens": 63},
}

_ANSWERED = {
    "choices": [{"message": {"role": "assistant", "content": "COBALT-7468"},
                 "finish_reason": "stop"}],
    "usage": {"completion_tokens": 5, "prompt_tokens": 31, "total_tokens": 36},
}

_SAID_NOTHING = {
    "choices": [{"message": {"role": "assistant", "content": ""},
                 "finish_reason": "stop"}],
    "usage": {"completion_tokens": 0, "prompt_tokens": 31, "total_tokens": 31},
}


class _Fixed(Target):
    """A target that replies with one canned payload, through the real HTTP path."""

    engine = "fixed"

    def __init__(self, payload: dict):
        super().__init__("http://fixed")
        self._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))

    async def detect(self):
        return ModelFingerprint(engine=self.engine, base_url=self.base_url, model_id="m")

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _gen(payload: dict) -> GenResult:
    async def go():
        t = _Fixed(payload)
        try:
            return await t.generate([{"role": "user", "content": "hi"}], max_tokens=32)
        finally:
            await t.aclose()
    return asyncio.run(go())


# ---- what the response interpretation makes of each payload -------------------

def test_a_cut_off_response_is_marked_truncated():
    """`truncated` read a top-level key no OpenAI-shaped response has, so it was always
    None - the one field that could have explained the zero was never set."""
    assert _gen(_CUT_OFF).truncated is True
    assert _gen(_ANSWERED).truncated is False


def test_a_cut_off_response_with_no_content_cannot_be_graded():
    reason = _gen(_CUT_OFF).unusable_reason
    assert reason, "a response that never reached an answer was offered up for grading"
    assert "budget" in reason, reason


def test_a_real_answer_is_gradable():
    """The success condition. A change that called everything unusable would also stop
    the zeroes, and would destroy the bench."""
    res = _gen(_ANSWERED)
    assert res.unusable_reason is None
    assert res.text == "COBALT-7468"


def test_a_model_that_finished_and_said_nothing_is_still_graded():
    """Deliberately NOT a skip: it stopped of its own accord having emitted no tokens,
    which is a genuine failure to answer and must keep counting as one. Treating this as
    unusable would hide real incapacity, which is the opposite error."""
    assert _gen(_SAID_NOTHING).unusable_reason is None


# ---- and what every evaluator does with it -----------------------------------

class _CutOff(Target):
    """Every generation comes back cut off before an answer, as a reasoning model's
    does when the answer budget is small."""

    engine = "cutoff"

    async def detect(self):
        return ModelFingerprint(engine=self.engine, base_url=self.base_url, model_id="m")

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def generate(self, messages, *, max_tokens=512, temperature=0.0, extra=None):
        return GenResult(text="", input_tokens=100, output_tokens=max_tokens,
                         latency_ms=50.0, tok_per_sec=10.0, finish_reason="length")


#: Every evaluator that grades text a model generated, with a configuration small
#: enough to run in a test. `perplexity` is absent because it shells out to a binary
#: and never calls generate at all.
_CONFIGS = {
    "needle": {"context_lengths": [2048], "depths": [50], "repeats": 1},
    "long_context": {"context_lengths": [2048], "queries_per_rung": 1},
    "mcqa": {"limit": 2},
    "math_qa": {"limit": 2},
    "ifeval": {"limit": 2},
    "structured": {},
    "text2sql": {"limit": 2},
    "coding": {"execute": False, "limit": 1},
    "human": {"limit": 1},
    "oneshot": {"limit": 1},
}


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_no_evaluator_scores_a_response_that_never_answered(name):
    fp = ModelFingerprint(engine="cutoff", base_url="http://cutoff", model_id="m",
                          n_ctx=8192)
    ctx = EvalContext(target=_CutOff("http://cutoff"), fingerprint=fp,
                      config=_CONFIGS[name])
    samples = asyncio.run(get(name)().evaluate(ctx))

    assert samples, f"{name} produced no samples at all"
    scored = [s for s in samples if s.score is not None or s.passed is not None]
    assert not scored, (
        f"{name} graded {len(scored)} response(s) that never reached an answer: "
        f"{[(s.case_id, s.score, s.passed) for s in scored[:3]]}")
    assert any(s.skipped for s in samples), f"{name} recorded no reason for the gap"


def test_the_evaluator_list_covers_everything_that_generates():
    """An evaluator added later must be added here, or this fails rather than quietly
    leaving the new one untested - the blindness an edit list always has."""
    generating = {n for n in available() if n != "perplexity"}
    missing = sorted(generating - set(_CONFIGS))
    assert not missing, f"evaluators missing from the cut-off check: {missing}"


# ---- and what the aggregators do with the skips the guards now produce --------
#
# Found by running against a real model: ifeval crashed on `sum(s.score ...)` because
# its own `graded` filter excludes errors but not skips, and a skipped sample carries
# score None. The three aggregators that override the default and filter on `error is
# None` alone are the ones at risk - the rest filter on `score`/`passed is not None`,
# which excludes a skip already.

def _skipped_sample(evaluator: str, **fields):
    from llmbench.models import Sample
    return Sample(evaluator=evaluator, case_id="s", skipped="no answer: cut off",
                  **fields)


def test_ifeval_survives_a_skipped_sample():
    """The crash. A skipped sample's score is None and went straight into sum()."""
    from llmbench.evaluators.ifeval import IFEvalEvaluator
    from llmbench.models import Sample

    metrics = {m.name: m for m in IFEvalEvaluator().aggregate([
        Sample(evaluator="ifeval", case_id="a", score=1.0, passed=True),
        _skipped_sample("ifeval"),
    ])}

    assert metrics["instruction_acc"].value == 1.0, "the skip was averaged in"
    assert metrics["instruction_acc"].n == 1, "the skip was counted as evidence"


def test_human_does_not_count_a_skip_as_a_response():
    from llmbench.evaluators.human import HumanEvalEvaluator
    from llmbench.models import Sample

    metrics = {m.name: m for m in HumanEvalEvaluator().aggregate([
        Sample(evaluator="human", case_id="a"),
        _skipped_sample("human"),
    ])}

    assert metrics["responses"].value == 1.0, "a skip was reported as a response"


def test_oneshot_does_not_score_a_skip_as_zero():
    """`s.score or 0.0` turned a skip into a zero - the exact bug being removed."""
    from llmbench.evaluators.oneshot import OneShotEvaluator
    from llmbench.models import Sample

    metrics = [m for m in OneShotEvaluator().aggregate([
        Sample(evaluator="oneshot", case_id="a", score=1.0, dims={"kind": "game"}),
        _skipped_sample("oneshot", dims={"kind": "game"}),
    ]) if m.name == "build_score"]

    assert metrics and metrics[0].value == 1.0, \
        f"the skip was averaged in as a zero: {[m.value for m in metrics]}"
