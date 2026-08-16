"""Finding three fragments and putting them back together, graded in four tiers.

Design B5. Every other long-context instrument in this project grades 1.0 or 0.0, so at
the rung where a configuration starts to degrade they say a line was crossed and nothing
about how far. The central question here is *how much* quality a compression setting
costs, and a step function cannot answer it.

The tiers exist so a low score can be explained rather than merely observed: "found two of
three" and "found all three and mistyped one character" are the same `exact_match` of zero
and completely different findings about the configuration.
"""
from __future__ import annotations

import asyncio
import re

import pytest

from llmbench.evaluators.base import EvalContext
from llmbench.evaluators.reassembly import (
    ReassemblyEvaluator, bit_accuracy, split_key,
)
from llmbench.models import ModelFingerprint, Sample
from llmbench.targets.base import GenResult, Target

_FRAGMENT = re.compile(r"FRAGMENT (ALPHA|BETA|GAMMA): ([0-9a-f]+)")


class _Responder(Target):
    """A model that reads the planted fragments back out of its own prompt.

    `reply(parts)` decides what it does with them — return them correctly, in the wrong
    order, drop one, mistype one. That drives the grading through the real path rather
    than hand-building samples, so the prompt, the placement and the grader are all
    exercised together.
    """

    engine = "responder"

    def __init__(self, reply, url="http://responder"):
        super().__init__(url)
        self._reply = reply
        self.prompts: list[str] = []

    async def detect(self):
        return ModelFingerprint(engine=self.engine, base_url=self.base_url, model_id="m")

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def generate(self, messages, *, max_tokens=512, temperature=0.0, extra=None):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        found = dict((label, value) for label, value in _FRAGMENT.findall(prompt))
        parts = [found.get(k, "") for k in ("ALPHA", "BETA", "GAMMA")]
        return GenResult(text=self._reply(parts), input_tokens=100, output_tokens=20,
                         latency_ms=10.0, tok_per_sec=100.0, server_prompt_tps=900.0,
                         server_gen_tps=50.0, finish_reason="stop")


def _run(reply, **config) -> tuple[list[Sample], _Responder]:
    target = _Responder(reply)
    fp = ModelFingerprint(engine="responder", base_url=target.base_url, model_id="m",
                          n_ctx=32768)
    cfg = {"context_lengths": [4096], "repeats": 1, **config}
    samples = asyncio.run(ReassemblyEvaluator().evaluate(
        EvalContext(target=target, fingerprint=fp, config=cfg)))
    return samples, target


def _one(reply, **config) -> Sample:
    samples, _ = _run(reply, **config)
    graded = [s for s in samples if s.skipped is None and s.error is None]
    assert len(graded) == 1, [s.skipped or s.error for s in samples]
    return graded[0]


# ---- bit accuracy, the figure this evaluator is for --------------------------

def test_a_perfect_answer_scores_every_bit():
    s = _one(lambda parts: "".join(parts))
    assert s.meta["bit_accuracy"] == 1.0
    assert s.meta["exact_match"] is True
    assert s.passed is True


def test_one_wrong_character_is_neither_right_nor_zero():
    """The whole reason for a bit-level figure. Every other long-context evaluator here
    would call this a flat failure, indistinguishable from finding nothing at all."""
    def mistype(parts):
        key = "".join(parts)
        wrong = "0" if key[-1] != "0" else "1"
        return key[:-1] + wrong

    s = _one(mistype)
    assert s.meta["exact_match"] is False, "the setup is wrong: this must not be exact"
    assert 0.9 < s.meta["bit_accuracy"] < 1.0, s.meta["bit_accuracy"]


def test_the_score_is_the_bit_accuracy_so_the_per_rung_figure_is_a_gradient():
    """`score` feeds the per-rung breakdown. If it were the exact match, the breakdown
    would be the step function this evaluator was built to replace."""
    s = _one(lambda parts: "".join(parts))
    assert s.score == s.meta["bit_accuracy"]


# ---- a wrong-length answer -------------------------------------------------

def test_a_wrong_length_answer_reports_bit_accuracy_as_unknown():
    """Comparing bits between different lengths measures alignment, not recall, and would
    produce a plausible-looking ~50% out of a structural failure."""
    s = _one(lambda parts: "".join(parts)[:-5])

    assert s.meta["bit_accuracy"] is None, "a length mismatch produced a number"
    assert s.score is None, "an unknown must not reach the average as a figure"


def test_a_wrong_length_answer_still_says_how_much_was_retrieved():
    """The tier below it keeps working, so the failure is diagnosable rather than blank.

    The answer here carries all three parts verbatim and two characters too many — the
    retrieval was perfect and only the assembly was not, which is precisely the case bit
    accuracy cannot describe and `parts_found` can.
    """
    s = _one(lambda parts: "".join(parts) + "ff")

    assert s.meta["bit_accuracy"] is None
    assert s.meta["parts_found"] == 3
    assert s.passed is False, "it is still a failure, and pass_rate must count it"


def test_a_truncated_answer_reports_the_part_it_mangled_as_missing():
    """Cutting the end off the key destroys the last fragment, so two of three is the
    honest count — `parts_found` measures what came back verbatim, not what was asked
    for."""
    s = _one(lambda parts: "".join(parts)[:-5])
    assert s.meta["parts_found"] == 2


def test_a_wrong_length_answer_is_counted_by_the_exact_match_figure():
    """It must not vanish from every figure just because one of them cannot describe it."""
    samples, _ = _run(lambda parts: "".join(parts)[:-5])
    metrics = {m.name: m for m in ReassemblyEvaluator().aggregate(samples)}

    assert metrics["exact_match"].value == 0.0
    assert metrics["exact_match"].n == 1
    assert "score_mean" not in metrics, "an unknown bit accuracy was averaged anyway"


# ---- the tiers isolate different failures -----------------------------------

def test_finding_two_of_three_is_reported_as_two_of_three():
    s = _one(lambda parts: parts[0] + parts[1])
    assert s.meta["parts_found"] == 2


def test_the_right_parts_in_the_wrong_order_is_an_assembly_failure():
    """Retrieval was perfect and the answer is wrong, which no other evaluator here can
    express. `parts_found` says the reading worked; `order_correct` says the writing did
    not."""
    s = _one(lambda parts: parts[2] + parts[1] + parts[0])

    assert s.meta["parts_found"] == 3, "retrieval was perfect"
    assert s.meta["order_correct"] is False
    assert s.meta["exact_match"] is False


def test_a_correct_answer_is_in_order():
    """The success condition for the check above: a rule that called everything
    out-of-order would also flag the reversed case and be useless."""
    assert _one(lambda parts: "".join(parts)).meta["order_correct"] is True


# ---- the document and the key ----------------------------------------------

def test_all_three_fragments_are_planted():
    _, target = _run(lambda parts: "".join(parts))
    assert len(_FRAGMENT.findall(target.prompts[0])) == 3


def test_the_key_is_hexadecimal():
    """Hex maps exactly four bits per character, which is what makes the bit comparison
    meaningful. Base64 would measure transcription luck through `l`/`I`/`1` and `O`/`0`."""
    s = _one(lambda parts: "".join(parts))
    assert re.fullmatch(r"[0-9a-f]+", s.meta["expected"]), s.meta["expected"]


def test_the_same_seed_produces_the_same_key():
    """Every comparison in this project rests on runs being reproducible."""
    first = _one(lambda parts: "".join(parts), seed=99)
    second = _one(lambda parts: "".join(parts), seed=99)
    assert first.meta["expected"] == second.meta["expected"]


def test_a_different_seed_produces_a_different_key():
    a = _one(lambda parts: "".join(parts), seed=1)
    b = _one(lambda parts: "".join(parts), seed=2)
    assert a.meta["expected"] != b.meta["expected"]


def test_each_cell_gets_its_own_key():
    """One key reused across rungs could be carried forward by a server reusing its cache
    between calls, and the run would measure that instead of retrieval."""
    samples, _ = _run(lambda parts: "".join(parts),
                      context_lengths=[2048, 4096], repeats=1)
    keys = [s.meta["expected"] for s in samples if s.meta.get("expected")]
    assert len(keys) == 2 and keys[0] != keys[1], keys


# ---- the helpers, at the edges ----------------------------------------------

def test_bit_accuracy_of_identical_keys_is_one():
    assert bit_accuracy("abcd", "abcd") == 1.0


def test_bit_accuracy_of_one_flipped_bit():
    """`0` and `1` differ in exactly one of the sixteen bits."""
    assert bit_accuracy("0000", "0001") == pytest.approx(15 / 16)


def test_bit_accuracy_refuses_a_length_mismatch():
    assert bit_accuracy("abcd", "abc") is None


def test_bit_accuracy_refuses_something_that_is_not_hexadecimal():
    """A model that answered in prose has not produced a key to compare bits with."""
    assert bit_accuracy("abcd", "zzzz") is None


def test_the_key_splits_into_equal_parts():
    """Equal so that no fragment is a smaller target than the others."""
    assert split_key("0123456789ab") == ["0123", "4567", "89ab"]
