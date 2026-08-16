"""How often the model answered at all, beside how often it was right.

Design B2. On 2026-08-05 this bench met a reasoning model that spent its whole answer
budget thinking and returned nothing. That was fixed - `test_unusable_response.py` proves
every evaluator now refuses to *score* a non-answer - but the distinction stopped there.
A configuration answering 99% of questions at 85% accuracy and one answering 80% at 85%
appeared as identical rows, and an accuracy computed over an unstated subset is exactly
the naked figure design D7 forbids.

The subtlety is the denominator. `Sample.skipped` carries two different situations: a
rung the machine could never attempt, and a question that was asked and produced nothing.
Only the second is a fact about the model, and counting the first would report a laptop's
memory limit as a model declining to answer.
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone

import pytest

from llmbench.evaluators.base import EvalContext, Evaluator, Verdict
from llmbench.models import ModelFingerprint, RunResult, Sample
from llmbench.registry import available, get
from llmbench.store import QUALITY_METRICS, Store
from llmbench.targets.base import GenResult, Target


class _Eval(Evaluator):
    name = "t"

    async def evaluate(self, ctx):        # never called; aggregate() is the subject
        return []


def _named(samples) -> dict:
    return {m.name: m for m in _Eval().aggregate(samples)}


def _answered(answered: bool) -> Sample:
    return Sample(evaluator="t", case_id="c", score=1.0 if answered else None,
                  answered=answered, skipped=None if answered else "no answer: cut off")


# ---- the figure itself -------------------------------------------------------

def test_the_answer_rate_is_the_share_that_produced_something_gradable():
    m = _named([_answered(True), _answered(True), _answered(True), _answered(False)])
    assert m["answer_rate"].value == 0.75
    assert m["answer_rate"].n == 4, "the figure must say how many questions were asked"


def test_a_configuration_that_answers_everything_is_a_different_row():
    """The whole point, stated as the two rows the leaderboard used to show identically."""
    diligent = _named([_answered(True), _answered(True), _answered(True), _answered(True)])
    evasive = _named([_answered(True), _answered(True), _answered(True), _answered(False)])

    assert diligent["score_mean"].value == evasive["score_mean"].value, (
        "the setup for this test is wrong: both must score the same on what they answered")
    assert diligent["answer_rate"].value != evasive["answer_rate"].value


def test_a_rung_the_machine_never_attempted_is_not_a_question_ducked():
    """`skipped` covers both "could not attempt" and "answered nothing". Only the second
    belongs in this denominator - otherwise a modest machine's honest limit is reported
    as the model declining to answer."""
    m = _named([
        _answered(True),
        Sample(evaluator="t", case_id="rung", skipped="not attempted: 512k needs more "
                                                      "memory than this machine has"),
    ])
    assert m["answer_rate"].value == 1.0
    assert m["answer_rate"].n == 1, "an unattempted rung was counted as a question"


def test_a_failed_call_is_not_counted_either():
    """A network fault is a fact about the wire, not about the model's willingness."""
    m = _named([_answered(True), Sample(evaluator="t", case_id="e", error="boom")])
    assert m["answer_rate"].n == 1


def test_nothing_is_reported_when_nothing_was_asked():
    """A rate over zero questions is not zero; it is absent."""
    m = _named([Sample(evaluator="t", case_id="s", skipped="not attempted: no budget")])
    assert "answer_rate" not in m


# ---- every evaluator that asks a model reports it ---------------------------

class _Silent(Target):
    """Every generation is cut off before an answer arrives, as a reasoning model's is
    when the answer budget is small."""

    engine = "silent"

    async def detect(self):
        return ModelFingerprint(engine=self.engine, base_url=self.base_url, model_id="m")

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    async def generate(self, messages, *, max_tokens=512, temperature=0.0, extra=None):
        return GenResult(text="", input_tokens=100, output_tokens=max_tokens,
                         latency_ms=50.0, tok_per_sec=10.0, finish_reason="length")


#: Every evaluator that asks a model something. `perplexity` shells out to a binary and
#: never generates, so it has no answer rate to report and must not invent one.
_CONFIGS = {
    "needle": {"context_lengths": [2048], "depths": [50], "repeats": 1},
    "long_context": {"context_lengths": [2048], "queries_per_rung": 1},
    "mcqa": {"limit": 2},
    "math_qa": {"limit": 2},
    "ifeval": {},
    "structured": {},
    "text2sql": {},
    "coding": {"execute": False},
    "human": {"limit": 1},
    "oneshot": {"limit": 1},
    "speed": {"scenarios": ["decode"], "trials": 1, "warmup": False},
    "reassembly": {"context_lengths": [2048], "repeats": 1},
    "agency": {"scenarios": ["lookup_email"], "max_rounds": 3},
}


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_a_model_that_answers_nothing_reports_an_answer_rate_of_zero(name):
    """Not a missing figure, and certainly not an accuracy of zero: the model was asked
    and said nothing, which is a measurement in its own right."""
    fp = ModelFingerprint(engine="silent", base_url="http://silent", model_id="m",
                          n_ctx=8192)
    evaluator = get(name)()
    samples = asyncio.run(evaluator.evaluate(
        EvalContext(target=_Silent("http://silent"), fingerprint=fp,
                    config=_CONFIGS[name])))
    metrics = {m.name: m for m in evaluator.aggregate(samples)}

    assert "answer_rate" in metrics, f"{name} reports no answer rate"
    assert metrics["answer_rate"].value == 0.0, (
        f"{name} reported {metrics['answer_rate'].value} for a model that never answered")


def test_the_evaluator_list_covers_everything_that_generates():
    missing = sorted({n for n in available() if n != "perplexity"} - set(_CONFIGS))
    assert not missing, f"evaluators missing from the answer-rate check: {missing}"


def test_perplexity_reports_no_answer_rate():
    """It never asks a model anything, so it has no denominator and must not print one."""
    metrics = {m.name for m in get("perplexity")().aggregate(
        [Sample(evaluator="perplexity", case_id="ppl", skipped="not attempted")])}
    assert "answer_rate" not in metrics


# ---- how it travels ---------------------------------------------------------

def test_the_answer_rate_pools_across_machines():
    """It describes the model, not the computer: a configuration that spends its budget
    thinking and returns nothing does so wherever it runs. The speed metrics are absent
    from this set for exactly the opposite reason."""
    assert "answer_rate" in QUALITY_METRICS


def test_the_flag_survives_the_database(tmp_path):
    store = Store(str(tmp_path / "s.db"))
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://x", model_id="m")
    store.start_run(RunResult(run_id="r1", fingerprint=fp, suite="t",
                              started_at=datetime.now(timezone.utc)))
    store.add_samples("r1", [_answered(True), _answered(False),
                             Sample(evaluator="t", case_id="rung", skipped="not attempted")])
    rows = sorted(r["answered"] for r in store.conn.execute(
        "SELECT answered FROM sample WHERE run_id='r1'") if r["answered"] is not None)
    unknown = store.conn.execute(
        "SELECT COUNT(*) FROM sample WHERE run_id='r1' AND answered IS NULL").fetchone()[0]
    store.close()

    assert rows == [0, 1]
    assert unknown == 1, "a rung nobody attempted must record no opinion, not a zero"


def test_an_older_database_gains_the_column(tmp_path):
    """CREATE TABLE IF NOT EXISTS does nothing to a database that already exists, so a
    column added after the first release reaches one only through the migration list."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE sample (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,"
        " evaluator TEXT, case_id TEXT, grp TEXT, dims_json TEXT, score REAL);"
        "INSERT INTO sample (run_id, case_id) VALUES ('old-run', 'c1');")
    conn.commit()
    conn.close()

    store = Store(str(db))
    columns = {r[1] for r in store.conn.execute("PRAGMA table_info(sample)")}
    kept = store.conn.execute("SELECT run_id, answered FROM sample").fetchall()
    store.close()

    assert "answered" in columns, "migration did not add the answered column"
    assert [tuple(r) for r in kept] == [("old-run", None)], (
        "an existing row was dropped, or backfilled with a guess about whether a model "
        "nobody asked had answered")
