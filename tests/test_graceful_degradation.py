"""A machine that cannot hold the top of the ladder, exercised through the real code.

Testing `climb()` with a fake rung function and testing the evaluators against a
healthy server would leave the seam between them untested, which is how this project
last shipped a feature that could not fire (LESSONS.md,
unit-tests-either-side-of-a-seam-do-not-test-the-seam). So these drive the real
evaluator against a target that really refuses, and assert that the rungs below the
limit were really graded - not merely that nothing raised.
"""
from __future__ import annotations

import asyncio

import selftest
from llmbench.evaluators.base import EvalContext
from llmbench.evaluators.long_context import LongContextEvaluator
from llmbench.evaluators.needle import NeedleEvaluator
from llmbench.models import ModelFingerprint


class _Ceiling(selftest.MockTarget):
    """A mock server that refuses a prompt beyond a fixed context, as a real one does."""

    max_context = 4096


def _ctx(target, **config) -> EvalContext:
    fp = ModelFingerprint(engine="mock", base_url="http://mock", model_id="m",
                          n_ctx=32768)
    return EvalContext(target=target, fingerprint=fp, config=config)


def _at(samples, length):
    return [s for s in samples if s.dims.get("context_len") == length]


def _graded(samples, length):
    return [s for s in _at(samples, length) if s.score is not None]


def test_needle_grades_the_rungs_below_the_limit():
    """The success condition. A climber that skipped everything would also not raise."""
    samples = asyncio.run(NeedleEvaluator().evaluate(_ctx(
        _Ceiling("http://mock", model="A"),
        context_lengths=[2048, 8192, 16384], depths=[50], repeats=1)))

    low = _graded(samples, 2048)
    assert len(low) == 1, f"the 2048 rung was not graded: {_at(samples, 2048)}"
    assert low[0].score == 1.0, "the mock answers correctly, so this rung must pass"


def test_needle_stops_climbing_at_the_first_refusal():
    """Never attempted, not attempted-and-failed.

    Asserting only that 16384 has no score would pass on a climber that carried on and
    failed there, which is the behaviour this phase removes - so the assertion is that
    the rung produced no *error* either. Nothing was sent at all.
    """
    samples = asyncio.run(NeedleEvaluator().evaluate(_ctx(
        _Ceiling("http://mock", model="A"),
        context_lengths=[2048, 8192, 16384], depths=[50], repeats=1)))

    assert [s.error for s in _at(samples, 8192) if s.error], "8192 should have failed"
    assert not [s for s in _at(samples, 16384) if s.error], \
        "16384 was attempted after a refusal instead of being skipped"
    assert _graded(samples, 16384) == []


def test_needle_records_the_unreached_rung_as_skipped_with_a_reason():
    samples = asyncio.run(NeedleEvaluator().evaluate(_ctx(
        _Ceiling("http://mock", model="A"),
        context_lengths=[2048, 8192, 16384], depths=[50], repeats=1)))

    skipped = [s for s in samples if s.skipped]
    assert [s.dims["context_len"] for s in skipped] == [16384]
    assert "8192" in skipped[0].skipped, skipped[0].skipped
    assert skipped[0].error is None, "a skip must never be recorded as an error"


def test_one_skipped_row_per_rung_and_not_one_per_depth():
    """Five depths, one unreached rung: the meaningful count is rungs, not cells."""
    samples = asyncio.run(NeedleEvaluator().evaluate(_ctx(
        _Ceiling("http://mock", model="A"),
        context_lengths=[2048, 8192, 16384], depths=[0, 25, 50, 75, 100], repeats=1)))

    assert len([s for s in samples if s.skipped]) == 1


def test_long_context_grades_the_rungs_below_the_limit():
    samples = asyncio.run(LongContextEvaluator().evaluate(_ctx(
        _Ceiling("http://mock", model="A"),
        context_lengths=[2048, 8192, 16384], queries_per_rung=1)))

    graded = _graded(samples, 2048)
    assert len(graded) == 2, f"multikey and vartrack should both be graded: {graded}"
    assert all(s.score == 1.0 for s in graded), "the mock answers both tasks correctly"


def test_long_context_skips_the_rungs_above_a_refusal():
    samples = asyncio.run(LongContextEvaluator().evaluate(_ctx(
        _Ceiling("http://mock", model="A"),
        context_lengths=[2048, 8192, 16384], queries_per_rung=1)))

    skipped = [s for s in samples if s.skipped]
    assert [s.dims["context_len"] for s in skipped] == [16384]
    assert not [s for s in _at(samples, 16384) if s.error], \
        "16384 was attempted after a refusal instead of being skipped"


def test_perplexity_without_a_binary_is_skipped_and_not_failed():
    """An opt-in module that was not opted into is not a fault of the model.

    This inflated the error count on every healthy run of the default suite.
    """
    from llmbench.evaluators.perplexity import PerplexityEvaluator

    samples = asyncio.run(PerplexityEvaluator().evaluate(
        _ctx(selftest.MockTarget("http://mock", model="A"))))

    assert len(samples) == 1
    assert samples[0].error is None, "a module nobody configured has not failed"
    assert samples[0].skipped, "and it must say that it was skipped"
    assert "binary" in samples[0].skipped, samples[0].skipped


def _suite(tmp_path, body: str):
    path = tmp_path / "s.yaml"
    path.write_text(body, encoding="utf-8")
    return str(path)


class _Boom:
    """An evaluator that fails the way a broken test module really fails."""

    name = "boom"

    async def evaluate(self, ctx):
        raise RuntimeError("this module is broken")

    def aggregate(self, samples):
        return []


def test_a_broken_module_does_not_cost_the_modules_after_it(tmp_path, monkeypatch):
    from llmbench import orchestrator, registry
    from llmbench.store import Store
    from llmbench.targets import _ENGINES

    _ENGINES["mock"] = selftest.MockTarget
    monkeypatch.setattr(orchestrator, "get_evaluator",
                        lambda name: _Boom if name == "boom" else registry.get(name))

    store = Store(str(tmp_path / "b.db"))
    orch = orchestrator.Orchestrator(store, log=lambda *a: None)
    results = asyncio.run(orch.run_suite(_suite(tmp_path,
        "name: t\ntargets:\n  - engine: mock\n    base_url: http://mock-a\n"
        "    model: A\nevaluators:\n  boom:\n  mcqa:\n")))
    status = store.conn.execute("SELECT status, error FROM run").fetchone()
    store.close()

    assert len(results) == 1
    graded = [s for s in results[0].samples if s.evaluator == "mcqa"]
    assert graded, "the module after the broken one never ran"
    assert status["status"] == "partial", f"status was {status['status']!r}"
    assert "boom" in (status["error"] or ""), "the run does not say what failed"


def test_a_broken_module_is_recorded_as_an_errored_sample(tmp_path, monkeypatch):
    from llmbench import orchestrator, registry
    from llmbench.store import Store
    from llmbench.targets import _ENGINES

    _ENGINES["mock"] = selftest.MockTarget
    monkeypatch.setattr(orchestrator, "get_evaluator",
                        lambda name: _Boom if name == "boom" else registry.get(name))

    store = Store(str(tmp_path / "b2.db"))
    orch = orchestrator.Orchestrator(store, log=lambda *a: None)
    asyncio.run(orch.run_suite(_suite(tmp_path,
        "name: t\ntargets:\n  - engine: mock\n    base_url: http://mock-a\n"
        "    model: A\nevaluators:\n  boom:\n")))
    row = store.conn.execute(
        "SELECT error FROM sample WHERE evaluator='boom'").fetchone()
    store.close()

    assert row is not None, "the failure left no trace in the samples"
    assert "this module is broken" in row["error"]


def test_a_target_that_cannot_be_detected_does_not_end_the_sweep(tmp_path):
    """The sweep exists to compare several servers; one being down is the normal case."""
    from llmbench import orchestrator
    from llmbench.store import Store
    from llmbench.targets import _ENGINES

    class _Undetectable(selftest.MockTarget):
        async def detect(self):
            raise RuntimeError("connection refused")

    _ENGINES["mock"] = selftest.MockTarget
    _ENGINES["mock-down"] = _Undetectable

    store = Store(str(tmp_path / "d.db"))
    orch = orchestrator.Orchestrator(store, log=lambda *a: None)
    results = asyncio.run(orch.run_suite(_suite(tmp_path,
        "name: t\ntargets:\n  - engine: mock-down\n    base_url: http://down\n"
        "    model: A\n  - engine: mock\n    base_url: http://mock-a\n"
        "    model: A\nevaluators:\n  mcqa:\n")))
    store.close()

    assert len(results) == 1, "the working target did not run"
    assert results[0].samples, "the working target graded nothing"
    assert [url for url, _ in orch.failed_targets] == ["http://down"]


def test_a_machine_at_its_limit_still_produces_a_completed_run(tmp_path):
    """Phase 5's definition of done, asserted as one whole suite.

    A ceilinged server, a ladder that goes past the ceiling, and every other module in
    the suite after it. The run must complete, the unreachable rungs must be skipped
    with a reason, and the modules that follow must still have run.
    """
    from llmbench import orchestrator
    from llmbench.store import Store
    from llmbench.targets import _ENGINES

    class _Small(selftest.MockTarget):
        max_context = 4096

    _ENGINES["mock-small"] = _Small

    store = Store(str(tmp_path / "limit.db"))
    orch = orchestrator.Orchestrator(store, log=lambda *a: None)
    results = asyncio.run(orch.run_suite(_suite(tmp_path,
        "name: t\ntargets:\n  - engine: mock-small\n    base_url: http://small\n"
        "    model: A\nevaluators:\n"
        "  needle:\n    context_lengths: [2048, 8192, 16384]\n    depths: [50]\n"
        "  mcqa:\n")))
    run_row = store.conn.execute("SELECT status FROM run").fetchone()
    skipped = store.conn.execute(
        "SELECT evaluator, skipped FROM sample WHERE skipped IS NOT NULL").fetchall()
    store.close()

    assert run_row["status"] == "ok", "a machine's honest limit made the run look broken"
    assert len(results) == 1
    assert [s["evaluator"] for s in skipped] == ["needle"]
    assert "8192" in skipped[0]["skipped"], skipped[0]["skipped"]
    assert [s for s in results[0].samples
            if s.evaluator == "mcqa" and s.score is not None], \
        "the module after the ladder never ran"


def test_the_rungs_that_did_fit_are_still_measured(tmp_path):
    """Degrading gracefully is worthless if it degrades to nothing."""
    from llmbench import orchestrator
    from llmbench.store import Store
    from llmbench.targets import _ENGINES

    class _Small(selftest.MockTarget):
        max_context = 4096

    _ENGINES["mock-small"] = _Small

    store = Store(str(tmp_path / "limit2.db"))
    orch = orchestrator.Orchestrator(store, log=lambda *a: None)
    results = asyncio.run(orch.run_suite(_suite(tmp_path,
        "name: t\ntargets:\n  - engine: mock-small\n    base_url: http://small\n"
        "    model: A\nevaluators:\n"
        "  needle:\n    context_lengths: [2048, 8192, 16384]\n    depths: [50]\n")))
    store.close()

    graded = [s for s in results[0].samples
              if s.evaluator == "needle" and s.score is not None]
    assert [s.dims["context_len"] for s in graded] == [2048]
    assert graded[0].score == 1.0
    effective = [m for m in results[0].metrics if m.name == "effective_ctx"]
    assert effective and effective[0].value == 2048.0, \
        "effective_ctx should report the largest rung this machine managed"


class _BoomAggregate:
    """A module that runs fine and fails while summarising.

    ifeval really did this on 2026-08-05: it graded every sample, then raised in
    aggregate() on a skipped sample's score of None. Because aggregation sat outside
    the failure boundary, one module's summary killed the whole target and the run was
    lost - the exact outcome design D3d exists to prevent.
    """

    name = "boom_agg"

    async def evaluate(self, ctx):
        from llmbench.models import Sample
        return [Sample(evaluator=self.name, case_id="a", score=1.0)]

    def aggregate(self, samples):
        raise TypeError("unsupported operand type(s) for +: 'float' and 'NoneType'")


def test_a_module_that_fails_while_summarising_costs_only_itself(tmp_path, monkeypatch):
    from llmbench import orchestrator, registry
    from llmbench.store import Store
    from llmbench.targets import _ENGINES

    _ENGINES["mock"] = selftest.MockTarget
    monkeypatch.setattr(orchestrator, "get_evaluator",
                        lambda name: _BoomAggregate if name == "boom_agg"
                        else registry.get(name))

    store = Store(str(tmp_path / "ba.db"))
    orch = orchestrator.Orchestrator(store, log=lambda *a: None)
    results = asyncio.run(orch.run_suite(_suite(tmp_path,
        "name: t\ntargets:\n  - engine: mock\n    base_url: http://mock-a\n"
        "    model: A\nevaluators:\n  boom_agg:\n  mcqa:\n")))
    status = store.conn.execute("SELECT status, error FROM run").fetchone()
    store.close()

    assert len(results) == 1, "the run was lost because one module could not summarise"
    graded = [s for s in results[0].samples if s.evaluator == "mcqa"]
    assert graded, "the module after the broken one never ran"
    assert status["status"] == "partial", f"status was {status['status']!r}"
    assert "boom_agg" in (status["error"] or ""), "the run does not say what failed"
