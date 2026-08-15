"""How much testing each configuration has actually received.

Derived from the runs and samples already stored, so it cannot be inflated and cannot
go stale (design D9a).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from llmbench.models import ModelFingerprint, RunResult, Sample
from llmbench.store import Store

_T0 = datetime(2026, 3, 1, tzinfo=timezone.utc)


def _fp(quant: str = "Q4_K_M") -> ModelFingerprint:
    return ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                            model_id=f"Qwen3-8B-{quant}", quant=quant, n_ctx=8192)


def _run(store: Store, run_id: str, fp: ModelFingerprint, *, days: int = 0,
         status: str = "ok", host: str = "host-a") -> None:
    store.start_run(RunResult(run_id=run_id, fingerprint=fp, suite="t",
                              started_at=_T0 + timedelta(days=days)),
                    host_hash=host)
    store.finish_run(run_id, status, (_T0 + timedelta(days=days)).isoformat())


def _effort(store: Store) -> dict[str, dict]:
    return {r["fp_hash"]: r for r in store.configuration_effort()}


def test_a_configuration_counts_its_runs(tmp_path):
    store = Store(str(tmp_path / "e.db"))
    fp = _fp()
    _run(store, "r1", fp)
    _run(store, "r2", fp, days=1)
    rows = _effort(store)
    store.close()

    assert rows[fp.fingerprint_hash]["runs"] == 2


def test_failed_and_partial_runs_are_counted_openly_and_still_count_as_runs(tmp_path):
    """'Six runs, two of which errored' is information; showing only four is not."""
    store = Store(str(tmp_path / "e.db"))
    fp = _fp()
    _run(store, "r1", fp)
    _run(store, "r2", fp, days=1, status="error")
    _run(store, "r3", fp, days=2, status="partial")
    rows = _effort(store)
    store.close()

    row = rows[fp.fingerprint_hash]
    assert (row["runs"], row["failed_runs"], row["partial_runs"]) == (3, 1, 1)


def test_the_sample_total_excludes_what_was_never_graded(tmp_path):
    """A skipped rung and a failed sample are not evidence."""
    store = Store(str(tmp_path / "e.db"))
    fp = _fp()
    _run(store, "r1", fp)
    _run(store, "r2", fp, days=1)
    store.add_samples("r1", [
        Sample(evaluator="mcqa", case_id="q1", score=1.0),
        Sample(evaluator="mcqa", case_id="q2", score=0.0),
        Sample(evaluator="mcqa", case_id="q3", error="boom"),
        Sample(evaluator="needle", case_id="8192:skipped", skipped="not attempted"),
    ])
    store.add_samples("r2", [Sample(evaluator="mcqa", case_id="q1", score=1.0)])
    rows = _effort(store)
    store.close()

    assert rows[fp.fingerprint_hash]["graded_samples"] == 3


def test_the_span_covers_the_first_and_last_run(tmp_path):
    store = Store(str(tmp_path / "e.db"))
    fp = _fp()
    _run(store, "r1", fp, days=10)
    _run(store, "r2", fp, days=0)
    rows = _effort(store)
    store.close()

    row = rows[fp.fingerprint_hash]
    assert row["first_run"][:10] == "2026-03-01"
    assert row["last_run"][:10] == "2026-03-11"


def test_a_run_with_no_machine_counts_as_a_run_and_not_as_a_machine(tmp_path):
    """Unknown is not a machine (design D9b)."""
    store = Store(str(tmp_path / "e.db"))
    fp = _fp()
    _run(store, "r1", fp, host="host-a")
    _run(store, "r2", fp, days=1, host="host-b")
    _run(store, "r3", fp, days=2, host=None)
    rows = _effort(store)
    store.close()

    row = rows[fp.fingerprint_hash]
    assert row["runs"] == 3
    assert row["machines"] == 2
