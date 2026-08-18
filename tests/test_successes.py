"""The numerator that makes an interval possible (design C1).

A rate and a count cannot produce a confidence interval on their own once the rate has
been rounded, and averaging rates across runs destroys the denominator entirely. The
numerator is known for free where the rate is computed and nowhere else without guessing.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from llmbench.models import Metric, ModelFingerprint, RunResult
from llmbench.store import Store


def _run(store: Store, run_id: str = "r1") -> None:
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                          model_id="Qwen3-8B", n_ctx=4096)
    store.start_run(RunResult(run_id=run_id, fingerprint=fp, suite="t",
                              started_at=datetime.now(timezone.utc)))


def test_the_metric_type_has_a_numerator():
    """Read back, never merely passed: pydantic ignores an unknown argument by
    default, so constructing with successes= proves nothing on its own
    (LESSONS: a-permissive-model-swallows-the-field-a-red-test-needs)."""
    assert "successes" in Metric.model_fields
    m = Metric(evaluator="mcqa", name="accuracy", value=0.83, n=6, successes=5)
    assert m.successes == 5


def test_a_figure_that_is_not_a_proportion_has_no_numerator():
    """None is the honest default. Zero would claim no successes were counted."""
    m = Metric(evaluator="speed", name="decode_tps", value=41.2, n=3)
    assert m.successes is None


def test_the_numerator_survives_a_round_trip_through_the_store(tmp_path):
    store = Store(str(tmp_path / "s.db"))
    _run(store)
    store.add_metrics("r1", [
        Metric(evaluator="mcqa", name="accuracy", value=0.83, n=6, successes=5),
        Metric(evaluator="speed", name="decode_tps", value=41.2, n=3),
    ])
    got = {r["name"]: r["successes"] for r in store.conn.execute(
        "SELECT name, successes FROM metric WHERE run_id='r1'")}
    store.close()

    assert got["accuracy"] == 5
    assert got["decode_tps"] is None


def test_the_numerator_reaches_the_dashboard(tmp_path):
    """metrics_for selects an explicit column list rather than *, so the column has
    to be named there or the figure arrives at the dashboard without it."""
    store = Store(str(tmp_path / "s.db"))
    _run(store)
    store.add_metrics("r1", [
        Metric(evaluator="mcqa", name="accuracy", value=0.83, n=6, successes=5)])
    rows = {r["name"]: r for r in store.metrics_for("r1")}
    store.close()

    assert rows["accuracy"]["successes"] == 5


def test_a_database_written_before_this_change_still_opens(tmp_path):
    """The migration adds the column to an existing file. CREATE TABLE IF NOT EXISTS
    does nothing to a database that already exists, so without the _ADDED_COLUMNS
    entry the column would reach only databases created after today."""
    db = tmp_path / "old.db"
    Store(str(db)).close()

    conn = sqlite3.connect(str(db))
    conn.execute("ALTER TABLE metric DROP COLUMN successes")
    conn.commit()
    conn.close()

    reopened = Store(str(db))
    cols = {r[1] for r in reopened.conn.execute("PRAGMA table_info(metric)")}
    reopened.close()

    assert "successes" in cols
