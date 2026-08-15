"""How many test items a figure rests on, stored with the figure.

The bundled test sets are tiny. An accuracy of 0.83 over six questions is one question
away from 0.67, and published without its count it reads as far more solid than it is
(design D7).
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


def test_the_field_is_read_back_from_the_model_not_merely_accepted():
    """pydantic ignores an unknown argument, so passing one proves nothing on its own."""
    assert Metric(evaluator="e", name="score_mean", value=0.5, n=6).n == 6
    assert Metric(evaluator="e", name="score_mean", value=0.5).n is None


def test_a_stored_metric_keeps_its_count(tmp_path):
    store = Store(str(tmp_path / "s.db"))
    _run(store)
    store.add_metrics("r1", [
        Metric(evaluator="mcqa", name="score_mean", value=0.8333, n=6),
        Metric(evaluator="mcqa", name="mystery", value=1.0),
    ])
    rows = {r["name"]: r["n"] for r in store.conn.execute(
        "SELECT name, n FROM metric WHERE run_id='r1'")}
    store.close()

    assert rows["score_mean"] == 6
    assert rows["mystery"] is None, "an unstated count is unknown, never zero"


def test_an_older_database_gains_the_n_column(tmp_path):
    """CREATE TABLE IF NOT EXISTS does nothing to a database that already exists."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE metric (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,"
        " evaluator TEXT, name TEXT, value REAL, unit TEXT, dims_json TEXT);"
        "INSERT INTO metric (run_id, name) VALUES ('old-run', 'score_mean');")
    conn.commit()
    conn.close()

    store = Store(str(db))
    columns = {r[1] for r in store.conn.execute("PRAGMA table_info(metric)")}
    kept = store.conn.execute("SELECT run_id FROM metric").fetchall()
    store.close()

    assert "n" in columns, "migration did not add the n column"
    assert [tuple(r) for r in kept] == [("old-run",)], "migration dropped existing rows"
