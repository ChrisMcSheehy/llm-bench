"""A rung that was never attempted, stored as such.

The reason travels with the gap. A skip with no reason is indistinguishable from a row
that was never written, and explaining the gap is the whole point of design D3.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from llmbench.models import ModelFingerprint, RunResult, Sample
from llmbench.store import Store


def _run(store: Store, run_id: str = "r1") -> None:
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                          model_id="Qwen3-8B", n_ctx=4096)
    store.start_run(RunResult(run_id=run_id, fingerprint=fp, suite="t",
                              started_at=datetime.now(timezone.utc)))


def test_a_skipped_sample_keeps_its_reason(tmp_path):
    store = Store(str(tmp_path / "s.db"))
    _run(store)
    store.add_samples("r1", [Sample(evaluator="needle", case_id="262144:skipped",
                                    skipped="not attempted: the 131072-token rung failed")])
    row = store.conn.execute(
        "SELECT skipped, error, score FROM sample WHERE run_id='r1'").fetchone()
    store.close()

    assert row["skipped"] == "not attempted: the 131072-token rung failed"
    assert row["error"] is None, "a skip is not an error"
    assert row["score"] is None, "a skip has no score, and must never be stored as zero"


def test_the_field_is_read_back_from_the_model_not_merely_accepted():
    """pydantic ignores an unknown argument, so passing one proves nothing on its own."""
    assert Sample(evaluator="e", case_id="c", skipped="because").skipped == "because"
    assert Sample(evaluator="e", case_id="c").skipped is None


def test_an_older_database_gains_the_skipped_column(tmp_path):
    """CREATE TABLE IF NOT EXISTS does nothing to a database that already exists."""
    db = tmp_path / "old.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE sample (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT,"
        " evaluator TEXT, case_id TEXT, grp TEXT, dims_json TEXT,"
        " input_tokens INTEGER, output_tokens INTEGER, latency_ms REAL,"
        " tok_per_sec REAL, server_prompt_tps REAL, server_gen_tps REAL,"
        " score REAL, passed INTEGER, error TEXT, meta_json TEXT, created_at TEXT);"
        "INSERT INTO sample (run_id, evaluator) VALUES ('old-run', 'needle');")
    conn.commit()
    conn.close()

    store = Store(str(db))
    columns = {r[1] for r in store.conn.execute("PRAGMA table_info(sample)")}
    kept = store.conn.execute("SELECT run_id FROM sample").fetchall()
    store.close()

    assert "skipped" in columns, "migration did not add the skipped column"
    assert [tuple(r) for r in kept] == [("old-run",)], "migration dropped existing rows"
