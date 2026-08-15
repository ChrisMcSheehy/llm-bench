"""What the dashboard does with a rung that was never attempted.

A gap with no explanation reads as a figure of zero, which is the one thing this
project never displays.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from llmbench.models import ModelFingerprint, RunResult, Sample
from llmbench.store import Store


def _store_with_a_skipped_rung(tmp_path) -> Store:
    store = Store(str(tmp_path / "d.db"))
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                          model_id="Qwen3-8B", n_ctx=32768)
    store.start_run(RunResult(run_id="r1", fingerprint=fp, suite="t",
                              started_at=datetime.now(timezone.utc)))
    store.add_samples("r1", [
        Sample(evaluator="needle", case_id="2048:50:0", group="2048", score=1.0,
               passed=True, dims={"context_len": 2048, "depth_pct": 50}),
        Sample(evaluator="needle", case_id="16384:skipped", group="16384",
               dims={"context_len": 16384},
               skipped="not attempted: the 8192-token rung failed (out of memory)"),
    ])
    return store


def test_the_heatmap_leaves_the_unattempted_rung_out_rather_than_drawing_a_zero(tmp_path):
    """A skipped row has no depth, so including it also crashes the query."""
    store = _store_with_a_skipped_rung(tmp_path)
    heat = store.needle_heatmap("r1")
    store.close()

    assert heat["lengths"] == [2048], f"an unattempted rung was drawn: {heat['lengths']}"


def test_the_reason_is_available_for_the_run(tmp_path):
    store = _store_with_a_skipped_rung(tmp_path)
    rows = store.skipped("r1")
    store.close()

    assert len(rows) == 1
    assert rows[0]["evaluator"] == "needle"
    assert "8192" in rows[0]["skipped"]


def test_the_api_serves_the_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "d.db"))
    _store_with_a_skipped_rung(tmp_path).close()

    from llmbench.dashboard.app import app
    rows = TestClient(app).get("/api/run/r1/skipped").json()

    assert len(rows) == 1
    assert "8192" in rows[0]["skipped"]
