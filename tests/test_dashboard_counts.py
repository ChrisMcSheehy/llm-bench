"""What the dashboard is given about the evidence behind each figure.

A figure and its count are served together or the figure is served alone and reads as
more solid than it is (design D7b).
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from llmbench.models import Metric, ModelFingerprint, RunResult, Sample
from llmbench.store import Store


def _store_with_a_run(tmp_path) -> Store:
    store = Store(str(tmp_path / "d.db"))
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                          model_id="Qwen3-8B", n_ctx=32768)
    store.start_run(RunResult(run_id="r1", fingerprint=fp, suite="t",
                              started_at=datetime.now(timezone.utc)))
    store.add_metrics("r1", [
        Metric(evaluator="mcqa", name="score_mean", value=0.8333, n=6),
        Metric(evaluator="coding", name="pass@1", value=0.5, n=8),
        Metric(evaluator="human", name="responses", value=3.0),
    ])
    store.add_samples("r1", [Sample(evaluator="mcqa", case_id="q1", score=1.0)])
    return store


def test_the_metrics_read_carries_the_count(tmp_path):
    store = _store_with_a_run(tmp_path)
    rows = {r["name"]: r for r in store.metrics_for("r1")}
    store.close()

    assert rows["score_mean"]["n"] == 6
    assert rows["responses"]["n"] is None, "an unstated count stays unknown"


def test_the_capability_headline_carries_the_count(tmp_path):
    """The bar chart's numbers are the ones most often read without their evidence."""
    store = _store_with_a_run(tmp_path)
    caps = {c["evaluator"]: c for c in store.capabilities("r1")}
    store.close()

    assert caps["mcqa"]["n"] == 6
    assert caps["coding"]["n"] == 8


def test_the_api_serves_the_count(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "d.db"))
    _store_with_a_run(tmp_path).close()

    from llmbench.dashboard.app import app
    rows = TestClient(app).get("/api/run/r1/metrics").json()

    assert {r["name"]: r["n"] for r in rows}["score_mean"] == 6


def test_the_heatmap_says_how_many_samples_each_cell_holds(tmp_path):
    """A cell from one sample and a cell from five are the same colour."""
    store = Store(str(tmp_path / "h.db"))
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                          model_id="Qwen3-8B", n_ctx=32768)
    store.start_run(RunResult(run_id="r1", fingerprint=fp, suite="t",
                              started_at=datetime.now(timezone.utc)))
    store.add_samples("r1", [
        Sample(evaluator="needle", case_id="2048:50:0", score=1.0,
               dims={"context_len": 2048, "depth_pct": 50}),
        Sample(evaluator="needle", case_id="2048:50:1", score=0.0,
               dims={"context_len": 2048, "depth_pct": 50}),
        Sample(evaluator="needle", case_id="8192:50:0", score=1.0,
               dims={"context_len": 8192, "depth_pct": 50}),
    ])
    heat = store.view_data("r1", "needle", x="context_len", y="depth_pct")
    store.close()

    assert heat["z"] == [[0.5, 1.0]], heat["z"]
    assert heat["n"] == [[2, 1]], "the cell counts do not match the cells"


def test_a_cell_nobody_probed_counts_zero_rather_than_one(tmp_path):
    """The count must not inherit the divide-by-zero guard the average uses."""
    store = Store(str(tmp_path / "g.db"))
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                          model_id="Qwen3-8B", n_ctx=32768)
    store.start_run(RunResult(run_id="r1", fingerprint=fp, suite="t",
                              started_at=datetime.now(timezone.utc)))
    store.add_samples("r1", [
        Sample(evaluator="needle", case_id="2048:0", score=1.0,
               dims={"context_len": 2048, "depth_pct": 0}),
        Sample(evaluator="needle", case_id="8192:50", score=1.0,
               dims={"context_len": 8192, "depth_pct": 50}),
    ])
    heat = store.view_data("r1", "needle", x="context_len", y="depth_pct")
    store.close()

    # Two lengths x two depths, and only two of the four corners were probed.
    assert sorted(sum(heat["n"], [])) == [0, 0, 1, 1], heat["n"]


def test_the_run_list_carries_the_memory_estimate(tmp_path):
    """Computed from the model's header since Phase 3 and displayed nowhere since."""
    store = Store(str(tmp_path / "m.db"))
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                          model_id="Qwen3-8B", n_ctx=32768, kv_cache_bytes=1610612736)
    store.start_run(RunResult(run_id="r1", fingerprint=fp, suite="t",
                              started_at=datetime.now(timezone.utc)))
    rows = store.runs()
    store.close()

    assert rows[0]["kv_cache_bytes"] == 1610612736


def test_a_configuration_with_no_memory_figure_reports_none_not_zero(tmp_path):
    """Unknown and 'costs nothing' must not print the same way."""
    store = Store(str(tmp_path / "m2.db"))
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                          model_id="Qwen3-8B", n_ctx=32768)
    store.start_run(RunResult(run_id="r1", fingerprint=fp, suite="t",
                              started_at=datetime.now(timezone.utc)))
    rows = store.runs()
    store.close()

    assert rows[0]["kv_cache_bytes"] is None
