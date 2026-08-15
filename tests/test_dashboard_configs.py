"""The configurations view: what has been tested, how much, and on what.

Phase 4 built pooled quality and speed figures and nothing ever displayed them.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from llmbench.models import Metric, ModelFingerprint, RunResult, Sample
from llmbench.store import Store


def _seed(tmp_path) -> None:
    store = Store(str(tmp_path / "c.db"))
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                          model_id="Qwen3-8B", quant="Q4_K_M", n_ctx=8192)
    store.start_run(RunResult(run_id="r1", fingerprint=fp, suite="t",
                              started_at=datetime.now(timezone.utc)),
                    host_hash="host-a")
    store.finish_run("r1", "ok", datetime.now(timezone.utc).isoformat())
    store.add_metrics("r1", [Metric(evaluator="mcqa", name="score_mean",
                                    value=0.8333, n=6)])
    store.add_samples("r1", [Sample(evaluator="mcqa", case_id="q1", score=1.0)])
    store.close()


def test_the_api_serves_effort_and_both_kinds_of_pooled_figure(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "c.db"))
    _seed(tmp_path)

    from llmbench.dashboard.app import app
    body = TestClient(app).get("/api/configurations").json()

    assert body["effort"][0]["runs"] == 1
    assert body["effort"][0]["graded_samples"] == 1
    assert any(q["name"] == "score_mean" for q in body["quality"])


def test_a_pooled_quality_figure_carries_the_items_behind_it(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "c.db"))
    _seed(tmp_path)

    from llmbench.dashboard.app import app
    body = TestClient(app).get("/api/configurations").json()

    row = [q for q in body["quality"] if q["name"] == "score_mean"][0]
    assert row["items"] == 6, "the pooled figure did not carry its evidence"


def test_the_page_is_served(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "c.db"))
    _seed(tmp_path)

    from llmbench.dashboard.app import app
    r = TestClient(app).get("/configs")

    assert r.status_code == 200
    assert "configurations" in r.text.lower()
