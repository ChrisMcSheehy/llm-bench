"""What `llmbench runs` prints beside each figure.

This table is the output most likely to be pasted somewhere the reader cannot ask
follow-up questions, so it carries the machine and the counts (design D7).
"""
from __future__ import annotations

from datetime import datetime, timezone

from typer.testing import CliRunner

from llmbench.cli import app
from llmbench.models import Metric, ModelFingerprint, RunResult
from llmbench.store import Store


def _db(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "cli.db"))
    store = Store(str(tmp_path / "cli.db"))
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                          model_id="Qwen3-8B", model_name="Qwen3-8B", quant="Q4_K_M",
                          n_ctx=8192)
    store.start_run(RunResult(run_id="r1", fingerprint=fp, suite="t",
                              started_at=datetime.now(timezone.utc)))
    store.add_metrics("r1", [
        Metric(evaluator="needle", name="score_mean", value=0.8333, n=6),
        Metric(evaluator="coding", name="pass@1", value=0.5, n=8),
    ])
    store.close()


def test_the_table_prints_each_figure_with_its_count(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    out = CliRunner().invoke(app, ["runs"], env={"COLUMNS": "200"}).stdout

    assert "0.833" in out, out
    assert "n=6" in out, f"the needle figure was printed without its count:\n{out}"
    assert "n=8" in out, f"the coding figure was printed without its count:\n{out}"


def test_a_run_with_no_machine_says_so_rather_than_leaving_a_blank(tmp_path, monkeypatch):
    """A blank column reads as 'no machine involved', which is never true."""
    _db(tmp_path, monkeypatch)
    out = CliRunner().invoke(app, ["runs"], env={"COLUMNS": "200"}).stdout

    assert "unknown" in out, f"the machine column was left blank:\n{out}"
