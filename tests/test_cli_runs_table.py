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


def test_a_proportion_prints_the_range_its_sample_supports():
    """0.833 over six items is one question away from 0.667, and the interval is
    what says so on the page rather than in the reader's head (design C8)."""
    from llmbench.cli import _fmt
    out = _fmt(0.833, n=6, successes=5)

    assert "0.833" in out
    assert "0.44" in out and "0.97" in out, f"no interval printed:\n{out}"
    assert "n=6" in out


def test_a_figure_with_no_numerator_prints_no_interval():
    """decode_tps is not a proportion, and a range around it would be arithmetic on
    the wrong kind of number."""
    from llmbench.cli import _fmt
    out = _fmt(41.2, n=3)

    assert "41.2" in out
    assert "," not in out.replace("[dim]", "").replace("[/dim]", ""), out


def test_a_run_stored_before_this_change_prints_its_figure_and_no_interval():
    from llmbench.cli import _fmt
    out = _fmt(0.83, n=6, successes=None)

    assert "0.830" in out
    assert "n=6" in out
    assert "[0." not in out, f"an interval was invented from no numerator:\n{out}"


def test_the_table_prints_the_interval_beside_the_figure(tmp_path, monkeypatch):
    """Through the real command, not just the formatter: the numerator has to reach
    the table through leaderboard() as well as be formattable once it arrives."""
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "ci.db"))
    store = Store(str(tmp_path / "ci.db"))
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                          model_id="Qwen3-8B", model_name="Qwen3-8B", n_ctx=8192)
    store.start_run(RunResult(run_id="r1", fingerprint=fp, suite="t",
                              started_at=datetime.now(timezone.utc)))
    store.add_metrics("r1", [
        Metric(evaluator="needle", name="score_mean", value=0.8333, n=6, successes=5)])
    store.close()

    out = CliRunner().invoke(app, ["runs"], env={"COLUMNS": "200"}).stdout
    assert "0.44" in out and "0.97" in out, f"no interval in the table:\n{out}"
