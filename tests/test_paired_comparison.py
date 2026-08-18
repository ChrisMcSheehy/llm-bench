"""Whether the difference between two runs is one the questions can actually show.

C-1 gave every rate an interval, which says how solid one figure is. It does not say
whether two figures differ: overlapping intervals are not a verdict, and treating two
runs as independent throws away the fact that both answered the same questions.
"""
from __future__ import annotations

from datetime import datetime, timezone

from llmbench.models import ModelFingerprint, RunResult, Sample
from llmbench.store import Store


def _fp(quant: str = "Q4_K_M") -> ModelFingerprint:
    return ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                            model_id="Qwen3-8B", quant=quant, n_ctx=4096)


def _run(store: Store, run_id: str, scores: dict[str, float | None],
         evaluator: str = "mcqa", **overrides) -> None:
    """One run whose case_ids are the keys of `scores`.

    A None score means the case exists but was not graded, which is how a skip or an
    error reaches the pairing.
    """
    store.start_run(RunResult(run_id=run_id, fingerprint=_fp(run_id), suite="t",
                              started_at=datetime.now(timezone.utc)))
    store.add_samples(run_id, [
        Sample(evaluator=evaluator, case_id=cid, score=s,
               passed=None if s is None else bool(s), answered=s is not None,
               skipped=overrides.get(cid), )
        for cid, s in scores.items()])


def _outcome(store: Store, a: str = "a", b: str = "b", evaluator: str = "mcqa") -> dict:
    rows = {r["evaluator"]: r for r in store.paired_outcomes(a, b)}
    return rows.get(evaluator, {})


def test_one_disagreement_in_six_cannot_separate_two_runs(tmp_path):
    """The headline case. Five of six against four of six looks like a 16-point gap
    and is one question; nothing in this bench could say so before."""
    store = Store(str(tmp_path / "c.db"))
    _run(store, "a", {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0, "q5": 1.0, "q6": 0.0})
    _run(store, "b", {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0, "q5": 0.0, "q6": 0.0})
    got = _outcome(store)
    store.close()

    assert got["paired"] == 6
    assert got["a_only"] == 1 and got["b_only"] == 0
    assert got["p"] == 1.0
    assert got["distinguishable"] is False


def test_twenty_wins_against_five_is_a_real_difference(tmp_path):
    store = Store(str(tmp_path / "c.db"))
    a, b = {}, {}
    for i in range(20):                      # a right, b wrong
        a[f"x{i}"], b[f"x{i}"] = 1.0, 0.0
    for i in range(5):                       # b right, a wrong
        a[f"y{i}"], b[f"y{i}"] = 0.0, 1.0
    _run(store, "a", a)
    _run(store, "b", b)
    got = _outcome(store)
    store.close()

    assert got["a_only"] == 20 and got["b_only"] == 5
    assert got["p"] < 0.01
    assert got["distinguishable"] is True


def test_a_run_compared_with_itself_finds_no_difference(tmp_path):
    """Zero disagreements. Not a division by zero, and not significance."""
    store = Store(str(tmp_path / "c.db"))
    _run(store, "a", {"q1": 1.0, "q2": 0.0, "q3": 1.0})
    _run(store, "b", {"q1": 1.0, "q2": 0.0, "q3": 1.0})
    got = _outcome(store)
    store.close()

    assert got["a_only"] == 0 and got["b_only"] == 0
    assert got["paired"] == 3
    assert got["p"] == 1.0
    assert got["distinguishable"] is False


def test_only_cases_both_runs_graded_are_paired(tmp_path):
    """A pair needs two results. A case one run never graded is a gap, and counting it
    as a loss would punish a machine for its own limits (design D3)."""
    store = Store(str(tmp_path / "c.db"))
    _run(store, "a", {"q1": 1.0, "q2": 1.0, "q3": 1.0})
    _run(store, "b", {"q1": 1.0, "q2": None, "q3": 0.0})
    got = _outcome(store)
    store.close()

    assert got["paired"] == 2, "the ungraded case should not have been paired"
    assert got["a_only"] == 1


def test_the_shared_subset_is_reported_rather_than_silently_used(tmp_path):
    """Two runs sharing three of many questions is a fact that changes how the verdict
    should be read (design C6)."""
    store = Store(str(tmp_path / "c.db"))
    _run(store, "a", {f"q{i}": 1.0 for i in range(10)})
    _run(store, "b", {f"q{i}": 1.0 for i in range(7, 20)})
    got = _outcome(store)
    store.close()

    assert got["paired"] == 3
    assert got["a_total"] == 10 and got["b_total"] == 13


def test_a_case_scored_on_a_gradient_is_excluded_and_counted(tmp_path):
    """McNemar needs right-or-wrong. A bit accuracy of 0.62 against 0.71 is a real
    difference and this is not the instrument for it, so it is left out and said to
    have been left out rather than rounded into a verdict."""
    store = Store(str(tmp_path / "c.db"))
    _run(store, "a", {"q1": 1.0, "q2": 0.62}, evaluator="reassembly")
    _run(store, "b", {"q1": 0.0, "q2": 0.71}, evaluator="reassembly")
    got = _outcome(store, evaluator="reassembly")
    store.close()

    assert got["paired"] == 1
    assert got["excluded"] == 1
    assert got["a_only"] == 1


def test_two_runs_sharing_nothing_report_no_pairs_rather_than_agreement(tmp_path):
    """An empty result must not read like 'no difference found'
    (LESSONS: assert-the-success-condition-not-the-absence-of-error)."""
    store = Store(str(tmp_path / "c.db"))
    _run(store, "a", {"q1": 1.0})
    _run(store, "b", {"z9": 1.0})
    got = _outcome(store)
    store.close()

    assert got["paired"] == 0
    assert got["p"] is None, "no pairs means no probability, not a probability of 1"
    assert got["distinguishable"] is False


# ---- the command (design C5) ----------------------------------------------
#
# What may be claimed matters as much as the arithmetic. A verdict printed without the
# count behind it is exactly as misleading as an accuracy printed without its n.

from typer.testing import CliRunner

from llmbench.cli import app


def _cli(tmp_path, monkeypatch, a: dict, b: dict, evaluator: str = "mcqa") -> str:
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "cmp.db"))
    store = Store(str(tmp_path / "cmp.db"))
    _run(store, "a", a, evaluator=evaluator)
    _run(store, "b", b, evaluator=evaluator)
    store.close()
    return CliRunner().invoke(app, ["compare", "a", "b"],
                              env={"COLUMNS": "200"}).stdout


def test_it_never_claims_two_runs_are_the_same(tmp_path, monkeypatch):
    """A non-significant result is not evidence of equivalence. At six questions
    nothing is distinguishable, and printing "no difference" would convert this
    bench's lack of data into a claim about the model (design C5)."""
    out = _cli(tmp_path, monkeypatch,
               {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0, "q5": 1.0, "q6": 0.0},
               {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0, "q5": 0.0, "q6": 0.0})

    assert "indistinguishable" in out, out
    for forbidden in ("equivalent", "the same", "no difference"):
        assert forbidden not in out.lower(), f"claimed {forbidden!r}:\n{out}"


def test_the_verdict_carries_the_counts_behind_it(tmp_path, monkeypatch):
    out = _cli(tmp_path, monkeypatch,
               {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0, "q5": 1.0, "q6": 0.0},
               {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0, "q5": 0.0, "q6": 0.0})

    assert "paired on all 6 graded" in out, out
    assert "1 disagreement" in out, out
    assert "6 questions cannot separate these" in out, out


def test_a_real_difference_is_reported_as_one(tmp_path, monkeypatch):
    a = {f"x{i}": 1.0 for i in range(20)} | {f"y{i}": 0.0 for i in range(5)}
    b = {f"x{i}": 0.0 for i in range(20)} | {f"y{i}": 1.0 for i in range(5)}
    out = _cli(tmp_path, monkeypatch, a, b)

    assert "A is ahead" in out, out
    assert "indistinguishable" not in out, out


def test_questions_left_out_are_said_to_have_been_left_out(tmp_path, monkeypatch):
    """A comparison that silently dropped gradient-scored cases is not the comparison
    the reader thinks they are reading."""
    out = _cli(tmp_path, monkeypatch,
               {"q1": 1.0, "q2": 0.62}, {"q1": 0.0, "q2": 0.71},
               evaluator="reassembly")

    assert "left out" in out, out
    assert "gradient" in out, out


def test_an_unknown_run_is_refused_rather_than_compared_with_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "cmp.db"))
    Store(str(tmp_path / "cmp.db")).close()
    result = CliRunner().invoke(app, ["compare", "nope", "also-nope"])

    assert result.exit_code == 1
    assert "No run" in result.stdout


def test_one_of_anything_is_not_printed_as_a_plural(tmp_path, monkeypatch):
    """"1 questions cannot separate these" passes a substring assertion and reads as a
    defect to anyone looking at it. Caught by reading the real output, not by a test."""
    out = _cli(tmp_path, monkeypatch, {"q1": 1.0}, {"q1": 0.0})

    assert "1 question cannot separate these" in out, out
    assert "1 questions" not in out, out
    assert "1 disagreement " in out and "1 disagreements" not in out, out


def test_a_partial_overlap_spells_out_both_totals(tmp_path, monkeypatch):
    """Two runs sharing three of many is the case where the intersection matters."""
    out = _cli(tmp_path, monkeypatch,
               {f"q{i}": 1.0 for i in range(10)},
               {f"q{i}": 1.0 for i in range(7, 20)})

    assert "paired on 3 of 10 and 13 graded" in out, out


# ---- the dashboard endpoint (design C8, L1, B6) ----------------------------

from fastapi.testclient import TestClient


def _api(tmp_path, monkeypatch, a: dict, b: dict):
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "api.db"))
    store = Store(str(tmp_path / "api.db"))
    _run(store, "a", a)
    _run(store, "b", b)
    store.close()
    from llmbench.dashboard.app import app as api
    return TestClient(api)


def test_the_endpoint_serves_the_paired_verdict(tmp_path, monkeypatch):
    client = _api(tmp_path, monkeypatch,
                  {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0, "q5": 1.0, "q6": 0.0},
                  {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0, "q5": 0.0, "q6": 0.0})
    rows = {r["evaluator"]: r for r in client.get("/api/compare/a/b").json()}

    assert rows["mcqa"]["paired"] == 6
    assert rows["mcqa"]["distinguishable"] is False


def test_the_endpoint_refuses_a_run_it_does_not_hold(tmp_path, monkeypatch):
    """An empty comparison reads exactly like agreement, so a name the store does not
    know is refused rather than answered."""
    client = _api(tmp_path, monkeypatch, {"q1": 1.0}, {"q1": 1.0})
    response = client.get("/api/compare/a/invented")

    assert response.status_code == 404
    assert "invented" in response.json()["detail"]


def test_the_endpoint_takes_names_and_nothing_else(tmp_path, monkeypatch):
    """A path traversal or a smuggled query is a run id that does not exist, and is
    refused by the same check as any other unknown name (design L1, B6)."""
    client = _api(tmp_path, monkeypatch, {"q1": 1.0}, {"q1": 1.0})

    for attempt in ("../../etc/passwd", "a' OR '1'='1", "a;b"):
        assert client.get(f"/api/compare/{attempt}/b").status_code in (404, 400)
