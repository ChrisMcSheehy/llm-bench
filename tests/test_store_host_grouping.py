"""Quality pools across machines. Speed never does.

This is the criterion Phase 4 exists for. Every test here builds one configuration
measured on two machines, with identical quality and different speed, because that is
the only shape where a wrong grouping is visible: pool them and the speed figure
becomes an average of two machines that never existed.
"""
from __future__ import annotations

from datetime import datetime, timezone

from llmbench.models import HostFingerprint, Metric, ModelFingerprint, RunResult
from llmbench.store import Store

_FAST = HostFingerprint(
    os="Linux", arch="x86_64", cpu_count=16, total_memory_bytes=64 * 1024 ** 3,
    devices=[{"id": "CUDA0", "backend": "CUDA", "name": "NVIDIA GeForce RTX 4090",
              "total_mib": 24564, "free_mib": 24000}])
_SLOW = HostFingerprint(
    os="Linux", arch="x86_64", cpu_count=8, total_memory_bytes=16 * 1024 ** 3,
    devices=[{"id": "Vulkan0", "backend": "Vulkan", "name": "AMD Radeon RX 6600",
              "total_mib": 8192, "free_mib": 8000}])


def _fp() -> ModelFingerprint:
    return ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                            model_id="Qwen3-8B", quant="Q4_K_M", n_ctx=4096)


def _record(store: Store, run_id: str, host, speed: float, quality: float = 0.9) -> None:
    """One run of the same configuration, on the given machine."""
    run = RunResult(run_id=run_id, fingerprint=_fp(), suite="t",
                    started_at=datetime.now(timezone.utc))
    host_hash = store.upsert_host(host) if host is not None else None
    store.start_run(run, host_hash=host_hash)
    store.add_metrics(run_id, [
        Metric(evaluator="needle", name="tok_per_sec_mean", value=speed, unit="tok/s"),
        Metric(evaluator="needle", name="score_mean", value=quality),
    ])


def _two_machines(tmp_path) -> Store:
    store = Store(str(tmp_path / "g.db"))
    _record(store, "run-fast", _FAST, speed=120.0)
    _record(store, "run-slow", _SLOW, speed=30.0)
    return store


def test_quality_pools_into_one_figure_for_the_configuration(tmp_path):
    """The same model answers the same questions equally well on any machine."""
    store = _two_machines(tmp_path)
    rows = store.pooled_quality()
    store.close()

    matching = [r for r in rows if r["name"] == "score_mean"]
    assert len(matching) == 1, f"quality should pool to one row, got {matching}"
    assert matching[0]["runs"] == 2
    assert matching[0]["value"] == 0.9


def test_speed_stays_one_figure_per_machine(tmp_path):
    """The failure this phase prevents: 120 and 30 tok/s averaging to a fictional 75."""
    store = _two_machines(tmp_path)
    rows = store.pooled_speed()
    store.close()

    speeds = sorted(r["value"] for r in rows if r["name"] == "tok_per_sec_mean")
    assert speeds == [30.0, 120.0], f"speed was pooled across machines: {speeds}"
    assert len({r["host_hash"] for r in rows}) == 2


def test_a_run_with_no_machine_never_pools_with_one_that_has_it(tmp_path):
    """Every run recorded before this phase has no host. Unknown is not a machine."""
    store = Store(str(tmp_path / "u.db"))
    _record(store, "run-known", _FAST, speed=120.0)
    _record(store, "run-unknown", None, speed=118.0)
    rows = store.pooled_speed()
    store.close()

    speed_rows = [r for r in rows if r["name"] == "tok_per_sec_mean"]
    assert len(speed_rows) == 2, "a known and an unknown machine were pooled"
    assert any(r["host_hash"] is None for r in speed_rows)


def test_an_unrecognised_metric_is_treated_as_machine_dependent(tmp_path):
    """The safe default.

    Wrongly refusing to pool costs a little statistical power. Wrongly pooling hides
    two machines inside one number, which is the failure this exists to prevent - so a
    metric nobody has classified is grouped by machine.
    """
    store = Store(str(tmp_path / "n.db"))
    for run_id, host, value in (("a", _FAST, 5.0), ("b", _SLOW, 9.0)):
        run = RunResult(run_id=run_id, fingerprint=_fp(), suite="t",
                        started_at=datetime.now(timezone.utc))
        store.start_run(run, host_hash=store.upsert_host(host))
        store.add_metrics(run_id, [
            Metric(evaluator="new", name="some_future_metric", value=value)])
    pooled = [r for r in store.pooled_quality() if r["name"] == "some_future_metric"]
    by_host = [r for r in store.pooled_speed() if r["name"] == "some_future_metric"]
    store.close()

    assert pooled == [], "an unclassified metric must not pool across machines"
    assert len(by_host) == 2


def test_the_leaderboard_says_which_machine_each_row_came_from(tmp_path):
    """A human comparing two rows must not have to guess."""
    store = _two_machines(tmp_path)
    board = store.leaderboard()
    store.close()

    assert len(board) == 2
    assert {r["host_hash"] for r in board} == {_FAST.host_hash, _SLOW.host_hash}
    assert all(r["host_label"] for r in board), "no machine shown on a leaderboard row"
    assert any("4090" in r["host_label"] for r in board)


def test_the_machine_is_read_once_per_run_and_not_once_per_sample(tmp_path, monkeypatch):
    """Reading the devices starts a process (`--list-devices`).

    Doing that between test items would move the very timings the run measures, so it
    must happen once per run. Counted rather than assumed.
    """
    import asyncio

    import selftest
    from llmbench import hostinfo, orchestrator
    from llmbench.targets import _ENGINES

    _ENGINES["mock"] = selftest.MockTarget
    calls = []
    monkeypatch.setattr(hostinfo, "devices", lambda binary, **kw: calls.append(binary) or [])

    suite = tmp_path / "s.yaml"
    suite.write_text(
        "name: t\ntargets:\n  - engine: mock\n    base_url: http://mock-a\n"
        "    model: A\nevaluators:\n  mcqa:\n", encoding="utf-8")

    store = Store(str(tmp_path / "once.db"))
    orch = orchestrator.Orchestrator(store, log=lambda *a: None)
    results = asyncio.run(orch.run_suite(str(suite)))
    samples = sum(len(r.samples) for r in results)
    store.close()

    assert samples > 1, "the suite graded too few samples to tell the difference"
    assert len(calls) == len(results) == 1, (
        f"devices read {len(calls)} times for {len(results)} run(s) "
        f"and {samples} samples")
