"""A test module gets a chart by declaring one, not by editing three other files.

Design E3. The plugin mechanism was only ever half true: a file dropped into
`evaluators/` was discovered and run automatically, and then stopped at the generic
leaderboard. Giving it a chart meant editing the store's SQL, adding an endpoint and
editing the dashboard HTML — three files its author was never told about. The project
described its test modules as self-contained plugins, which was true of running them and
false of presenting them.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from llmbench import registry
from llmbench.dashboard.app import app
from llmbench.evaluators.base import Evaluator, View
from llmbench.models import ModelFingerprint, RunResult, Sample
from llmbench.store import Store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMBENCH_DB", str(tmp_path / "v.db"))
    s = Store(str(tmp_path / "v.db"))
    fp = ModelFingerprint(engine="llama.cpp", base_url="http://localhost:8080",
                          model_id="Qwen3-8B", n_ctx=32768)
    s.start_run(RunResult(run_id="r1", fingerprint=fp, suite="t",
                          started_at=datetime.now(timezone.utc)))
    yield s
    s.close()


def _needle(length: int, depth: int, score: float) -> Sample:
    return Sample(evaluator="needle", case_id=f"{length}:{depth}", score=score,
                  passed=score >= 0.5, dims={"context_len": length, "depth_pct": depth})


# ---- what the generic query does --------------------------------------------

def test_a_two_dimension_view_averages_each_cell_and_counts_it(store):
    store.add_samples("r1", [_needle(2048, 50, 1.0), _needle(2048, 50, 0.0),
                             _needle(8192, 50, 1.0)])
    data = store.view_data("r1", "needle", x="context_len", y="depth_pct")

    assert data["x"] == [2048, 8192]
    assert data["z"] == [[0.5, 1.0]]
    assert data["n"] == [[2, 1]], "a colour from one attempt must be tellable from five"


def test_a_cell_nobody_probed_is_unknown_rather_than_zero(store):
    """The old hand-written heatmap put 0.0 in an unprobed cell, which draws as a
    confident dark square meaning "this failed". No measurement is not a score of nought,
    and the renderer draws a gap."""
    store.add_samples("r1", [_needle(2048, 0, 1.0), _needle(8192, 50, 1.0)])
    data = store.view_data("r1", "needle", x="context_len", y="depth_pct")

    flat_values = [v for row in data["z"] for v in row]
    assert None in flat_values, f"an unprobed cell carries a figure: {data['z']}"
    assert sorted(v for row in data["n"] for v in row) == [0, 0, 1, 1]


def test_a_numeric_axis_sorts_as_numbers(store):
    """Sorted as text, 1024 lands before 512 and the ladder reads backwards."""
    store.add_samples("r1", [_needle(512, 50, 1.0), _needle(1024, 50, 1.0),
                             _needle(131072, 50, 1.0)])
    data = store.view_data("r1", "needle", x="context_len", y="depth_pct")

    assert data["x"] == [512, 1024, 131072]


def test_a_sample_without_the_dimension_is_left_out(store):
    """A skipped rung carries a context length and no depth, so it belongs in no cell of
    a length-by-depth view — and including it also breaks the grouping."""
    store.add_samples("r1", [
        _needle(2048, 50, 1.0),
        Sample(evaluator="needle", case_id="16384:skipped", dims={"context_len": 16384},
               skipped="not attempted: the 8192-token rung failed"),
    ])
    data = store.view_data("r1", "needle", x="context_len", y="depth_pct")

    assert data["x"] == [2048], f"an unattempted rung was drawn: {data['x']}"


def test_a_one_dimension_view_is_a_flat_series(store):
    store.add_samples("r1", [
        Sample(evaluator="mcqa", case_id="a", score=1.0, dims={"subject": "physics"}),
        Sample(evaluator="mcqa", case_id="b", score=0.0, dims={"subject": "physics"}),
        Sample(evaluator="mcqa", case_id="c", score=1.0, dims={"subject": "chemistry"}),
    ])
    data = store.view_data("r1", "mcqa", x="subject")

    assert data["x"] == ["chemistry", "physics"]
    assert data["v"] == [1.0, 0.5]
    assert data["n"] == [1, 2]


def test_only_a_known_field_may_be_averaged(store):
    """A view is module code rather than user input, but a column name assembled from a
    string is not a habit worth having."""
    with pytest.raises(ValueError):
        store.view_data("r1", "needle", x="context_len", value="1; DROP TABLE sample")


# ---- what a module has to do to get a chart ---------------------------------

def test_a_declaration_with_an_unknown_kind_fails_when_the_module_loads():
    """At import rather than at render, so a typo is found by the person who made it and
    not by whoever opens the dashboard a week later."""
    with pytest.raises(ValueError):
        View("scatterplot3d", "nope", x="context_len")


def test_a_declaration_may_only_average_a_field_that_exists():
    with pytest.raises(ValueError):
        View("bar", "nope", x="subject", value="vibes")


@pytest.fixture
def a_third_party_module():
    """Register an evaluator the way an installed plugin would, then remove it.

    Reaching into the registry directly is deliberate: the point of this test is what a
    module gets *without* editing anything else, so it must be a module that exists
    nowhere in this repository. Cleanup is explicit because the registry is global and
    the coverage guards in other files assert over its contents.
    """
    class Elsewhere(Evaluator):
        name = "elsewhere"
        views = [View("bar", "score by widget", x="widget")]

        async def evaluate(self, ctx):
            return []

    # Discover first, then add. Setting `_discovered` by hand instead would suppress
    # discovery and leave a registry holding only this one module - which is
    # `empty-registry-is-not-an-initialised-flag` in LESSONS.md, met from the other side.
    registry.discover()
    registry._REGISTRY["elsewhere"] = Elsewhere
    try:
        yield Elsewhere
    finally:
        registry._REGISTRY.pop("elsewhere", None)


def test_a_module_nobody_wired_up_still_gets_its_chart(store, a_third_party_module):
    """The criterion the design set: a module's results appear with a chart, with no file
    edited outside the module itself. Nothing in the store, the endpoint or the front end
    knows this evaluator exists."""
    store.add_samples("r1", [
        Sample(evaluator="elsewhere", case_id="a", score=1.0, dims={"widget": "sprocket"}),
        Sample(evaluator="elsewhere", case_id="b", score=0.0, dims={"widget": "flange"}),
    ])

    served = TestClient(app).get("/api/run/r1/views").json()
    mine = [v for v in served if v["evaluator"] == "elsewhere"]

    assert len(mine) == 1, f"the declared view was not served: {[v['evaluator'] for v in served]}"
    assert mine[0]["kind"] == "bar"
    assert mine[0]["title"] == "score by widget"
    assert mine[0]["data"]["x"] == ["flange", "sprocket"]
    assert mine[0]["data"]["v"] == [0.0, 1.0]


def test_a_module_with_no_samples_in_this_run_draws_no_empty_axes(store,
                                                                  a_third_party_module):
    """A run whose suite did not include a module should show no chart for it, rather
    than an empty pair of axes implying it ran and found nothing."""
    store.add_samples("r1", [_needle(2048, 50, 1.0)])

    served = TestClient(app).get("/api/run/r1/views").json()

    assert not [v for v in served if v["evaluator"] == "elsewhere"]
    assert [v for v in served if v["evaluator"] == "needle"], "the needle view vanished"


def test_the_shipped_modules_declare_views_that_actually_resolve(store):
    """Every declaration in this repository names dimensions its own samples carry.

    A view naming a dimension the module never sets would serve an empty chart forever,
    and nothing else would complain.
    """
    from llmbench.registry import available, get

    for name in available():
        for view in get(name)().views:
            data = store.view_data("r1", name, view.x, view.y, view.value)
            assert "x" in data, f"{name}: {view.title} did not resolve"