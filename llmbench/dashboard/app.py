"""Dashboard API + static host.

JSON endpoints the frontend consumes:
  GET /api/leaderboard
  GET /api/runs
  GET /api/configurations
  GET /api/run/{run_id}/metrics
  GET /api/run/{run_id}/views      every chart every module declared (design E3)
  GET /api/run/{run_id}/skipped
  GET /api/trend?evaluator=needle&metric=score_mean
  GET /api/suites                  suite names the browser may ask for
  POST /api/run                    {suite, server} - names only, never bodies (design B6)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llmbench import launcher
from llmbench.registry import available, get
from llmbench.resources import available_suites, resolve_suite
from llmbench.store import Store, default_db_path

app = FastAPI(title="llmbench")
_STATIC = Path(__file__).parent / "static"


def store() -> Store:
    # Resolved per call, not once at import: callers (notably selftest.py) set
    # LLMBENCH_DB after this module has been imported, and an import-time lookup
    # would ignore them.
    return Store(str(default_db_path()))


@app.get("/api/leaderboard")
def leaderboard():
    s = store(); out = s.leaderboard(); s.close(); return out


@app.get("/api/hosts")
def hosts():
    """Every machine results have been recorded on."""
    s = store(); out = s.hosts(); s.close(); return out


@app.get("/api/pooled")
def pooled():
    """Figures grouped the two different ways they must be grouped.

    Quality pools across machines because the same model answers the same questions
    equally well anywhere. Speed never does — see design D1.
    """
    s = store()
    out = {"quality": s.pooled_quality(), "speed": s.pooled_speed()}
    s.close()
    return out


@app.get("/api/configurations")
def configurations():
    """Every configuration, what it rests on, and the figures pooled over it.

    Quality pools across machines and speed never does (design D1), and both now carry
    the number of graded items behind them (design D7b).
    """
    s = store()
    out = {"effort": s.configuration_effort(),
           "quality": s.pooled_quality(),
           "speed": s.pooled_speed()}
    s.close()
    return out


@app.get("/api/runs")
def runs():
    s = store(); out = s.runs(); s.close(); return out


@app.get("/api/run/{run_id}/metrics")
def metrics(run_id: str):
    s = store(); out = s.metrics_for(run_id); s.close(); return out


@app.get("/api/run/{run_id}/views")
def views(run_id: str):
    """Every chart every test module declared, with its data (design E3).

    One endpoint for all of them, replacing the three that each had an evaluator's name
    written into them. A module gets a chart by declaring `views` in its own file, and
    nothing here or in the front end needs to know it exists.

    Views with no data are dropped rather than returned empty: a run whose suite did not
    include a module should show no chart for it, not an empty pair of axes.
    """
    s = store()
    try:
        drawn = []
        for name in available():
            for view in get(name)().views:
                data = s.view_data(run_id, name, view.x, view.y, view.value)
                if not data.get("x"):
                    continue
                drawn.append({"evaluator": name, "kind": view.kind, "title": view.title,
                              "x_label": view.x, "y_label": view.y, "data": data})
        return drawn
    finally:
        s.close()


@app.get("/api/run/{run_id}/skipped")
def skipped(run_id: str):
    """What this run did not attempt, and why."""
    s = store(); out = s.skipped(run_id); s.close(); return out


@app.get("/api/run/{run_id}/capabilities")
def capabilities(run_id: str):
    s = store()
    out = {"scores": s.capabilities(run_id), "perplexity": s.perplexity(run_id)}
    s.close()
    return out


@app.get("/api/trend")
def trend(evaluator: str = "needle", metric: str = "score_mean"):
    s = store(); out = s.trend(evaluator, metric); s.close(); return out


import base64
import random
from collections import defaultdict

from pydantic import BaseModel


def _tok(fp: str) -> str:
    return base64.urlsafe_b64encode(fp.encode()).decode()


def _untok(t: str) -> str:
    return base64.urlsafe_b64decode(t.encode()).decode()


def _badge(r: dict) -> dict:
    return {"latency_ms": r.get("latency_ms"), "tok_per_sec": r.get("tok_per_sec"),
            "output_tokens": r.get("output_tokens")}


@app.get("/api/arena/matchup")
def arena_matchup():
    s = store()
    rows = s.gradable_responses(); s.close()
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_prompt[r["prompt_id"]].append(r)
    eligible = {p: rs for p, rs in by_prompt.items()
                if len({r["fp_hash"] for r in rs}) >= 2}
    if not eligible:
        return {"available": False,
                "reason": "Need responses from at least 2 configs. Run a suite "
                          "with the `human` and/or `oneshot` evaluator on 2+ targets."}
    pid = random.choice(list(eligible))
    pool, seen = [], set()
    for r in random.sample(eligible[pid], len(eligible[pid])):
        if r["fp_hash"] not in seen:
            pool.append(r); seen.add(r["fp_hash"])
        if len(pool) == 2:
            break
    random.shuffle(pool)
    a, b = pool
    return {"available": True, "prompt_id": pid, "category": a.get("category") or a.get("kind"),
            "render": a["render"], "prompt": a["prompt"],
            "a": {"token": _tok(a["fp_hash"]), "content": a["content"], **_badge(a)},
            "b": {"token": _tok(b["fp_hash"]), "content": b["content"], **_badge(b)}}


@app.get("/api/arena/single")
def arena_single():
    s = store()
    rows = s.gradable_responses(); s.close()
    if not rows:
        return {"available": False, "reason": "No human-eval responses yet."}
    r = random.choice(rows)
    return {"available": True, "prompt_id": r["prompt_id"],
            "category": r.get("category") or r.get("kind"), "render": r["render"],
            "prompt": r["prompt"], "token": _tok(r["fp_hash"]),
            "content": r["content"], **_badge(r)}


@app.get("/api/gallery")
def gallery():
    s = store(); out = s.gallery(); s.close()
    for g in out:
        for e in g["entries"]:
            e["token"] = _tok(e["fp"])
    return out


class PairVote(BaseModel):
    prompt_id: str
    a_token: str
    b_token: str
    winner: str            # a | b | tie | both_bad


class RateVote(BaseModel):
    prompt_id: str
    token: str
    score: int             # 1..5


@app.post("/api/arena/vote")
def arena_vote(v: PairVote):
    s = store()
    s.add_vote(v.prompt_id, "pairwise", fp_a=_untok(v.a_token),
               fp_b=_untok(v.b_token), winner=v.winner)
    s.close()
    return {"ok": True}


@app.post("/api/arena/rate")
def arena_rate(v: RateVote):
    s = store()
    s.add_vote(v.prompt_id, "rating", fp_a=_untok(v.token),
               score=max(1, min(5, v.score)))
    s.close()
    return {"ok": True}


@app.get("/api/arena/leaderboard")
def arena_leaderboard():
    s = store()
    out = {"board": s.arena_leaderboard(), "counts": s.vote_count()}
    s.close()
    return out


# ---- launching model servers ---------------------------------------------------
#
# The browser may name a profile. It may never supply a binary path or an argument
# list: that would let a web page run an arbitrary program on this machine. The
# profiles file on disk is the allowlist, and these endpoints only ever look a name up
# in it. See docs/ironclad/DESIGN-launcher.md, decision L1.

# ---- starting a run from the browser (design B6) ------------------------------------
#
# The browser may post **a suite name and a profile name**, and nothing else. Not a
# prompt, not a path, not an argument, not a suite body. This is decision L1 restated for
# a second verb and for the same reason: L1 stops the browser choosing which binary runs,
# and a run trigger that accepted a suite body would hand all of that back, because a
# suite names targets and a target is an address and an argument list.
#
# One run at a time, which is also why the sweep never starts two servers at once: two
# servers sharing a graphics card contend for it and corrupt the speed figures.

class RunRequest(BaseModel):
    suite: str
    server: Optional[str] = None
    # Anything else is rejected rather than ignored. Silently dropping an unexpected
    # field is how a request that meant to smuggle something gets a success back.
    model_config = {"extra": "forbid"}


_RUN_STATE: dict[str, Any] = {"status": "idle", "suite": None, "server": None,
                              "started_at": None, "finished_at": None, "error": None,
                              "runs": []}


@app.get("/api/suites")
def suites():
    """The suites that exist on disk, by name. The browser picks from these."""
    return sorted(available_suites())


@app.get("/api/run/active")
def active_run():
    return _RUN_STATE


async def _execute(suite_path: Path, profile_name: Optional[str]) -> None:
    """Run a suite, optionally against a launched profile, recording state as it goes.

    The target spec is built exactly as `cli._sweep` builds it, including `binary`: only
    the binary can report which graphics cards a run used, and we have it precisely
    because we started the server ourselves.
    """
    from llmbench.orchestrator import Orchestrator

    s = Store(str(default_db_path()))
    try:
        orchestrator = Orchestrator(s)
        if profile_name:
            profile = launcher.load_profiles()[profile_name]
            # `launched` stops the server however this block exits. A server left running
            # would hold the graphics card against every run after it.
            with launcher.launched(profile) as server:
                results = await orchestrator.run_suite(
                    str(suite_path),
                    target_specs=[{"engine": "llama.cpp", "base_url": server.base_url,
                                   "known_args": server.launch_args,
                                   "binary": server.profile.binary}])
        else:
            results = await orchestrator.run_suite(str(suite_path))
        _RUN_STATE.update(status="done", finished_at=_now(),
                          runs=[r.run_id for r in results])
    except Exception as exc:
        # Reported rather than raised: nothing is awaiting this task, so an exception
        # would vanish into the event loop and the dashboard would show "running" forever.
        _RUN_STATE.update(status="error", finished_at=_now(), error=repr(exc))
    finally:
        s.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.post("/api/run")
async def start_run(req: RunRequest):
    if _RUN_STATE["status"] == "running":
        raise HTTPException(409, "a run is already in progress")
    try:
        suite_path = resolve_suite(req.suite)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if req.server is not None and req.server not in launcher.load_profiles():
        raise HTTPException(404, f"no launch profile named {req.server!r}")

    _RUN_STATE.update(status="running", suite=req.suite, server=req.server,
                      started_at=_now(), finished_at=None, error=None, runs=[])
    asyncio.create_task(_execute(suite_path, req.server))
    return _RUN_STATE


@app.get("/api/servers")
def servers():
    profiles = launcher.load_profiles()
    live = launcher.running()
    out = []
    for name, profile in sorted(profiles.items()):
        record = live.get(name)
        up = bool(record) and launcher.is_listening(int(record["port"]))
        out.append({
            "name": name,
            "binary": profile.binary,
            "model": profile.model,
            "args": profile.args,
            "running": up,
            "base_url": record["base_url"] if up else None,
        })
    return out


def _profile_or_404(name: str):
    profiles = launcher.load_profiles()
    if name not in profiles:
        # 404 rather than 400: to this API a name that is not in the file does not exist.
        raise HTTPException(status_code=404, detail=f"no profile named {name!r}")
    return profiles[name]


@app.post("/api/servers/{name}/start")
def start_server(name: str):
    profile = _profile_or_404(name)
    record = launcher.running().get(name)
    if record and launcher.is_listening(int(record["port"])):
        return {"name": name, "running": True, "base_url": record["base_url"]}
    try:
        server = launcher.start(profile)
    except launcher.LaunchError as exc:
        # The server's own output is the useful part; passing it through is the whole
        # reason the launcher captures it.
        raise HTTPException(status_code=500, detail=str(exc))
    launcher.record_running(server)
    return {"name": name, "running": True, "base_url": server.base_url}


@app.post("/api/servers/{name}/stop")
def stop_server(name: str):
    _profile_or_404(name)
    stopped = launcher.stop_by_name(name)
    return {"name": name, "running": False, "stopped": stopped}


@app.get("/servers")
def servers_page():
    return FileResponse(_STATIC / "servers.html")


@app.get("/configs")
def configs_page():
    return FileResponse(_STATIC / "configs.html")


@app.get("/arena")
def arena():
    return FileResponse(_STATIC / "arena.html")


@app.get("/gallery")
def gallery_page():
    return FileResponse(_STATIC / "gallery.html")


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC), name="static")
