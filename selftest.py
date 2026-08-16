"""End-to-end smoke test with a mock in-process target (no live server needed).

Exercises detection + every HTTP evaluator + store + dashboard queries. The mock
acts as an oracle where it can (retrieval, structured, tool-calls) to prove the
graders accept correct answers, and neutrally otherwise. Run: python selftest.py
"""
from __future__ import annotations

import asyncio
import functools
import json
import re
import tempfile

import yaml

from llmbench.models import ModelFingerprint
from llmbench.targets.base import GenResult, Target, approx_tokens
import llmbench.targets as targets_pkg
from llmbench.resources import data_path
from llmbench.store import Store
from llmbench.orchestrator import Orchestrator

_NEEDLE = re.compile(r"special access code for the city of (\w+) is ([A-Z]+-\d+)")
_MK_Q = re.compile(r"what is the code for (\w+)\?", re.I)
_ENTRY = re.compile(r"function `(\w+)`")
# Reached through resources.py rather than a relative path, so this works from any
# directory and does not go stale when the bundled data moves.
_PROBLEMS = data_path("problems", "coding")


@functools.lru_cache(maxsize=1)
def _reference_solutions() -> list[tuple[str, str]]:
    """(prompt, solution) for every bundled problem, read once.

    Keyed by the **prompt**, not by the entry-point name. The entry point is not unique:
    HumanEval alone has six names shared by two problems each (`add`, `solve`,
    `sort_array` and three more), and the older substring scan over problem.yaml matched
    the wrong file first for twenty of the 168 problems - handing back a correct solution
    to a different question, which then failed its tests and read as a harness fault.
    The prompt appears verbatim in the message the evaluator builds and is unique.
    """
    out = []
    for directory in sorted(_PROBLEMS.iterdir()):
        meta = directory / "problem.yaml"
        if not (directory.is_dir() and meta.exists()):
            continue
        parsed = yaml.safe_load(meta.read_text(encoding="utf-8"))
        out.append((parsed["prompt"].strip(),
                    (directory / "solution.py").read_text(encoding="utf-8")))
    # Longest first, so a prompt that contains another as a substring cannot shadow it.
    return sorted(out, key=lambda pair: -len(pair[0]))


class MockTarget(Target):
    engine = "mock"

    # A context this server refuses to exceed, or None for a server with no limit.
    # Simulates the machine that cannot hold the top of the ladder, which is otherwise
    # untestable without owning constrained hardware. None by default, so the
    # self-test's own runs are unaffected.
    max_context = None

    async def detect(self) -> ModelFingerprint:
        quant = {"A": "Q4_K_M", "B": "Q6_K"}.get(self.model, "Q4_K_M")
        return ModelFingerprint(
            engine="llama.cpp", build_number=6543, build_commit="abcdef1234",
            base_url=self.base_url, model_id=f"Qwen3-8B-{quant}", model_name="Qwen3-8B",
            quant=quant, n_params="8B", n_ctx=16384, kv_cache_k="q8_0",
            kv_cache_v="q8_0", flash_attn="on", spec_type="draft-mtp", mtp=True,
            sampling={"temperature": 0.0}, chat_template_sha="deadbeef0001",
            launch_args=["-c", "16384", "-ctk", "q8_0", "--spec-type", "draft-mtp"])

    async def count_tokens(self, text: str) -> int:
        return approx_tokens(text)

    def _reply(self, user: str) -> str:
        # one-shot artifact builds
        if "self-contained HTML file" in user:
            return ("<!doctype html><html><head><style>@keyframes k{from{}to{}}</style></head>"
                    "<body><canvas id='c' width='300' height='300'></canvas>"
                    "<div>score queue kpi agents grid chart labels arrow game over progress "
                    "operator display button timer</div>"
                    "<script>requestAnimationFrame(()=>{});function start(){}"
                    "document.onclick=()=>{};</script></body></html>")
        # needle
        m = _NEEDLE.search(user)
        if m and "special access code" in user:
            return "50%" not in user and len(user) < 120000 and m.group(2) or "UNKNOWN"
        # long_context multikey: find "the code for <w> is <CODE>" for queried word
        q = _MK_Q.search(user)
        if q:
            w = q.group(1)
            mm = re.search(rf"code for {w} is (\S+?)\.", user)
            return mm.group(1) if mm else "NONE"
        # long_context vartrack
        if "numeric value of" in user:
            vm = re.search(r"v0 = (\d+)", user)
            return vm.group(1) if vm else "0"
        # structured json
        if "JSON array" in user:
            return '```json\n[{"sku":"a","qty":1},{"sku":"b","qty":2},{"sku":"c","qty":3}]\n```'
        if "JSON object" in user and "active" in user:
            return '{"name":"Ada","age":30,"active":true}'
        # tool calls
        if "get_weather" in user:
            return '{"name":"get_weather","arguments":{"city":"Leeds","units":"celsius"}}'
        if "create_event" in user:
            return '{"name":"create_event","arguments":{"title":"Team sync","date":"2026-01-15"}}'
        # coding: answer with this problem's own reference solution, so anything below a
        # perfect score means the harness failed to run correct code.
        if _ENTRY.search(user):
            for prompt, solution in _reference_solutions():
                if prompt and prompt in user:
                    return f"```python\n{solution}\n```"
        # ifeval: satisfy a couple of easy constraints generically
        if "three bullet points" in user:
            return "- one\n- two\n- three"
        if "JSON object with keys" in user:
            return '{"name":"Sam","age":22}'
        # mcqa / math / text2sql / fallback
        if "single letter" in user:
            return "A"
        if "\\boxed" in user:
            return "The answer is \\boxed{0}."
        if "SQL SELECT" in user:
            return "SELECT 1"
        return "ok"

    async def generate(self, messages, *, max_tokens=512, temperature=0.0, extra=None):
        prompt = messages[-1]["content"]
        asked = approx_tokens(prompt)
        if self.max_context is not None and asked > self.max_context:
            # Shaped after what a real server does when the prompt will not fit. The
            # wording is a stand-in: llmbench treats any exception here as a refusal,
            # so nothing depends on matching a particular backend's message.
            raise RuntimeError(
                f"the request exceeds the available context size: "
                f"{asked} > {self.max_context}")
        text = self._reply(prompt)
        return GenResult(text=text, input_tokens=asked,
                         output_tokens=max(1, approx_tokens(text)), latency_ms=100.0,
                         tok_per_sec=45.0, server_gen_tps=47.0, server_prompt_tps=850.0)


def main():
    targets_pkg._ENGINES["mock"] = MockTarget
    suite = """
name: selftest
targets:
  - engine: mock
    base_url: http://mock-a
    model: A
  - engine: mock
    base_url: http://mock-b
    model: B
evaluators:
  mcqa:
  math_qa:
  ifeval:
  structured:
  text2sql:
  needle:
    context_lengths: [2048, 8192]
    depths: [0, 50, 100]
  long_context:
    context_lengths: [2048, 8192]
    queries_per_rung: 2
  coding:
    execute: true
    # A handful, not all 168. This checks that the harness runs correct code correctly,
    # which four problems establish as well as 168 do - and 168 pytest subprocesses would
    # add three minutes to every run of the suite on every platform in the matrix.
    # `test_humaneval_problems.py` is what covers the converted set.
    problem_ids: [two_sum, kadane, humaneval_000, humaneval_032]
  human:
    limit: 3
  oneshot:
    limit: 2
  perplexity:
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False,
                                     encoding="utf-8") as f:
        f.write(suite); suite_path = f.name
    dbf = tempfile.mktemp(suffix=".db")
    store = Store(dbf)
    orch = Orchestrator(store, log=print)
    results = asyncio.run(orch.run_suite(suite_path))
    run = results[0]; rid = run.run_id

    print("\n=== metrics (headline) ===")
    for m in run.metrics:
        if not m.dims:
            print(f"  {m.evaluator}.{m.name} = {m.value}")

    print("\n=== capabilities ===")
    caps = store.capabilities(rid)
    for c in caps:
        print(f"  {c['evaluator']:14s} {c['score']:.2f}")

    evs = {m.evaluator for m in run.metrics}
    for want in ["mcqa","math_qa","ifeval","structured","text2sql","needle",
                 "long_context","coding","human","oneshot","perplexity"]:
        assert want in evs, f"missing metrics for {want}"
    # Every figure states how many graded items it rests on (design D7b). Asserted over
    # a whole real run rather than over a list of files, because the aggregator most
    # likely to be missing a count is the one nobody remembered to edit.
    naked = sorted({f"{m.evaluator}.{m.name}" for m in run.metrics if m.n is None})
    assert not naked, f"figures with no count behind them: {naked}"
    capmap = {c["evaluator"]: c["score"] for c in caps}
    assert capmap.get("structured") == 1.0, f"structured={capmap.get('structured')}"
    assert capmap.get("long_context") == 1.0, f"long_context={capmap.get('long_context')}"
    # The mock backend answers coding problems with their own reference solution, so
    # anything below 1.0 means the harness failed to run correct code. Without this,
    # a harness that executed nothing at all still read as an ordinary weak result.
    assert capmap.get("coding") == 1.0, f"coding={capmap.get('coding')}"
    perr = [s for s in run.samples if s.evaluator=="perplexity"]
    assert perr and perr[0].skipped, "perplexity should skip gracefully"
    assert perr[0].error is None, "an unconfigured module is not a failed one"
    assert len(store.leaderboard()) == 2, "expected 2 configs"
    hresp = store.human_responses()
    assert len({r["fp_hash"] for r in hresp}) == 2, "human responses from 2 configs"

    # Every run records the machine it ran on (design D1). Without it, two machines'
    # speed figures are indistinguishable once stored, which is the failure Phase 4
    # exists to prevent.
    rows = store.conn.execute("SELECT run_id, host_hash FROM run").fetchall()
    assert rows, "no runs recorded"
    for run_id, host_hash in rows:
        assert host_hash, f"run {run_id} recorded no host"
        assert store.conn.execute(
            "SELECT COUNT(*) FROM host WHERE hash=?", (host_hash,)).fetchone()[0] == 1, \
            f"run {run_id} points at a host row that does not exist"

    # The same property, read back out of the database rather than off the objects: a
    # count that never reached a column cannot be displayed beside anything.
    unexplained = store.conn.execute(
        "SELECT evaluator, name FROM metric WHERE n IS NULL").fetchall()
    assert not unexplained, \
        f"stored figures with no count: {[tuple(r) for r in unexplained]}"
    store.close()

    # ---- arena (human eval) via the real FastAPI app --------------------
    import os
    os.environ["LLMBENCH_DB"] = dbf
    from fastapi.testclient import TestClient
    from llmbench.dashboard.app import app
    c = TestClient(app)

    mu = c.get("/api/arena/matchup").json()
    assert mu["available"] and mu["a"]["token"] != mu["b"]["token"], "need a blind pair"
    for winner in ("a", "b", "tie", "a"):
        m = c.get("/api/arena/matchup").json()
        r = c.post("/api/arena/vote", json={"prompt_id": m["prompt_id"],
                   "a_token": m["a"]["token"], "b_token": m["b"]["token"],
                   "winner": winner})
        assert r.json()["ok"]
    single = c.get("/api/arena/single").json()
    assert single["available"]
    c.post("/api/arena/rate", json={"prompt_id": single["prompt_id"],
           "token": single["token"], "score": 4})
    lb = c.get("/api/arena/leaderboard").json()
    assert lb["counts"]["pairwise"] == 4 and lb["counts"]["rating"] == 1
    assert lb["board"] and lb["board"][0]["elo"] != 1000.0, "Elo should have moved"

    gal = c.get("/api/gallery").json()
    assert gal, "gallery should list one-shot prompts"
    g0 = gal[0]
    assert len(g0["entries"]) == 2, "artifact from each config"
    e = g0["entries"][0]
    assert "<html" in (e["html"] or "").lower() and e["token"], "renderable artifact + token"
    assert e["latency_ms"] is not None and e["build_score"] is not None
    assert c.post("/api/arena/rate", json={"prompt_id": g0["prompt_id"],
                  "token": e["token"], "score": 5}).json()["ok"]
    print(f"\n=== gallery ===\n  {len(gal)} one-shot prompts · "
          f"{sum(len(x['entries']) for x in gal)} artifacts rendered")

    print("\n=== arena leaderboard ===")
    for r in lb["board"]:
        print(f"  {r['label'][:28]:28s} elo={r['elo']}  games={r['games']}  "
              f"win%={r['win_rate']}  stars={r['avg_stars']}")

    print("\nALL CHECKS PASSED \u2705")


if __name__ == "__main__":
    main()
