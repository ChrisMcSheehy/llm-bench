"""Orchestrator: the conductor.

For each target in a suite it detects the fingerprint, then runs each configured
evaluator, aggregates metrics, and streams everything to the store. Evaluators
run sequentially per target (they each saturate the GPU; parallelism there just
adds noise to throughput numbers), but targets can be processed one after
another in a single invocation.
"""
from __future__ import annotations

import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from llmbench import hostinfo
from llmbench.config import load_suite
from llmbench.evaluators.base import EvalContext
from llmbench.models import HostFingerprint, ModelFingerprint, RunResult, Sample
from llmbench.registry import get as get_evaluator
from llmbench.store import Store
from llmbench.targets import build_target


def _now() -> datetime:
    return datetime.now(timezone.utc)


def current_host(binary: Optional[str] = None) -> HostFingerprint:
    """This machine, as far as it can be known.

    Composed here rather than in `hostinfo` or `models`, because both of those are
    leaf modules that import nothing else in this project and this needs both.

    `binary` is a llama.cpp executable to read the graphics cards from. There is one
    whenever llmbench started the server itself; when the user pointed it at an
    address instead, the cards are simply unknown — the same asymmetry as the launch
    settings (design D6a, D8b).
    """
    facts = hostinfo.machine_facts()
    declared = hostinfo.load_declared()
    # The driver version is recorded and never hashed (D1): it moves too often to be
    # part of an identity, but without it an unexplained discrepancy cannot be traced.
    found = hostinfo.attach_drivers(hostinfo.devices(binary),
                                    hostinfo.driver_versions())
    return HostFingerprint(
        os=facts["os"], os_release=facts["os_release"], arch=facts["arch"],
        cpu_count=facts["cpu_count"], total_memory_bytes=facts["total_memory_bytes"],
        cpu_model=declared.get("cpu_model"),
        devices=found, declared=declared)


class Orchestrator:
    def __init__(self, store: Store, log: Optional[Callable[[str], None]] = None):
        self.store = store
        self.log = log or (lambda m: None)
        # Targets that never produced a run, as (base_url, reason). Reset by each
        # run_suite call. A caller that wants to report failures needs somewhere to
        # read them, and the return value only carries the targets that worked.
        self.failed_targets: list[tuple[str, str]] = []

    async def run_suite(self, suite_path: str,
                        target_specs: Optional[list[dict]] = None) -> list[RunResult]:
        """Run a suite against the targets it defines, or against ones supplied here.

        Supplied targets are how a launched server is benched: the caller starts it,
        passes its address and the arguments it was started with, and stops it
        afterwards. Starting processes is deliberately not this class's job — see
        docs/ironclad/DESIGN-launcher.md.
        """
        suite = load_suite(suite_path, require_targets=target_specs is None)
        specs = suite["targets"] if target_specs is None else target_specs
        results = []
        self.failed_targets = []
        for spec in specs:
            try:
                results.append(await self._run_target(suite, spec))
            except Exception as exc:
                # A target that cannot even be detected must not cost the targets after
                # it (design D3d). Comparing several servers is the whole point of a
                # sweep, and one of them being down is the ordinary case.
                url = spec.get("base_url", "?")
                self.failed_targets.append((url, repr(exc)))
                self.log(f"target {url} produced no run: {exc!r}")
        return results

    async def detect_only(self, spec: dict[str, Any]) -> ModelFingerprint:
        target = build_target(spec)
        try:
            return await target.detect()
        finally:
            await target.aclose()

    async def _run_target(self, suite: dict, spec: dict) -> RunResult:
        target = build_target(spec)
        run_id = uuid.uuid4().hex[:12]
        try:
            fp = await target.detect()
            self.log(f"[{run_id}] detected: {fp.label}  ({fp.fingerprint_hash})")
            run = RunResult(run_id=run_id, fingerprint=fp, suite=suite["name"])
            # Once per run, never per sample: reading the devices starts a process, and
            # doing that between test items would move the timings the run measures.
            host = current_host(spec.get("binary"))
            self.store.start_run(run, host_hash=self.store.upsert_host(host))

            failures: list[str] = []
            for ev_name, ev_cfg in suite["evaluators"].items():
                if ev_cfg is None:
                    ev_cfg = {}
                if ev_cfg.get("enabled") is False:
                    continue
                failure = await self._run_evaluator(
                    run_id, target, fp, ev_name, ev_cfg, run)
                if failure:
                    failures.append(failure)

            # `partial` rather than `ok`, because part of the suite did not run, and
            # rather than `error`, because the rest of it did (design D3d).
            run.status = "partial" if failures else "ok"
            run.error = "; ".join(failures) or None
            run.finished_at = _now()
            self.store.finish_run(run_id, run.status, run.finished_at.isoformat(),
                                  run.error)
            self.log(f"[{run_id}] {run.status}")
            return run
        except Exception as e:
            self.log(f"[{run_id}] FAILED: {e}\n{traceback.format_exc()}")
            self.store.finish_run(run_id, "error", _now().isoformat(), str(e))
            raise
        finally:
            await target.aclose()

    async def _run_evaluator(self, run_id, target, fp, ev_name, ev_cfg, run):
        """Run one test module. Returns a description of its failure, or None.

        A module that raises is recorded as one errored sample and the modules after it
        still run: a broken test module is not a reason to throw away the rest of a
        suite that may have taken an hour (design D3d).
        """
        cls = get_evaluator(ev_name)
        evaluator = cls()
        ctx = EvalContext(target=target, fingerprint=fp,
                          config={k: v for k, v in ev_cfg.items() if k != "enabled"})
        self.log(f"[{run_id}] running {ev_name}...")
        failure = None
        try:
            samples = await evaluator.evaluate(ctx)
        except Exception as exc:
            failure = f"{ev_name}: {exc!r}"
            self.log(f"[{run_id}] {ev_name} FAILED: {exc!r}\n{traceback.format_exc()}")
            samples = [Sample(evaluator=ev_name, case_id="evaluator",
                              error=f"{ev_name} raised: {exc!r}")]

        # Aggregation is inside the boundary too (design D3d). A module can grade every
        # sample and still fail while summarising them - ifeval did exactly that on
        # 2026-08-05, raising on a skipped sample's score of None - and with this call
        # outside the guard, one module's summary cost the entire target its run.
        try:
            metrics = evaluator.aggregate(samples)
        except Exception as exc:
            failure = f"{ev_name} aggregate: {exc!r}"
            self.log(f"[{run_id}] {ev_name} FAILED to aggregate: {exc!r}\n"
                     f"{traceback.format_exc()}")
            # The samples are kept: they were graded successfully and are the evidence.
            # Only the summary is missing, and the errored sample below says why.
            metrics = []
            samples = samples + [Sample(evaluator=ev_name, case_id="aggregate",
                                        error=f"{ev_name} aggregate raised: {exc!r}")]
        self.store.add_samples(run_id, samples)
        self.store.add_metrics(run_id, metrics)
        run.samples.extend(samples)
        run.metrics.extend(metrics)
        ok = sum(1 for s in samples if s.error is None and s.skipped is None)
        skipped = sum(1 for s in samples if s.skipped is not None)
        self.log(f"[{run_id}] {ev_name}: {ok}/{len(samples)} graded, "
                 f"{skipped} skipped, {len(metrics)} metrics")
        return failure
