"""Coding evaluator: solve pre-solved problems, grade against held-out tests.

Each problem lives in problems/coding/<id>/ as:
    problem.yaml   id, prompt, entrypoint, timeout_s, (optional) n_samples
    tests.py       pytest file importing `entrypoint` from module `solution`
    solution.py    canonical reference (not shown to the model; sanity-checked)

For each problem we sample `n_samples` completions, extract the code block,
write it to solution.py in an isolated temp dir, and run tests.py under pytest
in a subprocess with a timeout. Scores use the unbiased pass@k estimator.

SECURITY: model-generated code is executed. It runs in a temp dir with a
scrubbed environment and a wall-clock timeout that kills the whole process tree.
That is ALL it is, on every operating system — there are no memory, CPU or
filesystem limits anywhere. This is NOT a sandbox. Only run it against
models/problems you trust, set execute:false to grade extraction only, or run
the whole bench inside a container if the output could be adversarial.
"""
from __future__ import annotations

import asyncio
import math
import os
import re
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from llmbench.evaluators.base import EvalContext, Evaluator
from llmbench.models import Metric, Sample
from llmbench.registry import register
from llmbench.resources import data_path

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

_DEFAULTS = {
    "problems_dir": None,     # None = the bundled problem set
    "problem_ids": None,      # None = all discovered
    "n_samples": 1,           # completions per problem (pass@k needs >1)
    "k": [1],                 # which pass@k to report
    "max_tokens": 1024,
    "temperature": 0.0,       # raise to ~0.8 when n_samples > 1
    "execute": True,
    "per_test_timeout_s": 30,
}


def _extract_code(text: str) -> str:
    blocks = _CODE_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()  # some models skip fences; try raw


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al. 2021): n samples, c correct."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def _child_env() -> dict[str, str]:
    """The environment the model's code runs in — deliberately almost empty.

    SYSTEMROOT is the one exception, and it is not optional: without it Windows
    cannot initialise its socket layer, so the child died on `import asyncio`
    during pytest startup, before a single test ran. That scored every coding
    problem zero — including the project's own known-correct reference solutions —
    while still looking like an ordinary failed run.
    """
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    return env


def _spawn_kwargs() -> dict[str, Any]:
    """Extra process-creation arguments that make the test run killable as a unit.

    On POSIX a new session makes the child its own process group leader, so one
    signal reaches every descendant it goes on to spawn. Windows has no equivalent
    to set here; see _kill_process_tree for how it is handled instead.
    """
    return {} if os.name == "nt" else {"start_new_session": True}


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill a timed-out test run *and everything it started*.

    This is the code path that executes model-generated code. Killing only the
    direct child leaves whatever that child spawned running with no supervision
    and no timeout — so a timeout that stops only the child is not a timeout.
    """
    if os.name == "nt":
        # ironclad: taskkill /F /T — rejected Win32 job objects: assigning a job
        # without a spawn race needs CREATE_SUSPENDED, which asyncio's subprocess
        # API cannot pass. /T walks the parent-child tree, which covers every
        # descendant alive at this moment.
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/F", "/T", "/PID", str(proc.pid),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
        return
    try:
        # _spawn_kwargs started a new session, so the child is its own group leader
        # and its pid doubles as the group id.
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # it finished between the timeout firing and this call


@register
class CodingEvaluator(Evaluator):
    name = "coding"
    version = "1"
    default_config = _DEFAULTS

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        problems = self._load_problems(cfg)
        samples: list[Sample] = []
        for prob in problems:
            for i in range(cfg["n_samples"]):
                samples.append(await self._attempt(ctx, cfg, prob, i))
        return samples

    def _load_problems(self, cfg: dict[str, Any]) -> list[dict]:
        root = (Path(cfg["problems_dir"]) if cfg["problems_dir"]
                else data_path("problems", "coding"))
        # Returning an empty list here (the old behaviour) meant a missing problems
        # directory produced a run reporting zero coding problems as though that were
        # a result.
        if not root.exists():
            raise FileNotFoundError(f"Coding problems directory not found: {root}")
        out = []
        for d in sorted(root.iterdir()):
            meta_f = d / "problem.yaml"
            if not (d.is_dir() and meta_f.exists()):
                continue
            meta = yaml.safe_load(meta_f.read_text(encoding="utf-8"))
            if cfg["problem_ids"] and meta.get("id") not in cfg["problem_ids"]:
                continue
            meta["_dir"] = d
            out.append(meta)
        return out

    async def _attempt(self, ctx, cfg, prob, idx) -> Sample:
        pid = prob.get("id", prob["_dir"].name)
        prompt = (f"{prob['prompt']}\n\nWrite a complete Python solution. "
                  f"Define exactly the function `{prob['entrypoint']}`. "
                  f"Return only a single ```python code block.")
        dims = {"problem": pid, "sample": idx}
        try:
            res = await ctx.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
            )
        except Exception as e:
            return Sample(evaluator=self.name, case_id=f"{pid}:{idx}", group=pid,
                          dims=dims, error=repr(e))

        if res.unusable_reason:
            return Sample(evaluator=self.name, case_id=f"{pid}:{idx}", group=pid,
                          dims=dims, skipped=res.unusable_reason)

        code = _extract_code(res.text)
        passed, detail = (None, "not executed")
        if cfg["execute"]:
            passed, detail = await self._run_tests(
                code, prob, cfg.get("per_test_timeout_s", prob.get("timeout_s", 30)))

        return Sample(
            evaluator=self.name, case_id=f"{pid}:{idx}", group=pid, dims=dims,
            score=1.0 if passed else 0.0 if passed is not None else None,
            passed=passed,
            input_tokens=res.input_tokens, output_tokens=res.output_tokens,
            latency_ms=res.latency_ms, tok_per_sec=res.tok_per_sec,
            server_prompt_tps=res.server_prompt_tps, server_gen_tps=res.server_gen_tps,
            meta={"detail": detail[:500], "code": code[:2000]},
        )

    async def _run_tests(self, code: str, prob: dict, timeout: int) -> tuple[bool, str]:
        tmp = Path(tempfile.mkdtemp(prefix="llmbench_"))
        try:
            # encoding is explicit because this is *model* output: any character can
            # appear in it, and the platform default locale cannot necessarily encode it.
            (tmp / "solution.py").write_text(code, encoding="utf-8")
            shutil.copy(prob["_dir"] / "tests.py", tmp / "tests.py")
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pytest", "-q", "tests.py",
                cwd=tmp, env=_child_env(),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                **_spawn_kwargs(),
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                await _kill_process_tree(proc)
                await proc.wait()  # reap, so no zombie is left behind
                return False, "timeout"
            return proc.returncode == 0, out.decode(errors="replace")[-500:]
        except Exception as e:
            return False, f"harness error: {e!r}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def aggregate(self, samples: list[Sample]) -> list[Metric]:
        metrics = super().aggregate(samples)
        graded = [s for s in samples if s.error is None and s.passed is not None]
        if not graded:
            return metrics
        by_prob: dict[str, list[bool]] = {}
        for s in graded:
            by_prob.setdefault(s.dims["problem"], []).append(bool(s.passed))
        # pass@k over all k that make sense for the sample count.
        n = min(len(v) for v in by_prob.values())
        for k in sorted({kk for kk in self.default_config["k"] if kk <= n} | {1}):
            vals = [pass_at_k(len(v), sum(v), k) for v in by_prob.values()]
            # Every graded attempt, not every problem: pass@k is estimated from the
            # attempts, and the number of problems alone hides how many times each ran.
            metrics.append(Metric(evaluator=self.name, name=f"pass@{k}",
                                  value=round(sum(vals) / len(vals), 4),
                                  n=len(graded)))
        for pid, v in sorted(by_prob.items()):
            metrics.append(Metric(evaluator=self.name, name="problem_pass_rate",
                                  value=round(sum(v) / len(v), 4), n=len(v),
                                  dims={"problem": pid}))
        return metrics
