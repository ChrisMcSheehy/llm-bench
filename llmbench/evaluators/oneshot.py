"""One-shot build prompts — a model's single-shot design/coding ability.

Each prompt asks for a self-contained HTML artifact (a game, a call-centre
dashboard, an animated page, a dataviz, a small app). We capture the wall-clock
time-to-complete, tok/s, output size, and a cheap heuristic "did it actually
build" score — then store the artifact so you can render and human-rate it in
the dashboard gallery/arena. The heuristic is only a floor; the real signal is
your rating of the rendered result.

JSONL item schema:
    {"id","kind","category","features":["canvas","score",...],"prompt": "..."}

`features` are lowercase keywords used only for the heuristic completeness check.
"""
from __future__ import annotations

import re

from llmbench.evaluators.base import Breakdown, EvalContext, Evaluator, Verdict, View
from llmbench.models import Sample
from llmbench.registry import register
from llmbench.resources import load_jsonl

_HTML_BLOCK = re.compile(r"```(?:html)?\s*\n(.*?)```", re.DOTALL | re.I)
_DOC = re.compile(r"(<!doctype html.*?</html>|<html.*?</html>)", re.DOTALL | re.I)

_DEFAULTS = {
    "data_file": None,          # None = the bundled build prompts
    "limit": None,
    "max_tokens": 4096,        # artifacts are large
    "temperature": 0.4,
}


def extract_html(text: str) -> str:
    blocks = _HTML_BLOCK.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    m = _DOC.search(text)
    return m.group(1).strip() if m else text.strip()


def build_score(html: str, kind: str, features: list[str]) -> tuple[float, dict]:
    """Rough renderability/completeness heuristic in [0,1]. Not a quality score."""
    low = html.lower()
    checks = {
        "has_structure": "<html" in low and "</html>" in low,
        "not_truncated": low.rstrip().endswith("</html>"),
        "has_interactivity": "<script" in low or "onclick" in low,
        "features_present": (
            sum(1 for f in features if f.lower() in low) / len(features)
            if features else 1.0),
    }
    # kind-specific expectation
    need = {"game": "<canvas", "dataviz": "<canvas", "animation": "keyframes",
            "dashboard": "<canvas", "app": "<script"}.get(kind)
    checks["kind_marker"] = (need in low) if need else True
    weights = {"has_structure": 0.25, "not_truncated": 0.2, "has_interactivity": 0.2,
               "features_present": 0.2, "kind_marker": 0.15}
    score = sum(weights[k] * (v if isinstance(v, float) else float(bool(v)))
                for k, v in checks.items())
    return round(score, 3), checks


@register
class OneShotEvaluator(Evaluator):
    name = "oneshot"
    version = "1"
    default_config = _DEFAULTS
    # Three decimal places because the per-sample build score has three; this figure is
    # their mean and gains no precision by being printed wider.
    breakdowns = [Breakdown("build_score", ("kind",), round_to=3)]
    views = [View("bar", "build score by kind", x="kind")]

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        items = load_jsonl(cfg["data_file"], "datasets", "oneshot_prompts.jsonl",
                           limit=cfg["limit"])
        return [await self._one(ctx, cfg, it) for it in items]

    async def _one(self, ctx, cfg, it) -> Sample:
        kind = it.get("kind", "app")

        def grade(res) -> Verdict:
            html = extract_html(res.text)
            score, checks = build_score(html, kind, it.get("features", []))
            # score is the heuristic build score, not a quality judgement: the real
            # signal is a human rating the rendered artifact in the gallery.
            return Verdict(score=score,
                           meta={"prompt_id": it["id"], "prompt": it["prompt"],
                                 "kind": kind, "artifact": html,
                                 "build_checks": checks, "build_score": score,
                                 "html_bytes": len(html)})

        return await self.run_case(
            ctx, case_id=it["id"], group=kind,
            dims={"kind": kind, "prompt_id": it["id"]},
            messages=[{"role": "user", "content": it["prompt"]}], grade=grade,
            max_tokens=cfg["max_tokens"], temperature=cfg["temperature"])
