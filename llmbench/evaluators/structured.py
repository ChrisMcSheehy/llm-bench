"""Structured output + function calling.

Two deterministic checks that matter for anyone wiring local models into
pipelines (and that quantisation degrades):

  json_schema — ask for JSON matching a small schema; grade on parse + required
                keys + basic type conformance.
  tool_call   — give a tool signature and a request; grade whether the emitted
                call has the right function name and required arguments (BFCL
                flavoured, without needing the model's native tool API).

Schema checks are done with a tiny built-in validator (no jsonschema dep).
"""
from __future__ import annotations

from typing import Any

from llmbench.evaluators._extract import first_json
from llmbench.evaluators.base import Breakdown, EvalContext, Evaluator, Verdict, View
from llmbench.models import Sample
from llmbench.registry import register

_TYPES = {"string": str, "number": (int, float), "integer": int,
          "boolean": bool, "array": list, "object": dict}


def _validate(obj: Any, schema: dict) -> tuple[bool, str]:
    if schema.get("type") == "object":
        if not isinstance(obj, dict):
            return False, "not an object"
        for key in schema.get("required", []):
            if key not in obj:
                return False, f"missing key {key!r}"
        for key, spec in schema.get("properties", {}).items():
            if key in obj and spec.get("type") in _TYPES:
                if not isinstance(obj[key], _TYPES[spec["type"]]):
                    return False, f"{key!r} wrong type"
        return True, "ok"
    if schema.get("type") == "array":
        return (isinstance(obj, list), "ok" if isinstance(obj, list) else "not an array")
    return True, "ok"


_JSON_TASKS = [
    ("user_obj",
     'Output only a JSON object for a user with keys "name" (string), '
     '"age" (integer) and "active" (boolean).',
     {"type": "object", "required": ["name", "age", "active"],
      "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                     "active": {"type": "boolean"}}}),
    ("order_arr",
     'Output only a JSON array of exactly 3 objects, each with keys "sku" '
     '(string) and "qty" (integer).',
     {"type": "array"}),
]

# tool spec: name, required args, prompt, expected arg values (subset match)
_TOOL_TASKS = [
    ("get_weather",
     'You can call get_weather(city: string, units: string). A user asks: '
     '"What\'s the temperature in Leeds in celsius?" Output only a JSON object '
     '{"name": ..., "arguments": {...}}.',
     {"name": "get_weather", "required": ["city", "units"],
      "expect": {"city": "leeds", "units": "celsius"}}),
    ("create_event",
     'You can call create_event(title: string, date: string). A user asks: '
     '"Put \'Team sync\' on my calendar for 2026-01-15." Output only a JSON '
     'object {"name": ..., "arguments": {...}}.',
     {"name": "create_event", "required": ["title", "date"],
      "expect": {"date": "2026-01-15"}}),
]

_DEFAULTS = {"max_tokens": 256, "temperature": 0.0,
             "tasks": ["json_schema", "tool_call"]}


@register
class StructuredEvaluator(Evaluator):
    name = "structured"
    version = "1"
    default_config = _DEFAULTS
    breakdowns = [Breakdown("pass_rate", ("task",))]
    views = [View("bar", "pass rate by task", x="task")]

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)
        out: list[Sample] = []
        if "json_schema" in cfg["tasks"]:
            for tid, prompt, schema in _JSON_TASKS:
                out.append(await self._json(ctx, cfg, tid, prompt, schema))
        if "tool_call" in cfg["tasks"]:
            for tid, prompt, spec in _TOOL_TASKS:
                out.append(await self._tool(ctx, cfg, tid, prompt, spec))
        return out

    async def _case(self, ctx, cfg, tid, prompt, task: str, grade) -> Sample:
        return await self.run_case(
            ctx, case_id=tid, group=task, dims={"task": task},
            messages=[{"role": "user", "content": prompt}], grade=grade,
            max_tokens=cfg["max_tokens"], temperature=cfg["temperature"])

    async def _json(self, ctx, cfg, tid, prompt, schema) -> Sample:
        def grade(res) -> Verdict:
            obj = first_json(res.text)
            ok, why = (False, "no json") if obj is None else _validate(obj, schema)
            return Verdict(score=1.0 if ok else 0.0, passed=ok,
                           meta={"why": why, "answer": res.text[:200]})

        return await self._case(ctx, cfg, tid, prompt, "json_schema", grade)

    async def _tool(self, ctx, cfg, tid, prompt, spec) -> Sample:
        def grade(res) -> Verdict:
            ok, why = self._grade_tool(first_json(res.text), spec)
            return Verdict(score=1.0 if ok else 0.0, passed=ok,
                           meta={"why": why, "answer": res.text[:200]})

        return await self._case(ctx, cfg, tid, prompt, "tool_call", grade)

    def _grade_tool(self, obj, spec) -> tuple[bool, str]:
        if not isinstance(obj, dict):
            return False, "no json object"
        if obj.get("name") != spec["name"]:
            return False, f"wrong function: {obj.get('name')!r}"
        args = obj.get("arguments") or obj.get("args") or {}
        if not isinstance(args, dict):
            return False, "arguments not an object"
        for r in spec["required"]:
            if r not in args:
                return False, f"missing arg {r!r}"
        for k, v in spec.get("expect", {}).items():
            if str(args.get(k, "")).strip().lower() != str(v).lower():
                return False, f"arg {k!r} = {args.get(k)!r} != {v!r}"
        return True, "ok"
