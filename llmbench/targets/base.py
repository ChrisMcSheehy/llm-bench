"""Target adapters: everything the framework needs to talk to a model backend.

A Target does two things:
  1. detect()  -> ModelFingerprint   (what is deployed, right now)
  2. generate()/count_tokens()       (drive it during evaluation)

All backends expose an OpenAI-compatible /v1/chat/completions, so generation is
shared here. Detection and token-counting differ per backend and are overridden.
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from llmbench.models import ModelFingerprint


class DetectionError(RuntimeError):
    """A backend could not be interrogated, so it has no identity.

    Raised rather than returning a fingerprint built from defaults. Identity is what
    every comparison, pooled average and stored vote is keyed by, so a detection path
    that degrades quietly turns a transient fault into permanent bad data filed under a
    plausible-looking hash. The orchestrator records this per target and carries on to
    the next one (design D3d), so raising costs the sweep nothing.
    """


@dataclass
class GenResult:
    text: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    latency_ms: float
    tok_per_sec: Optional[float]
    server_prompt_tps: Optional[float] = None
    server_gen_tps: Optional[float] = None
    truncated: Optional[bool] = None
    #: Why generation stopped, straight from the choice: "stop", "length", "tool_calls".
    #: The only reliable truncation signal an OpenAI-shaped response carries.
    finish_reason: Optional[str] = None
    raw: Optional[dict[str, Any]] = None

    @property
    def unusable_reason(self) -> Optional[str]:
        """Why this response cannot be graded, or None if it can.

        A response that never reached an answer is not a wrong answer, and grading it as
        one produces a confident zero where there is no measurement at all - the same
        category error design D3 removed from the context ladder, one layer further down.

        Deliberately narrow. Only two situations count, and both are positive evidence
        that something other than the model's ability produced the empty text:

        1. It was cut off. `finish_reason == "length"` means our token budget ended the
           response, not the model.
        2. It emitted tokens that did not arrive as content. A reasoning model puts them
           in `reasoning_content`, which this bench does not read.

        A model that stopped of its own accord having produced nothing is **graded**, and
        that matters: it is a genuine failure to answer, and calling it unusable would
        hide real incapacity, which is the opposite mistake and the worse one.
        """
        if self.text.strip():
            return None
        if self.finish_reason == "length":
            return ("no answer: the whole token budget went on the response without one "
                    "arriving. Raise this evaluator's answer token budget - a reasoning "
                    "model spends the budget thinking before it answers")
        if self.output_tokens:
            return (f"no answer: the model emitted {self.output_tokens} tokens but none "
                    "arrived as content - a reasoning model returns them separately")
        return None


class Target(abc.ABC):
    engine: str = "generic"

    def __init__(self, base_url: str, model: Optional[str] = None,
                 api_key: Optional[str] = None, timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    # ---- helpers ----------------------------------------------------------
    @property
    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def _get_required(self, path: str) -> dict:
        """Read an endpoint the identity depends on. Failure is an error, not a blank.

        Anything short of a parseable response - refused connection, a 503 from a server
        still loading its model, malformed JSON - stops detection here.
        """
        try:
            r = await self._client.get(f"{self.base_url}{path}", headers=self._headers)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            raise DetectionError(
                f"{self.engine}: could not read {path} from {self.base_url} ({exc!r}). "
                f"Detection stops here rather than filing results under an identity "
                f"assembled from defaults. If the server is still loading its model, "
                f"wait for it to finish and try again."
            ) from exc

    async def _get(self, path: str) -> Optional[dict]:
        """Read an endpoint the identity can do without. None when it is not there.

        For genuinely optional probes only - a fallback URL, a nice-to-have version
        string. Anything the fingerprint depends on goes through `_get_required`.
        """
        try:
            return await self._get_required(path)
        except DetectionError:
            return None

    # ---- generation (shared, OpenAI chat) --------------------------------
    async def generate(self, messages: list[dict], *, max_tokens: int = 512,
                       temperature: float = 0.0, extra: Optional[dict] = None) -> GenResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        # llama.cpp honours this and returns a `timings` block; harmless elsewhere.
        body["timings_per_token"] = True
        if extra:
            body.update(extra)

        t0 = time.perf_counter()
        r = await self._client.post(
            f"{self.base_url}/v1/chat/completions", headers=self._headers, json=body,
        )
        r.raise_for_status()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        data = r.json()

        text, finish_reason = "", None
        choices = data.get("choices") or []
        if choices:
            text = (choices[0].get("message") or {}).get("content") or ""
            finish_reason = choices[0].get("finish_reason")

        usage = data.get("usage") or {}
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")

        timings = data.get("timings") or {}
        server_gen = timings.get("predicted_per_second")
        server_prompt = timings.get("prompt_per_second")

        tps = None
        if out_tok and dt_ms > 0:
            tps = out_tok / (dt_ms / 1000.0)

        return GenResult(
            text=text, input_tokens=in_tok, output_tokens=out_tok,
            latency_ms=dt_ms, tok_per_sec=tps,
            server_prompt_tps=server_prompt, server_gen_tps=server_gen,
            # From the choice, not from a top-level "truncated" key: no OpenAI-shaped
            # response has ever carried one, so this field read None on every sample
            # recorded before 2026-08-05.
            truncated=(finish_reason == "length") if finish_reason else None,
            finish_reason=finish_reason, raw=data,
        )

    # ---- to be implemented per backend -----------------------------------
    @abc.abstractmethod
    async def detect(self) -> ModelFingerprint:
        ...

    @abc.abstractmethod
    async def count_tokens(self, text: str) -> int:
        ...


def approx_tokens(text: str) -> int:
    """Backend-agnostic fallback. ~4 chars/token is a decent English heuristic."""
    return max(1, round(len(text) / 4))
