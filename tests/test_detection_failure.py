"""A backend that cannot be interrogated has no identity.

Found on 2026-08-05: `llmbench detect` against a real llama-server returned a complete
fingerprint of nothing - model_id "unknown", no n_ctx, no build - and hashed it into
fdaa92e6cab29d3f, which is indistinguishable from a real identity. The same all-empty
fingerprint turned out to be sitting in the database from 2026-08-02, unnoticed.

Identity is what every comparison, pooled average and stored vote is keyed by, so a
detection path that degrades to defaults turns a transient fault into permanent bad data.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from llmbench.targets.base import DetectionError
from llmbench.targets.llamacpp import LlamaCppTarget
from llmbench.targets.ollama import OllamaTarget

# Trimmed from a real llama-server b10144-d73c1d6b2 response.
_PROPS = {
    "default_generation_settings": {"n_ctx": 32768, "params": {"temperature": 1.0}},
    "total_slots": 4,
    "model_path": "/models/Qwen3-8B-Q4_K_M.gguf",
    "model_ftype": "Q4_K_M",
    "build_info": "b10144-d73c1d6b2",
}


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _serve(routes: dict[str, dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={"error": "no such endpoint"})
        return httpx.Response(200, json=body)
    return handler


def _with_client(target, handler):
    target._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return target


def _detect(target):
    async def go():
        try:
            return await target.detect()
        finally:
            await target.aclose()
    return asyncio.run(go())


def test_an_unreachable_llama_server_is_an_error_not_an_empty_identity():
    target = _with_client(LlamaCppTarget("http://127.0.0.1:9"), _refuse)

    with pytest.raises(DetectionError) as caught:
        _detect(target)

    message = str(caught.value)
    assert "/props" in message, message
    assert "127.0.0.1:9" in message, message


def test_a_server_that_refuses_the_props_endpoint_is_also_an_error():
    """A 503 while a model is still loading looks exactly like this."""
    target = _with_client(LlamaCppTarget("http://x"),
                          lambda r: httpx.Response(503, json={"error": "loading"}))

    with pytest.raises(DetectionError):
        _detect(target)


def test_detection_still_works_when_the_server_answers():
    """The success condition. Making every failure raise would also stop the empty
    fingerprints, and would leave the bench unable to detect anything at all."""
    target = _with_client(LlamaCppTarget("http://x"), _serve({"/props": _PROPS}))

    fp = _detect(target)

    assert fp.n_ctx == 32768
    assert fp.build_number == 10144
    assert fp.build_commit == "d73c1d6b2"
    assert fp.n_parallel == 4
    assert fp.quant == "Q4_K_M"


def test_a_missing_optional_endpoint_does_not_stop_detection():
    """/v1/models is absent above and detection succeeded anyway: only the endpoint
    that carries the identity is required."""
    target = _with_client(LlamaCppTarget("http://x"), _serve({"/props": _PROPS}))

    assert _detect(target).model_id == "/models/Qwen3-8B-Q4_K_M.gguf"


def test_an_unreachable_ollama_is_an_error_too():
    """`_get` is on the shared base class, so every adapter inherited the behaviour."""
    target = _with_client(OllamaTarget("http://127.0.0.1:9", model="qwen3"), _refuse)

    with pytest.raises(DetectionError):
        _detect(target)
