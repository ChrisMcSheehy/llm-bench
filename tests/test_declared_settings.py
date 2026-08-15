"""Declaring the two settings that no endpoint reports.

A server llmbench did not start reports neither its physical batch size nor whether its
cache is unified, and both change the memory figure — so on a multi-slot server, or any
sliding-window model, the honest answer is "unknown" (D8b, D8c). Three of five models on
the machine where this was written are sliding-window, so that is the common case rather
than a corner.

D2 already answers this shape of problem: what cannot be read, the user declares. A
declaration is a claim, so it feeds the memory figure - which is derived, and records
its inputs - and never the identity hash, which must only ever record what was observed.
"""
from __future__ import annotations

import asyncio
import copy
import json
import pathlib

from llmbench.targets.base import DetectionError
from llmbench.targets.llamacpp import LlamaCppTarget

_FIXTURE = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "llamacpp_b10144.json")
    .read_text(encoding="utf-8"))


def _swa_model(tmp_path) -> str:
    """A sliding-window model, whose cache size depends on the batch size."""
    from tests.test_gguf import (_array_bool, _array_u32, _kv, _string, _u32,
                                 _write_gguf, ARRAY, STRING, U32)
    return _write_gguf(tmp_path / "swa.gguf", [
        _kv("general.architecture", STRING, _string("gemma4")),
        _kv("gemma4.block_count", U32, _u32(4)),
        _kv("gemma4.attention.head_count", U32, _u32(8)),
        _kv("gemma4.attention.head_count_kv", ARRAY, _array_u32([2, 2, 2, 2])),
        _kv("gemma4.attention.key_length", U32, _u32(64)),
        _kv("gemma4.attention.value_length", U32, _u32(64)),
        _kv("gemma4.attention.key_length_swa", U32, _u32(32)),
        _kv("gemma4.attention.value_length_swa", U32, _u32(32)),
        _kv("gemma4.attention.sliding_window", U32, _u32(512)),
        _kv("gemma4.attention.sliding_window_pattern", ARRAY,
            _array_bool([True, True, True, False])),
        _kv("gemma4.embedding_length", U32, _u32(512)),
    ])


def _dense_model(tmp_path) -> str:
    """A plain model with no sliding window, so only the slot count is in question.

    Built here rather than inherited from the fixture's `model_path`. That path was
    captured on the author's machine and still resolves there, so a test relying on it
    reads a real model at home and nothing anywhere else - the same defect recorded in
    LESSONS.md as a-captured-fixture-carries-paths-that-still-exist-at-home, seen from the
    other side: that one depended on the file being absent, this one on it being present.
    Both let the machine decide the verdict.
    """
    from tests.test_gguf import _kv, _string, _u32, _write_gguf, STRING, U32
    return _write_gguf(tmp_path / "dense.gguf", [
        _kv("general.architecture", STRING, _string("llama")),
        _kv("llama.block_count", U32, _u32(4)),
        _kv("llama.attention.head_count", U32, _u32(8)),
        _kv("llama.attention.head_count_kv", U32, _u32(2)),
        _kv("llama.attention.key_length", U32, _u32(64)),
        _kv("llama.attention.value_length", U32, _u32(64)),
        _kv("llama.embedding_length", U32, _u32(512)),
    ])


def _detect(tmp_path, *, model=None, slots=1, declared=None, n_ctx=1024):
    f = copy.deepcopy(_FIXTURE["plain"])
    if model:
        f["props"]["model_path"] = model
    f["props"]["default_generation_settings"]["n_ctx"] = n_ctx
    f["props"]["total_slots"] = slots

    class _Served(LlamaCppTarget):
        def __init__(self):
            super().__init__(base_url="http://127.0.0.1:0", declared=declared)
            self._served = {"/props": f["props"], "/v1/models": f["v1_models"]}

        async def _get_required(self, path):
            # The read primitive, so `_get` is served from here too. A path this
            # fixture does not carry raises, as an absent endpoint really does.
            body = self._served.get(path)
            if body is None:
                raise DetectionError(f"fixture has no {path}")
            return body

    async def run():
        t = _Served()
        try:
            return await t.detect()
        finally:
            await t.aclose()
    return asyncio.run(run())


def test_without_a_declaration_a_window_model_is_still_unknown(tmp_path):
    """The baseline this exists to improve on."""
    fp = _detect(tmp_path, model=_swa_model(tmp_path))
    assert fp.kv_cache_bytes is None


def test_a_declared_batch_size_produces_a_figure(tmp_path):
    fp = _detect(tmp_path, model=_swa_model(tmp_path), declared={"ubatch": 512})
    # 3 window layers cache window+batch = 1024 tokens at head dimension 32; the one
    # full-attention layer caches all 1024 at 64. Times two for keys and values.
    expected = 3 * (2 * 32 * 1024 * 2) * 2 + (2 * 64 * 1024 * 2) * 2
    assert fp.kv_cache_bytes == expected


def test_a_declared_value_is_recorded_as_declared_not_observed(tmp_path):
    """A claim about a server is not a reading of it, and the record must say which."""
    fp = _detect(tmp_path, model=_swa_model(tmp_path), declared={"ubatch": 512})
    assert fp.kv_cache_derivation["n_ubatch"] == 512
    assert fp.kv_cache_derivation["n_ubatch_source"] == "declared"


def test_declaring_does_not_claim_the_launch_settings_were_observed(tmp_path):
    """D6a's field means the backend told us. A declaration is not the backend."""
    fp = _detect(tmp_path, model=_swa_model(tmp_path), declared={"ubatch": 512})
    assert fp.launch_settings_observed is False
    assert "launch:unreported" in fp.label


def test_declaring_does_not_change_the_identity(tmp_path):
    """The memory figure is derived and may use a claim. The hash may not."""
    model = _swa_model(tmp_path)
    bare = _detect(tmp_path, model=model)
    declared = _detect(tmp_path, model=model, declared={"ubatch": 512})
    assert bare.fingerprint_hash == declared.fingerprint_hash


def test_a_declared_unified_flag_resolves_a_multi_slot_server(tmp_path):
    """Four slots and no argv is unknown; declaring the cache shape settles it."""
    model = _dense_model(tmp_path)
    unknown = _detect(tmp_path, model=model, slots=4)
    assert unknown.kv_cache_bytes is None

    unified = _detect(tmp_path, model=model, slots=4, declared={"kv_unified": True})
    split = _detect(tmp_path, model=model, slots=4, declared={"kv_unified": False})
    assert unified.kv_cache_bytes is not None
    assert split.kv_cache_bytes == unified.kv_cache_bytes * 4
    assert split.kv_cache_derivation["kv_unified_source"] == "declared"


def test_an_observed_value_always_beats_a_declared_one(tmp_path):
    """We saw what the server was started with; the user's claim cannot improve on it."""
    f = copy.deepcopy(_FIXTURE["plain"])
    f["props"]["model_path"] = _swa_model(tmp_path)
    f["props"]["default_generation_settings"]["n_ctx"] = 1024
    f["props"]["total_slots"] = 1

    class _Served(LlamaCppTarget):
        def __init__(self):
            super().__init__(base_url="http://127.0.0.1:0",
                             known_args=["-ub", "256"], declared={"ubatch": 512})
            self._served = {"/props": f["props"], "/v1/models": f["v1_models"]}

        async def _get_required(self, path):
            # The read primitive, so `_get` is served from here too. A path this
            # fixture does not carry raises, as an absent endpoint really does.
            body = self._served.get(path)
            if body is None:
                raise DetectionError(f"fixture has no {path}")
            return body

    async def run():
        t = _Served()
        try:
            return await t.detect()
        finally:
            await t.aclose()

    fp = asyncio.run(run())
    assert fp.kv_cache_derivation["n_ubatch"] == 256
    assert fp.kv_cache_derivation["n_ubatch_source"] == "observed"
