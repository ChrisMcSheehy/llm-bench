"""Detection tested through the seam it actually crosses: a served payload.

Phase 2 added four launch-setting fields to the identity and shipped 21 passing tests,
none of which ran `detect()`. Against a real server the fields all arrived as None,
because the adapter reads them from a `status.args` field that only exists when
llama-server runs in router mode. Unit tests either side of that seam could not see it.

The payloads here are captured from llama-server b10144-d73c1d6b2, not hand-written.
See docs/ironclad/PROBE-2026-08-04-model-shape.md.
"""
from __future__ import annotations

import asyncio
import copy
import json
import pathlib

import pytest

from llmbench.targets.base import DetectionError
from llmbench.targets.llamacpp import LlamaCppTarget

_FIXTURE = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "llamacpp_b10144.json")
    .read_text(encoding="utf-8")
)


class _ServedTarget(LlamaCppTarget):
    """A target whose HTTP GETs are answered from a captured payload."""

    def __init__(self, props: dict, v1_models: dict):
        super().__init__(base_url="http://127.0.0.1:0")
        self._served = {"/props": props, "/v1/models": v1_models}

    async def _get_required(self, path: str):
        """The read primitive, so both `_get` and `_get_required` are served from here.

        A path this fixture does not carry raises, exactly as an absent endpoint does on
        a real server - which is what makes `_get`'s optional-probe behaviour real here
        rather than assumed.
        """
        body = self._served.get(path)
        if body is None:
            raise DetectionError(f"fixture has no {path}")
        return body


def _detect(props: dict, v1_models: dict):
    async def run():
        t = _ServedTarget(props, v1_models)
        try:
            return await t.detect()
        finally:
            await t.aclose()
    return asyncio.run(run())


def _plain():
    f = copy.deepcopy(_FIXTURE["plain"])
    return _detect(f["props"], f["v1_models"])


def _router(extra_args: list[str] | None = None):
    """Router mode, where the server does report the argv it launched a model with.

    The captured router payload's own args carry only --host/--port/--alias/--model,
    because the probe ran without a preset. Extra flags are appended to that real
    structure rather than invented alongside it, so the shape stays the captured one.
    """
    props = copy.deepcopy(_FIXTURE["plain"]["props"])
    models = copy.deepcopy(_FIXTURE["router"]["v1_models"])
    if extra_args:
        models["data"][0]["status"]["args"].extend(extra_args)
    return _detect(props, models)


def test_the_plain_deployment_reports_no_launch_settings():
    """The regression. This server was launched with -ngl 99 and reports none of it."""
    fp = _plain()
    assert fp.launch_args == [], "the captured plain payload should carry no argv"
    assert fp.n_gpu_layers is None
    assert fp.n_batch is None
    assert fp.n_ubatch is None


def test_the_plain_deployment_says_the_settings_were_not_observed():
    """None must mean 'unknown', not 'unset' — that distinction is the whole fix."""
    assert _plain().launch_settings_observed is False


def test_router_mode_does_observe_them():
    fp = _router(["-ngl", "99", "-b", "4096", "-ub", "512"])
    assert fp.launch_settings_observed is True
    assert fp.n_gpu_layers == "99"
    assert fp.n_batch == 4096
    assert fp.n_ubatch == 512


def test_an_observed_setup_never_pools_with_an_unobserved_one():
    """The point of D6a.

    A run whose layer split is known to be 99 and a run whose layer split is simply
    unknown are not the same configuration, and averaging them is the silent failure
    this work exists to remove.
    """
    known = _router(["-ngl", "99"])
    unknown = _plain()
    assert known.fingerprint_hash != unknown.fingerprint_hash


def test_the_slot_count_comes_from_the_server_rather_than_the_argv():
    """-np defaults to -1 (auto). The argv can never tell you what auto resolved to.

    The captured server reported total_slots 4 having been given no -np at all, so the
    server-reported figure carries information the launch arguments could not.
    """
    fp = _plain()
    assert fp.n_parallel == 4, "total_slots was not used"


def test_a_server_reported_slot_count_beats_a_stale_argument():
    """Both sources present and disagreeing: the server's own figure must win.

    llama-server was separately observed rewriting a launch value it could not honour
    ('setting n_batch = n_ubatch = 512'), so an argument is a request, not a fact.
    """
    fp = _router(["-np", "9"])
    assert fp.n_parallel == 4, "the argv value overrode the server's own slot count"


def test_the_label_marks_a_run_whose_settings_were_not_observed():
    """Two rows differing only in what is unknown about them must read differently."""
    plain_label = _plain().label
    router_label = _router(["-ngl", "99"]).label
    assert "launch:unreported" in plain_label
    assert "launch:unreported" not in router_label


@pytest.mark.parametrize("field", ["engine", "model_id", "n_ctx", "kv_cache_k"])
def test_detection_still_fills_the_fields_it_always_did(field):
    """Guards against the fix breaking ordinary detection."""
    assert getattr(_plain(), field) is not None


# --- launched by us: the same plain payload, but we know what we started ------------

def _launched(known_args: list[str]):
    """Detection against the plain-mode payload, as if llmbench had started the server."""
    f = copy.deepcopy(_FIXTURE["plain"])

    async def run():
        t = _ServedTarget(f["props"], f["v1_models"])
        t.known_args = list(known_args)
        try:
            return await t.detect()
        finally:
            await t.aclose()
    return asyncio.run(run())


def test_launching_the_server_ourselves_completes_the_identity():
    """The point of the launcher.

    This is the same server payload that yields 'launch:unreported' when we merely
    connected to it. Because we started it, the layer split is known.
    """
    fp = _launched(["-ngl", "99", "-c", "16384"])
    assert fp.n_gpu_layers == "99"
    assert fp.launch_settings_observed is True
    assert "launch:unreported" not in fp.label


def test_two_layer_splits_we_launched_are_two_identities():
    """Design criterion 7, finally met on an ordinary (non-router) build."""
    a = _launched(["-ngl", "40"])
    b = _launched(["-ngl", "99"])
    assert a.fingerprint_hash != b.fingerprint_hash
    assert a.label != b.label


def test_the_server_still_wins_where_it_reports_a_value():
    """We asked for 9 slots; the server says it has 4. The server is describing reality.

    -np defaults to auto, and llama-server was separately observed rewriting a batch
    size it could not honour, so an argument is a request rather than a fact.
    """
    assert _launched(["-np", "9"]).n_parallel == 4


def test_a_served_argv_beats_the_one_we_supplied():
    """In router mode the server reports its own argv; that is what is running."""
    props = copy.deepcopy(_FIXTURE["plain"]["props"])
    models = copy.deepcopy(_FIXTURE["router"]["v1_models"])
    models["data"][0]["status"]["args"].extend(["-ngl", "10"])

    async def run():
        t = _ServedTarget(props, models)
        t.known_args = ["-ngl", "99"]        # what we would have asked for
        try:
            return await t.detect()
        finally:
            await t.aclose()

    assert asyncio.run(run()).n_gpu_layers == "10"


def test_detection_reports_unknown_when_the_model_file_is_not_reachable(tmp_path):
    """A server on another host names a path this machine cannot open.

    Unknown is the correct answer there, and it must be None rather than 0.

    The path is made absent deliberately rather than relying on the captured one being
    absent: that payload was captured on a developer machine, where the file it names
    still exists, so a test written against it would assert a property of the machine
    instead of a property of the code.
    """
    f = copy.deepcopy(_FIXTURE["plain"])
    f["props"]["model_path"] = str(tmp_path / "not-on-this-machine.gguf")
    fp = _detect(f["props"], f["v1_models"])

    assert fp.kv_cache_bytes is None
    assert fp.kv_cache_derivation.get("known") is False


def test_detection_computes_the_estimate_when_the_file_is_there(tmp_path):
    """A real GGUF header written to disk, served as the model_path."""
    import copy

    # `tests` has no __init__.py, but pyproject.toml sets pythonpath = ["."], which puts
    # the repository root on sys.path and makes `tests` importable as a namespace
    # package. Verified working on 2026-08-04. If this ever raises ImportError, that
    # setting is what changed.
    from tests.test_gguf import _kv, _string, _u32, _write_gguf, STRING, U32

    model = _write_gguf(tmp_path / "tiny.gguf", [
        _kv("general.architecture", STRING, _string("llama")),
        _kv("llama.block_count", U32, _u32(4)),
        _kv("llama.attention.head_count", U32, _u32(8)),
        _kv("llama.attention.head_count_kv", U32, _u32(2)),
        _kv("llama.attention.key_length", U32, _u32(64)),
        _kv("llama.attention.value_length", U32, _u32(64)),
        _kv("llama.embedding_length", U32, _u32(512)),
    ])

    f = copy.deepcopy(_FIXTURE["plain"])
    f["props"]["model_path"] = model
    f["props"]["default_generation_settings"]["n_ctx"] = 1024
    # One slot, so the reported context is unambiguously the whole cache. The captured
    # payload has four, which under D8b is a configuration whose total cannot be known
    # from HTTP alone - a different case, covered by its own test below.
    f["props"]["total_slots"] = 1
    fp = _detect(f["props"], f["v1_models"])

    assert fp.kv_cache_bytes == 4 * 2 * 64 * 2 * 2 * 1024
    assert fp.kv_cache_derivation["architecture"] == "llama"
    assert fp.kv_cache_derivation["n_ctx"] == 1024


# --- how many tokens the cache must hold (D8b) --------------------------------------
#
# The server reports the context *per slot*, so on a multi-slot server the total is the
# reported figure times the slot count - unless the cache is unified, which no endpoint
# reports. See docs/ironclad/PROBE-2026-08-04-host-facts.md, Finding 4.

def _tiny_model(tmp_path):
    from tests.test_gguf import _kv, _string, _u32, _write_gguf, STRING, U32
    return _write_gguf(tmp_path / "tiny.gguf", [
        _kv("general.architecture", STRING, _string("llama")),
        _kv("llama.block_count", U32, _u32(4)),
        _kv("llama.attention.head_count", U32, _u32(8)),
        _kv("llama.attention.head_count_kv", U32, _u32(2)),
        _kv("llama.attention.key_length", U32, _u32(64)),
        _kv("llama.attention.value_length", U32, _u32(64)),
        _kv("llama.embedding_length", U32, _u32(512)),
    ])


#: The tiny model's cache for 1024 tokens: 4 layers x 2 kv heads x 64 dim x 2 (K and V)
#: x 2 bytes x 1024 tokens.
_PER_1024 = 4 * 2 * 64 * 2 * 2 * 1024


def _served(tmp_path, *, slots: int, known_args: list[str] | None = None, n_ctx=1024):
    """Detection against the captured payload, with the model reachable on this machine."""
    f = copy.deepcopy(_FIXTURE["plain"])
    f["props"]["model_path"] = _tiny_model(tmp_path)
    f["props"]["default_generation_settings"]["n_ctx"] = n_ctx
    f["props"]["total_slots"] = slots

    async def run():
        t = _ServedTarget(f["props"], f["v1_models"])
        t.known_args = list(known_args or [])
        try:
            return await t.detect()
        finally:
            await t.aclose()
    return asyncio.run(run())


def test_a_single_slot_server_needs_no_slot_reasoning_at_all(tmp_path):
    """Per-slot and total coincide on one slot, so the figure stays knowable.

    This is the ordinary deployment, and it must keep a real number even though we did
    not launch it and cannot tell whether its cache is unified.
    """
    fp = _served(tmp_path, slots=1)
    assert fp.kv_cache_bytes == _PER_1024
    assert fp.kv_cache_derivation["cache_pools"] == 1


def test_a_multi_slot_server_we_did_not_launch_reports_unknown(tmp_path):
    """Two candidate answers a factor of total_slots apart, and no way to choose.

    D8b: a confident wrong number is worse than no number, so this reports neither.
    """
    fp = _served(tmp_path, slots=4)
    assert fp.kv_cache_bytes is None
    assert fp.kv_cache_derivation["known"] is False
    assert "slot" in fp.kv_cache_derivation["reason"].lower()
    assert fp.kv_cache_derivation["total_slots"] == 4


def test_launching_with_an_explicit_slot_count_sizes_the_whole_cache(tmp_path):
    """-np turns unification off, so each of the two slots holds its own 1024 tokens."""
    fp = _served(tmp_path, slots=2, known_args=["-np", "2"])
    assert fp.kv_cache_bytes == _PER_1024 * 2
    assert fp.kv_cache_derivation["kv_unified"] is False
    assert fp.kv_cache_derivation["cache_pools"] == 2


def test_launching_without_a_slot_count_leaves_the_cache_unified(tmp_path):
    """-kvu defaults on when the slot count is auto, so the pool is not multiplied."""
    fp = _served(tmp_path, slots=4, known_args=["-ngl", "99"])
    assert fp.kv_cache_bytes == _PER_1024
    assert fp.kv_cache_derivation["kv_unified"] is True


def test_an_explicit_unified_flag_beats_the_slot_count(tmp_path):
    """-kvu with -np is the case the default rule would otherwise get backwards."""
    fp = _served(tmp_path, slots=2, known_args=["-np", "2", "-kvu"])
    assert fp.kv_cache_bytes == _PER_1024


def test_an_explicit_no_unified_flag_is_honoured_without_a_slot_count(tmp_path):
    fp = _served(tmp_path, slots=2, known_args=["-ngl", "99", "--no-kv-unified"])
    assert fp.kv_cache_bytes == _PER_1024 * 2


def test_a_split_cache_is_two_caches_rather_than_one_long_one(tmp_path):
    """Each slot gets its own cache, so the slot count multiplies bytes, not tokens.

    On a dense model the two readings agree, which is why this test uses a
    sliding-window one: a window layer caches the same tokens however long the context
    is, so doubling the context does not double it - but running two slots does.

    The server itself reports the distinction: `llama_kv_cache: size = 128.00 MiB
    (4096 cells, 8 layers, 2/2 seqs)` - cells are per sequence and the buffer holds
    both. Measured 2026-08-04, docs/ironclad/PROBE-2026-08-04-host-facts.md.
    """
    from tests.test_gguf import (_array_bool, _array_u32, _kv, _string, _u32,
                                 _write_gguf, ARRAY, STRING, U32)

    model = _write_gguf(tmp_path / "swa.gguf", [
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

    f = copy.deepcopy(_FIXTURE["plain"])
    f["props"]["model_path"] = model
    f["props"]["default_generation_settings"]["n_ctx"] = 1024
    f["props"]["total_slots"] = 2

    async def run():
        t = _ServedTarget(f["props"], f["v1_models"])
        t.known_args = ["-np", "2"]
        try:
            return await t.detect()
        finally:
            await t.aclose()
    fp = asyncio.run(run())

    # A window layer caches window + ubatch = 512 + 512 = 1024 tokens, at head
    # dimension 32; the one full-attention layer caches the whole 1024 at dimension 64.
    # The trailing x2 on each term is keys and values.
    per_slot = 3 * (2 * 32 * 1024 * 2) * 2 + (2 * 64 * 1024 * 2) * 2
    assert fp.kv_cache_bytes == per_slot * 2, "the slot count must multiply bytes"

    one_long_context = 3 * (2 * 32 * 1024 * 2) * 2 + (2 * 64 * 2048 * 2) * 2
    assert fp.kv_cache_bytes != one_long_context, (
        "sizing one cache of 2048 tokens is the wrong shape: a window layer caches the "
        "same amount either way, so this understates a split cache")
