"""Reading a GGUF header, tested against files this test builds byte by byte.

A fixture cannot be a real model file — the smallest on the probe machine was 83 MB.
So the tests write minimal GGUF files with known contents. The parser was separately
validated against three real models on 2026-08-04, where the values it read matched
what a running llama-server independently reported for the same file (see
docs/ironclad/PROBE-2026-08-04-model-shape.md).
"""
from __future__ import annotations

import struct

import pytest

from llmbench.gguf import read_shape

U32, F32, STRING, ARRAY, BOOL = 4, 6, 8, 9, 7


def _kv(key: str, tag: int, payload: bytes) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", tag) + payload


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _string(v: str) -> bytes:
    b = v.encode()
    return struct.pack("<Q", len(b)) + b


def _array_u32(values: list[int]) -> bytes:
    return struct.pack("<IQ", U32, len(values)) + b"".join(_u32(v) for v in values)


def _array_bool(values: list[bool]) -> bytes:
    return struct.pack("<IQ", BOOL, len(values)) + b"".join(
        struct.pack("<?", v) for v in values)


def _write_gguf(path, entries: list[bytes]) -> str:
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
    blob += struct.pack("<Q", len(entries)) + b"".join(entries)
    path.write_bytes(blob)
    return str(path)


def test_a_simple_dense_model_is_read(tmp_path):
    p = _write_gguf(tmp_path / "m.gguf", [
        _kv("general.architecture", STRING, _string("llama")),
        _kv("llama.block_count", U32, _u32(32)),
        _kv("llama.attention.head_count", U32, _u32(32)),
        _kv("llama.attention.head_count_kv", U32, _u32(8)),
        _kv("llama.attention.key_length", U32, _u32(128)),
        _kv("llama.attention.value_length", U32, _u32(128)),
        _kv("llama.embedding_length", U32, _u32(4096)),
        _kv("llama.context_length", U32, _u32(8192)),
    ])
    s = read_shape(p)
    assert s.architecture == "llama"
    assert s.block_count == 32
    assert s.head_count_kv == [8] * 32, "a scalar must expand to one value per layer"
    assert s.key_length == 128
    assert s.sliding_window_pattern == [False] * 32


def test_per_layer_arrays_are_read_as_arrays(tmp_path):
    """The gemma-4 shape: KV head count and window pattern differ per layer."""
    p = _write_gguf(tmp_path / "g.gguf", [
        _kv("general.architecture", STRING, _string("gemma4")),
        _kv("gemma4.block_count", U32, _u32(6)),
        _kv("gemma4.attention.head_count", U32, _u32(16)),
        _kv("gemma4.attention.head_count_kv", ARRAY, _array_u32([8, 8, 8, 8, 8, 1])),
        _kv("gemma4.attention.key_length", U32, _u32(512)),
        _kv("gemma4.attention.value_length", U32, _u32(512)),
        _kv("gemma4.attention.key_length_swa", U32, _u32(256)),
        _kv("gemma4.attention.value_length_swa", U32, _u32(256)),
        _kv("gemma4.attention.sliding_window", U32, _u32(1024)),
        _kv("gemma4.attention.sliding_window_pattern", ARRAY,
            _array_bool([True] * 5 + [False])),
        _kv("gemma4.embedding_length", U32, _u32(3840)),
    ])
    s = read_shape(p)
    assert s.head_count_kv == [8, 8, 8, 8, 8, 1]
    assert s.sliding_window_pattern == [True, True, True, True, True, False]
    assert s.sliding_window == 1024
    assert s.key_length_swa == 256


def test_a_missing_head_count_kv_falls_back_to_head_count(tmp_path):
    """Models without grouped-query attention omit the field; it then equals head_count."""
    p = _write_gguf(tmp_path / "b.gguf", [
        _kv("general.architecture", STRING, _string("bert")),
        _kv("bert.block_count", U32, _u32(12)),
        _kv("bert.attention.head_count", U32, _u32(12)),
        _kv("bert.embedding_length", U32, _u32(768)),
    ])
    s = read_shape(p)
    assert s.head_count_kv == [12] * 12
    assert s.key_length == 64, "absent key_length is embedding_length / head_count"


def test_a_file_that_is_not_gguf_is_unknown_rather_than_an_error(tmp_path):
    p = tmp_path / "not.gguf"
    p.write_bytes(b"this is not a model")
    assert read_shape(str(p)) is None


def test_a_missing_file_is_unknown(tmp_path):
    assert read_shape(str(tmp_path / "absent.gguf")) is None


def test_a_huge_array_is_skipped_rather_than_loaded(tmp_path):
    """Tokenizer vocabularies are arrays of tens of thousands of strings.

    Reading them would cost memory and time for data this never uses, so they are
    stepped over. The test proves the parser still reaches keys that follow one.
    """
    vocab = struct.pack("<IQ", STRING, 3) + b"".join(_string(w) for w in ("a", "b", "c"))
    p = _write_gguf(tmp_path / "v.gguf", [
        _kv("general.architecture", STRING, _string("llama")),
        _kv("tokenizer.ggml.tokens", ARRAY, vocab),
        _kv("llama.block_count", U32, _u32(4)),
        _kv("llama.attention.head_count", U32, _u32(4)),
        _kv("llama.embedding_length", U32, _u32(256)),
    ])
    s = read_shape(p)
    assert s.block_count == 4, "a key after a skipped array was not reached"
