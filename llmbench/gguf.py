"""Read a model's shape out of a GGUF file header.

llama.cpp's HTTP API reports no layer count, no key/value head count and no head
dimension (verified against a running server on 2026-08-04 — see
docs/ironclad/PROBE-2026-08-04-model-shape.md). Those numbers do exist in the model
file's header, which is a key/value block at the very start: reading it touches only
the first few hundred kilobytes and never loads the model.

Nothing here raises on a bad file. A shape that cannot be read is None, because the
caller's honest answer in that case is "unknown" rather than a failure.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, BinaryIO, Optional

# GGUF metadata value type tags, from the format specification.
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STRING, _ARRAY, _U64, _I64, _F64 = range(13)

_FIXED: dict[int, tuple[str, int]] = {
    _U8: ("<B", 1), _I8: ("<b", 1), _U16: ("<H", 2), _I16: ("<h", 2),
    _U32: ("<I", 4), _I32: ("<i", 4), _F32: ("<f", 4), _BOOL: ("<?", 1),
    _U64: ("<Q", 8), _I64: ("<q", 8), _F64: ("<d", 8),
}

# Reading an array this long into memory serves nothing here: the only arrays this
# module wants are per-layer, one entry per block, and no model has this many layers.
_MAX_ARRAY = 4096


@dataclass
class ModelShape:
    """Everything the KV-cache calculation needs, already expanded per layer."""

    architecture: str
    block_count: int
    head_count: int
    head_count_kv: list[int]          # one entry per layer
    key_length: int
    value_length: int
    sliding_window_pattern: list[bool]  # one entry per layer; all False if not declared
    sliding_window: Optional[int] = None
    key_length_swa: Optional[int] = None
    value_length_swa: Optional[int] = None
    context_length: Optional[int] = None


def _read(fh: BinaryIO, n: int) -> bytes:
    b = fh.read(n)
    if len(b) != n:
        raise ValueError("truncated GGUF header")
    return b


def _read_string(fh: BinaryIO) -> str:
    n = struct.unpack("<Q", _read(fh, 8))[0]
    return _read(fh, n).decode("utf-8", "replace")


def _read_value(fh: BinaryIO, tag: int) -> Any:
    if tag in _FIXED:
        fmt, size = _FIXED[tag]
        return struct.unpack(fmt, _read(fh, size))[0]
    if tag == _STRING:
        return _read_string(fh)
    if tag == _ARRAY:
        elem = struct.unpack("<I", _read(fh, 4))[0]
        count = struct.unpack("<Q", _read(fh, 8))[0]
        if elem in _FIXED and count <= _MAX_ARRAY:
            fmt, size = _FIXED[elem]
            return [struct.unpack(fmt, _read(fh, size))[0] for _ in range(count)]
        # Too long to be per-layer data, or an element type this never needs: step over
        # it so that keys after it are still reachable.
        if elem in _FIXED:
            fh.seek(_FIXED[elem][1] * count, 1)
        elif elem == _STRING:
            for _ in range(count):
                fh.seek(struct.unpack("<Q", _read(fh, 8))[0], 1)
        else:
            raise ValueError(f"unsupported GGUF array element type {elem}")
        return None
    raise ValueError(f"unsupported GGUF value type {tag}")


def read_header(path: str) -> Optional[dict[str, Any]]:
    """Every key/value pair in the header, or None if this is not a readable GGUF file."""
    try:
        with open(path, "rb") as fh:
            if _read(fh, 4) != b"GGUF":
                return None
            _read(fh, 4)                      # format version, unused
            _read(fh, 8)                      # tensor count, unused
            n_kv = struct.unpack("<Q", _read(fh, 8))[0]
            out: dict[str, Any] = {}
            for _ in range(n_kv):
                key = _read_string(fh)
                tag = struct.unpack("<I", _read(fh, 4))[0]
                out[key] = _read_value(fh, tag)
            return out
    except (OSError, ValueError, struct.error, UnicodeDecodeError):
        return None


def _per_layer(value: Any, layers: int) -> Optional[list]:
    """Expand a value that may be a scalar or already one entry per layer."""
    if value is None:
        return None
    if isinstance(value, list):
        return value if len(value) == layers else None
    return [value] * layers


def read_shape(path: str) -> Optional[ModelShape]:
    """The shape needed to size a KV cache, or None if it cannot be determined."""
    kv = read_header(path)
    if not kv:
        return None
    arch = kv.get("general.architecture")
    if not isinstance(arch, str):
        return None

    def field(name: str) -> Any:
        return kv.get(f"{arch}.{name}")

    layers = field("block_count")
    heads = field("attention.head_count")
    n_embd = field("embedding_length")
    if not isinstance(layers, int) or not isinstance(heads, int) or heads <= 0:
        return None

    heads_kv = _per_layer(field("attention.head_count_kv"), layers)
    if heads_kv is None:
        # Absent means no grouped-query attention: every attention head has its own
        # key/value head.
        heads_kv = [heads] * layers

    key_length = field("attention.key_length")
    value_length = field("attention.value_length")
    if not isinstance(key_length, int):
        # The documented default: the embedding width divided across the heads.
        if not isinstance(n_embd, int):
            return None
        key_length = n_embd // heads
    if not isinstance(value_length, int):
        value_length = key_length

    swa_pattern = _per_layer(field("attention.sliding_window_pattern"), layers)
    if swa_pattern is None:
        swa_pattern = [False] * layers

    return ModelShape(
        architecture=arch,
        block_count=layers,
        head_count=heads,
        head_count_kv=[int(h) for h in heads_kv],
        key_length=key_length,
        value_length=value_length,
        sliding_window_pattern=[bool(b) for b in swa_pattern],
        sliding_window=field("attention.sliding_window"),
        key_length_swa=field("attention.key_length_swa"),
        value_length_swa=field("attention.value_length_swa"),
        context_length=field("context_length"),
    )
