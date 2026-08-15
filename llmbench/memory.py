"""What a configuration costs in memory, as arithmetic rather than measurement.

The key/value cache is the memory a server sets aside to remember the conversation so
far. Its size follows from the model's shape and two settings the user controls: the
context length, and how hard the cache is compressed. The answer is identical on every
machine, which is why it belongs beside the configuration rather than beside the speed
figures — and why it can be answered before anything is loaded.

Every path that cannot produce a correct number returns None. None means unknown and
must never be rendered as zero, which would read as "this costs nothing".
"""
from __future__ import annotations

from typing import Optional

from llmbench.gguf import ModelShape

# Bytes per cached element, per cache type. Derived from the block structs in ggml's
# ggml-common.h, where ggml_half is 2 bytes and each quantised block covers 32 elements:
# q8_0 is 2 + 32 = 34 bytes per 32 values, q4_0 is 2 + 16 = 18, and so on. The nine keys
# here are exactly the values llama-server's --cache-type-k accepts.
#
# A type absent from this table is unknown, never estimated: a fork can add cache types
# (the machine this was written on has one), and a made-up size is worse than no answer.
BYTES_PER_ELEMENT: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}


#: What llama.cpp uses when `-ub` is not given. Only ever applied where the launch
#: arguments were seen at all — an unobserved batch size is unknown, not this.
DEFAULT_UBATCH = 512


def swa_cells(shape: ModelShape, n_ctx: int, n_ubatch: int,
              seqs_sharing: int = 1) -> int:
    """How many tokens a sliding-window layer actually caches.

    Not the window. llama.cpp adds one physical batch of headroom, and where several
    sequences share one unified cache it makes room for a window each. Measured against
    a real server's own allocation on 2026-08-04 (see
    docs/ironclad/PROBE-2026-08-04-host-facts.md, Finding 5) across five configurations:

        window 1024, ubatch 512, 1 seq,  ctx 2048  -> 1536 cells
        window 1024, ubatch 256, 1 seq,  ctx 4096  -> 1280 cells
        window 1024, ubatch 512, 1 seq,  ctx 512   ->  512 cells   (capped by context)
        window 1024, ubatch 512, 4 seqs, ctx 8192  -> 4608 cells   (unified)

    A cache can never hold more than the context does, hence the cap.
    """
    return min(n_ctx, (shape.sliding_window or 0) * seqs_sharing + n_ubatch)


def _side_bytes(shape: ModelShape, n_ctx: int, head_dim_full: int,
                head_dim_swa: Optional[int], per_element: float,
                n_ubatch: int, seqs_sharing: int) -> float:
    """One side of the cache — all the keys, or all the values — summed over layers."""
    total = 0.0
    window_tokens = swa_cells(shape, n_ctx, n_ubatch, seqs_sharing)
    for layer in range(shape.block_count):
        sliding = shape.sliding_window_pattern[layer]
        if sliding and shape.sliding_window:
            tokens = window_tokens
            head_dim = head_dim_swa or head_dim_full
        else:
            tokens = n_ctx
            head_dim = head_dim_full
        total += shape.head_count_kv[layer] * head_dim * tokens * per_element
    return total


def has_sliding_window(shape: Optional[ModelShape]) -> bool:
    """Whether any layer is a sliding-window one, and so needs the batch size."""
    return bool(shape and shape.sliding_window and any(shape.sliding_window_pattern))


def kv_cache_bytes(shape: Optional[ModelShape], n_ctx: Optional[int],
                   cache_k: Optional[str], cache_v: Optional[str],
                   n_ubatch: Optional[int] = None,
                   seqs_sharing: int = 1) -> Optional[int]:
    """Size of one key/value cache in bytes, or None if it cannot be determined.

    Keys and values are summed separately because -ctk and -ctv are separate flags and
    may be set to different types.

    `n_ubatch` is required only for a model with sliding-window layers, whose size
    depends on it — see `swa_cells`. Passing None for such a model returns None rather
    than assuming a batch size, because the answer would be wrong by up to the whole
    window. A model without window layers ignores it entirely.
    """
    if shape is None or not n_ctx or n_ctx <= 0:
        return None
    k_per = BYTES_PER_ELEMENT.get((cache_k or "f16").lower())
    v_per = BYTES_PER_ELEMENT.get((cache_v or "f16").lower())
    if k_per is None or v_per is None:
        return None
    if len(shape.head_count_kv) != shape.block_count:
        return None
    if len(shape.sliding_window_pattern) != shape.block_count:
        return None
    if has_sliding_window(shape) and n_ubatch is None:
        return None

    ub = n_ubatch or 0
    keys = _side_bytes(shape, n_ctx, shape.key_length, shape.key_length_swa, k_per,
                       ub, seqs_sharing)
    values = _side_bytes(shape, n_ctx, shape.value_length, shape.value_length_swa, v_per,
                         ub, seqs_sharing)
    return int(keys + values)


def derivation(shape: Optional[ModelShape], n_ctx: Optional[int],
               cache_k: Optional[str], cache_v: Optional[str],
               n_ubatch: Optional[int] = None, seqs_sharing: int = 1) -> dict:
    """The inputs the figure came from, stored alongside it.

    D8 requires the estimate to travel with what produced it, so that a wrong answer can
    be diagnosed rather than merely disbelieved.
    """
    if shape is None:
        return {"known": False, "reason": "model shape unavailable"}
    return {
        "known": True,
        "architecture": shape.architecture,
        "block_count": shape.block_count,
        "head_count_kv": shape.head_count_kv,
        "key_length": shape.key_length,
        "value_length": shape.value_length,
        "key_length_swa": shape.key_length_swa,
        "value_length_swa": shape.value_length_swa,
        "sliding_window": shape.sliding_window,
        "sliding_window_layers": sum(shape.sliding_window_pattern),
        "n_ubatch": n_ubatch,
        "seqs_sharing_cache": seqs_sharing,
        "swa_cells": (swa_cells(shape, n_ctx, n_ubatch, seqs_sharing)
                      if (n_ctx and n_ubatch is not None and has_sliding_window(shape))
                      else None),
        "n_ctx": n_ctx,
        "cache_k": cache_k,
        "cache_v": cache_v,
        "bytes_per_element_k": BYTES_PER_ELEMENT.get((cache_k or "f16").lower()),
        "bytes_per_element_v": BYTES_PER_ELEMENT.get((cache_v or "f16").lower()),
        "method": ("per-layer sum; sliding-window layers cache "
                   "min(n_ctx, window x seqs_sharing + n_ubatch) tokens"),
    }
