"""KV-cache sizing, including the case where the obvious formula is 41x wrong."""
from __future__ import annotations

import pytest

from llmbench.gguf import ModelShape
from llmbench.memory import BYTES_PER_ELEMENT, kv_cache_bytes


def _dense(layers=32, heads_kv=8, head_dim=128) -> ModelShape:
    return ModelShape(
        architecture="llama", block_count=layers, head_count=32,
        head_count_kv=[heads_kv] * layers, key_length=head_dim, value_length=head_dim,
        sliding_window_pattern=[False] * layers,
    )


def _gemma4_12b() -> ModelShape:
    """The real shape read from gemma-4-12B-it-heretic-Q8_0.gguf on 2026-08-04.

    40 sliding-window layers with 8 KV heads at head dimension 256, and 8
    full-attention layers with 1 KV head at head dimension 512.
    """
    pattern, heads = [], []
    for i in range(48):
        swa = (i % 6) != 5
        pattern.append(swa)
        heads.append(8 if swa else 1)
    return ModelShape(
        architecture="gemma4", block_count=48, head_count=16, head_count_kv=heads,
        key_length=512, value_length=512, sliding_window_pattern=pattern,
        sliding_window=1024, key_length_swa=256, value_length_swa=256,
    )


def test_a_dense_model_is_the_plain_product():
    """32 layers x 8 kv heads x 128 dim x 2 (K and V) x 2 bytes x 4096 tokens."""
    got = kv_cache_bytes(_dense(), n_ctx=4096, cache_k="f16", cache_v="f16")
    assert got == 32 * 8 * 128 * 2 * 2 * 4096


def test_the_sliding_window_model_is_not_the_plain_product():
    """The 39x case. A single product would return 96 GiB; the answer is 2.47 GiB.

    The figure was 2.31 GiB until 2026-08-04, when a real server's own allocation showed
    that a window layer caches window + ubatch tokens rather than the window.
    """
    got = kv_cache_bytes(_gemma4_12b(), n_ctx=131072, cache_k="f16", cache_v="f16",
                         n_ubatch=512)
    assert got == 2_650_800_128, f"expected the per-layer sum, got {got}"

    naive = 48 * 8 * 512 * 2 * 2 * 131072
    assert naive / got > 38, "the test model no longer exercises the trap"


def test_a_window_larger_than_the_context_does_not_inflate_the_answer():
    """A 1024-token window cannot cache more than a 512-token context holds."""
    small = kv_cache_bytes(_gemma4_12b(), n_ctx=512, cache_k="f16", cache_v="f16",
                           n_ubatch=512)
    per_swa_layer = 8 * 256 * 2 * 2 * 512
    per_full_layer = 1 * 512 * 2 * 2 * 512
    assert small == 40 * per_swa_layer + 8 * per_full_layer


def test_compressing_the_cache_reduces_the_figure():
    """The trade-off the tool exists to measure."""
    f16 = kv_cache_bytes(_dense(), n_ctx=4096, cache_k="f16", cache_v="f16")
    q8 = kv_cache_bytes(_dense(), n_ctx=4096, cache_k="q8_0", cache_v="q8_0")
    assert q8 < f16
    assert q8 == pytest.approx(f16 * 1.0625 / 2.0)


def test_keys_and_values_can_use_different_types():
    """-ctk and -ctv are separate flags, so they are summed separately."""
    mixed = kv_cache_bytes(_dense(), n_ctx=4096, cache_k="q8_0", cache_v="f16")
    both_k = kv_cache_bytes(_dense(), n_ctx=4096, cache_k="q8_0", cache_v="q8_0")
    both_v = kv_cache_bytes(_dense(), n_ctx=4096, cache_k="f16", cache_v="f16")
    assert mixed == (both_k + both_v) // 2


@pytest.mark.parametrize("cache_type", sorted(BYTES_PER_ELEMENT))
def test_every_documented_cache_type_produces_a_number(cache_type):
    assert kv_cache_bytes(_dense(), n_ctx=1024,
                          cache_k=cache_type, cache_v=cache_type) > 0


def test_an_unrecognised_cache_type_is_unknown_not_a_guess():
    """A fork may add cache types. Inventing a size for one is the forbidden failure."""
    assert kv_cache_bytes(_dense(), n_ctx=4096, cache_k="q1_0", cache_v="f16") is None


def test_no_shape_is_unknown():
    assert kv_cache_bytes(None, n_ctx=4096, cache_k="f16", cache_v="f16") is None


def test_no_context_length_is_unknown():
    """Without a context length there is nothing to multiply by."""
    assert kv_cache_bytes(_dense(), n_ctx=None, cache_k="f16", cache_v="f16") is None


# --- checked against a real server's own allocation ---------------------------------
#
# Every other test here compares the arithmetic against its own inputs. These compare it
# against what llama-server said it allocated, which is the only check that could have
# caught the two errors this file previously shipped. Figures captured 2026-08-04 from
# gemma-4-12B-it-heretic-Q8_0 on build b10148; see
# docs/ironclad/PROBE-2026-08-04-host-facts.md, Finding 5.

MIB = 1024 ** 2


def test_it_matches_what_the_server_allocated_for_a_split_cache():
    """`-c 8192 -np 2`: two caches of 4096, windows padded by one 512-token batch.

    Server said: 128.00 MiB (4096 cells, 8 layers, 2/2 seqs)
             and 960.00 MiB (1536 cells, 40 layers, 2/2 seqs).
    """
    per_pool = kv_cache_bytes(_gemma4_12b(), n_ctx=4096, cache_k="f16", cache_v="f16",
                              n_ubatch=512, seqs_sharing=1)
    assert per_pool * 2 == 1088 * MIB


def test_it_matches_what_the_server_allocated_for_a_unified_cache():
    """`-c 8192` with slots left automatic: one cache, four sequences sharing it.

    Server said: 128.00 MiB (8192 cells, 8 layers, 4/1 seqs)
             and 1440.00 MiB (4608 cells, 40 layers, 4/1 seqs) - the window layers
    make room for four windows plus a batch, which is why seqs_sharing matters.
    """
    got = kv_cache_bytes(_gemma4_12b(), n_ctx=8192, cache_k="f16", cache_v="f16",
                         n_ubatch=512, seqs_sharing=4)
    assert got == 1568 * MIB


def test_a_context_shorter_than_the_window_caps_at_the_context():
    """`-c 512`: the server allocated 512 cells for window layers, not 1536."""
    from llmbench.memory import swa_cells
    assert swa_cells(_gemma4_12b(), n_ctx=512, n_ubatch=512) == 512
    assert swa_cells(_gemma4_12b(), n_ctx=2048, n_ubatch=512) == 1536
    assert swa_cells(_gemma4_12b(), n_ctx=4096, n_ubatch=256) == 1280
    assert swa_cells(_gemma4_12b(), n_ctx=8192, n_ubatch=512, seqs_sharing=4) == 4608


def test_a_window_model_without_a_batch_size_is_unknown():
    """The batch size is a launch argument no endpoint reports, and it moves the answer.

    Assuming the default here would be wrong by up to a whole window per layer, so a
    model that needs it and has not been given it gets no number at all.
    """
    assert kv_cache_bytes(_gemma4_12b(), n_ctx=4096, cache_k="f16",
                          cache_v="f16") is None


def test_a_dense_model_needs_no_batch_size():
    """No window layers, so the batch size cannot affect the answer and is not required."""
    assert kv_cache_bytes(_dense(), n_ctx=4096, cache_k="f16", cache_v="f16") is not None


def test_it_matches_the_server_for_a_compressed_cache():
    """The compression table was read from ggml's block structs and never confronted.

    Only the f16 path had ever been checked against a real allocation, so q8_0 was
    arithmetic agreeing with itself - the exact shape of the error this file already
    shipped once. Captured 2026-08-04 with `-c 8192 -np 1 -ctk q8_0 -ctv q8_0 -fa on`:
    68.00 MiB non-SWA + 255.00 MiB SWA.
    """
    got = kv_cache_bytes(_gemma4_12b(), n_ctx=8192, cache_k="q8_0", cache_v="q8_0",
                         n_ubatch=512, seqs_sharing=1)
    assert got == 323 * MIB


def test_it_matches_the_server_when_keys_and_values_differ():
    """-ctk q8_0 -ctv f16, summed separately: 98.00 MiB + 367.50 MiB.

    The half-MiB total is what makes this worth keeping - it cannot agree by rounding.
    """
    got = kv_cache_bytes(_gemma4_12b(), n_ctx=8192, cache_k="q8_0", cache_v="f16",
                         n_ubatch=512, seqs_sharing=1)
    assert got == int(465.5 * MIB)


def test_the_uncompressed_baseline_for_the_same_configuration():
    """608.00 MiB, so the three figures above are directly comparable."""
    got = kv_cache_bytes(_gemma4_12b(), n_ctx=8192, cache_k="f16", cache_v="f16",
                         n_ubatch=512, seqs_sharing=1)
    assert got == 608 * MIB
