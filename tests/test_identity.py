"""What forks a fingerprint, and what deliberately does not."""
from __future__ import annotations

from llmbench.models import ModelFingerprint, parse_quant


def _fp(**overrides) -> ModelFingerprint:
    """A fixed baseline so each test varies exactly one thing."""
    base = dict(
        engine="llama.cpp", build_commit="abcdef1", base_url="http://localhost:8080",
        model_id="Qwen3-8B-Q4_K_M", quant="Q4_K_M", n_params="8B", n_ctx=16384,
        kv_cache_k="q8_0", kv_cache_v="q8_0", flash_attn="on",
        sampling={"temperature": 0.0}, chat_template_sha="deadbeef0001",
    )
    base.update(overrides)
    return ModelFingerprint(**base)


def test_the_gpu_layer_split_forks_the_identity():
    """Half the model on the card and all of it are different deployments.

    They differ hugely in speed, and because the card and the processor run
    separately-written implementations of the same mathematics, they can differ in
    output too. Filing both under one identity averages them together.
    """
    assert _fp(n_gpu_layers="40").fingerprint_hash != _fp(n_gpu_layers="99").fingerprint_hash


def test_an_unset_gpu_layer_count_is_not_the_same_as_a_set_one():
    assert _fp().fingerprint_hash != _fp(n_gpu_layers="99").fingerprint_hash


def test_batch_settings_fork_the_identity():
    baseline = _fp().fingerprint_hash
    assert _fp(n_batch=4096).fingerprint_hash != baseline
    assert _fp(n_ubatch=512).fingerprint_hash != baseline
    assert _fp(n_parallel=4).fingerprint_hash != baseline


def test_the_three_batch_settings_are_told_apart_from_each_other():
    """A hash built by concatenating values would collide these three."""
    hashes = {
        _fp(n_batch=512).fingerprint_hash,
        _fp(n_ubatch=512).fingerprint_hash,
        _fp(n_parallel=512).fingerprint_hash,
    }
    assert len(hashes) == 3, "two of the batch settings share an identity"


def test_the_server_address_still_does_not_fork_the_identity():
    """This must KEEP passing.

    The same setup on a different port is deliberately the same thing. If this breaks,
    the change over-fired and every port change now looks like a new configuration.
    """
    a = _fp(base_url="http://localhost:8080")
    b = _fp(base_url="http://192.168.1.4:9090")
    assert a.fingerprint_hash == b.fingerprint_hash


def test_the_label_shows_what_forked_the_identity():
    """Two rows with the same text and different numbers are unreadable."""
    partial = _fp(n_gpu_layers="40").label
    full = _fp(n_gpu_layers="99").label
    assert partial != full, f"both rows read {partial!r}"
    assert "40" in partial and "99" in full


def test_the_label_shows_batch_settings_when_they_were_set():
    label = _fp(n_batch=4096, n_ubatch=512, n_parallel=4).label
    assert "b:4096" in label
    assert "ub:512" in label
    assert "np:4" in label


def test_the_label_stays_quiet_about_settings_nobody_set():
    """Most launches set none of these. Their labels must not grow noise."""
    label = _fp().label
    for noise in ("ngl", "b:", "ub:", "np:"):
        assert noise not in label, f"{noise!r} appeared in {label!r}"


def test_the_memory_estimate_does_not_change_the_identity():
    """A derived figure must not fork the history.

    Two runs of the same configuration must stay comparable even if a later formula
    improvement changes the estimate between them.
    """
    a = _fp(kv_cache_bytes=1_000_000)
    b = _fp(kv_cache_bytes=2_000_000)
    assert a.fingerprint_hash == b.fingerprint_hash


def test_an_unknown_estimate_is_none_and_not_zero():
    """Zero would read as 'this configuration costs no memory'."""
    assert _fp().kv_cache_bytes is None


# ---- the quantisation scheme, not just its nominal size ----------------------
#
# Found on 2026-08-15 while reading Unsloth's documentation. Their Dynamic quants choose
# the type per layer, so `UD-Q4_K_M` and a stock `Q4_K_M` are the same nominal size and
# very nearly the same file size while being different programs - their own measurements
# put Dynamic "Q4" near uniform Q5 for perplexity. `parse_quant` matched the plain token
# inside the longer name and returned `Q4_K_M` for both, which is the one field this
# bench exists to compare.

def test_a_dynamic_quant_keeps_its_prefix():
    assert parse_quant("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf") == "UD-Q4_K_M"
    assert parse_quant("Qwen3-30B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert parse_quant("model-UD-IQ2_M.gguf") == "UD-IQ2_M"


def test_a_stock_quant_gains_no_prefix():
    """The success condition. A change that labelled everything UD- would also stop the
    collision, and would be wrong about every other model on disk."""
    assert parse_quant("Qwen3.6-35B-A3B-Q4_K_M.gguf") == "Q4_K_M"
    assert parse_quant("Qwen3-30B-IQ4_XS.gguf") == "IQ4_XS"
    assert parse_quant("gpt-oss-20b-MXFP4.gguf") == "MXFP4"


def test_a_word_that_merely_ends_in_ud_is_not_a_dynamic_quant():
    """`cloud-q4_k_m` is a stock quant in a folder with an unfortunate name."""
    assert parse_quant("cloud-q4_k_m.gguf") == "Q4_K_M"
    assert parse_quant("my-STUD-Q4_K_M.gguf") == "Q4_K_M"


def test_a_dynamic_quant_and_a_stock_quant_are_two_configurations():
    """The point of the parsing above: they must not pool into one leaderboard row.

    They differ in what the file *is*, so averaging their scores together would hide the
    comparison the bench was pointed at.
    """
    assert (_fp(quant="UD-Q4_K_M").fingerprint_hash
            != _fp(quant="Q4_K_M").fingerprint_hash)


def test_the_label_says_which_quantisation_scheme_it_was():
    """Two rows both reading 'Q4_K_M' is the reader-facing half of the same defect."""
    assert "UD-Q4_K_M" in _fp(quant="UD-Q4_K_M").label
