"""What forks a fingerprint, and what deliberately does not."""
from __future__ import annotations

from llmbench.models import ModelFingerprint


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
