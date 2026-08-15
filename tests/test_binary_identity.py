"""Which binary produced a result, as part of the identity.

The bench exists to compare builds of llama.cpp - forks, branches, pull requests. The
identity recorded the build commit that /props reports and nothing about the executable,
so two builds of the SAME commit were one configuration: a Vulkan build and a ROCm build
of b10144 would have been pooled together, silently averaging exactly the difference the
run was measuring.

Observed for real on 2026-08-05: this machine has a Vulkan llama.cpp on PATH while
Ollama was driving the same card through ROCm.

The binary is hashed rather than named. A path is not an identity - two people keep the
same build in different folders, and one person overwrites a path with a new build every
time they rebuild. Hashing is also observation rather than declaration, which is what
design D2a requires of anything that reaches the identity.
"""
from __future__ import annotations

import asyncio

import httpx

from llmbench.targets.base import DetectionError
from llmbench.targets.llamacpp import LlamaCppTarget

_PROPS = {
    "default_generation_settings": {"n_ctx": 32768, "params": {"temperature": 1.0}},
    "total_slots": 4,
    "model_path": "/models/Qwen3-8B-Q4_K_M.gguf",
    "build_info": "b10144-d73c1d6b2",
}


def _serve(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/props":
        return httpx.Response(200, json=_PROPS)
    raise DetectionError("no such endpoint")


def _detect(binary=None):
    async def go():
        t = LlamaCppTarget("http://x", binary=binary)
        t._client = httpx.AsyncClient(transport=httpx.MockTransport(_serve))
        try:
            return await t.detect()
        finally:
            await t.aclose()
    return asyncio.run(go())


def _binary(tmp_path, name: str, content: bytes):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def test_two_builds_of_the_same_commit_are_two_identities(tmp_path):
    """The regression, and the whole point. Same commit, same model, same settings."""
    vulkan = _detect(_binary(tmp_path, "vulkan.exe", b"VULKAN BUILD"))
    rocm = _detect(_binary(tmp_path, "rocm.exe", b"ROCM BUILD"))

    assert vulkan.build_commit == rocm.build_commit == "d73c1d6b2"
    assert vulkan.fingerprint_hash != rocm.fingerprint_hash, \
        "two different binaries reporting one commit collapsed into one configuration"


def test_the_same_binary_in_two_places_is_one_identity(tmp_path):
    """A path is not an identity: the same build copied elsewhere is the same build."""
    here = _binary(tmp_path, "here.exe", b"IDENTICAL BYTES")
    there = _binary(tmp_path, "there.exe", b"IDENTICAL BYTES")

    assert _detect(here).fingerprint_hash == _detect(there).fingerprint_hash


def test_the_hash_is_recorded_so_a_reader_can_check_it(tmp_path):
    fp = _detect(_binary(tmp_path, "a.exe", b"BUILD"))

    assert fp.binary_sha, "the binary hash is not recorded anywhere a reader can see"
    assert len(fp.binary_sha) == 16, "expected a short hash, like the other hashes here"


def test_a_server_we_did_not_launch_has_no_binary_and_still_detects():
    """Connecting to a server someone else started is the ordinary case, and the
    binary is then genuinely unknown rather than absent."""
    fp = _detect(binary=None)

    assert fp.binary_sha is None
    assert fp.n_ctx == 32768, "detection broke when there was no binary to hash"


def test_an_unknown_binary_does_not_collide_with_a_known_one(tmp_path):
    """Unknown is its own state, exactly as D6a made it for the launch settings."""
    known = _detect(_binary(tmp_path, "a.exe", b"BUILD"))
    unknown = _detect(binary=None)

    assert known.fingerprint_hash != unknown.fingerprint_hash


def test_a_path_that_does_not_exist_is_unknown_rather_than_an_error(tmp_path):
    """A profile pointing at a moved binary must not stop the run: the server is up and
    answering, and an unreadable file makes the build unknown, not the run broken."""
    fp = _detect(str(tmp_path / "gone.exe"))

    assert fp.binary_sha is None
    assert fp.n_ctx == 32768
