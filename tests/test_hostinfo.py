"""Machine facts, asserted as properties rather than as this machine's values.

A test that asserts 8 processors passes only on the machine that wrote it — see
LESSONS.md, a-captured-fixture-carries-paths-that-still-exist-at-home. These assert
what must be true of any machine the suite runs on, including three CI platforms.
"""
from __future__ import annotations

from llmbench.hostinfo import machine_facts, total_memory_bytes


def test_it_reports_the_platform_it_is_running_on():
    f = machine_facts()
    assert f["os"] in {"Windows", "Linux", "Darwin"}, f["os"]
    assert f["arch"], "architecture must never be empty"
    assert f["cpu_count"] >= 1


def test_total_memory_is_plausible_for_any_machine_that_can_run_this():
    total = total_memory_bytes()
    assert total is not None, "no total-memory reading on this platform"
    assert total > 512 * 1024 ** 2, "under 512 MiB cannot be right"
    assert total < 8 * 1024 ** 4, "over 8 TiB is not a machine this runs on"


def test_the_reading_is_stable_between_calls():
    """Total memory is a fact about the machine, not a reading that drifts.

    Free memory drifts, which is why D1 keeps it out of the hash and this is in it.
    """
    assert total_memory_bytes() == total_memory_bytes()


def test_facts_are_json_safe():
    """They are stored as JSON, so nothing exotic may appear in them."""
    import json
    json.dumps(machine_facts())
