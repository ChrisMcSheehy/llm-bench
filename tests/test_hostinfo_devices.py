"""Parsing the device list the model server's own binary prints.

The fixture is real captured output from llama-server b10148-ddfc2288e
(2026-08-04), confirmed byte for byte against a live `--list-devices` run. A
hand-written one would only prove the parser matches its author's imagination —
LESSONS.md, unit-tests-either-side-of-a-seam-do-not-test-the-seam.
"""
from __future__ import annotations

import pathlib

from llmbench.hostinfo import parse_devices

_CAPTURED = (pathlib.Path(__file__).parent / "fixtures" /
             "list_devices_vulkan.txt").read_text(encoding="utf-8")


def test_both_cards_are_read_from_real_output():
    devices = parse_devices(_CAPTURED)
    assert len(devices) == 2
    first = devices[0]
    assert first["id"] == "Vulkan0"
    assert first["backend"] == "Vulkan"
    assert first["name"] == "AMD Radeon RX 7900 XTX"
    assert first["total_mib"] == 24560
    assert first["free_mib"] == 23749


def test_a_name_containing_brackets_survives():
    """'AMD Radeon(TM) Graphics' — the parser must not stop at the first bracket."""
    assert parse_devices(_CAPTURED)[1]["name"] == "AMD Radeon(TM) Graphics"


def test_output_with_no_devices_is_an_empty_list_not_an_error():
    assert parse_devices("Available devices:\n") == []


def test_unrecognised_output_is_an_empty_list():
    """A future build may print something else. That is unknown, not a crash."""
    assert parse_devices("ggml_vulkan: no devices found\n") == []


def test_the_free_column_is_kept_separate_from_the_total():
    """D1 hashes total memory and never free memory, so they must not be confused."""
    d = parse_devices(_CAPTURED)[0]
    assert d["total_mib"] != d["free_mib"]


def test_no_binary_means_no_devices_rather_than_an_error():
    """A server we merely connected to has no binary to ask; that is unknown."""
    from llmbench.hostinfo import devices
    assert devices(None) == []


def test_a_binary_that_is_not_there_is_unknown_rather_than_a_crash(tmp_path):
    from llmbench.hostinfo import devices
    assert devices(str(tmp_path / "no-such-binary")) == []


# --- backends other than the one this machine has -----------------------------------
#
# The line format is authoritative: `"  %s: %s (%zu MiB, %zu MiB free)\n"` with the
# first field from ggml_backend_dev_name() and the second from
# ggml_backend_dev_description() (read from llama.cpp's common_print_available_devices
# on 2026-08-04). The CUDA line below is real output quoted in llama.cpp's own issue
# tracker; the ROCm, Metal and SYCL lines are *constructed from that verified format*,
# because this machine has only Vulkan and no capture of those was available. They test
# the parser's handling of shape, never of vendor spelling — which is the point: the
# parser must not know a list of backend names at all.

_MULTI = (pathlib.Path(__file__).parent / "fixtures" /
          "list_devices_multi.txt").read_text(encoding="utf-8")


def test_every_backend_is_read_without_the_parser_knowing_their_names():
    devices = parse_devices(_MULTI)
    assert [d["id"] for d in devices] == ["CUDA0", "ROCm0", "Metal", "SYCL0"]
    assert [d["backend"] for d in devices] == ["CUDA", "ROCm", "Metal", "SYCL"]


def test_a_single_device_backend_with_no_index_is_not_dropped():
    """Metal names its device 'Metal', with no trailing number.

    A parser that required one would silently return no devices on every Mac, and a
    machine with no devices hashes differently from the same machine with them - so
    this is a wrong identity, not a missing field.
    """
    metal = [d for d in parse_devices(_MULTI) if d["id"] == "Metal"]
    assert len(metal) == 1
    assert metal[0]["backend"] == "Metal"
    assert metal[0]["name"] == "Apple M3 Max"
    assert metal[0]["total_mib"] == 32768


def test_the_description_may_contain_anything_including_digits():
    d = parse_devices(_MULTI)[0]
    assert d["name"] == "NVIDIA GeForce RTX 4060 Ti"
