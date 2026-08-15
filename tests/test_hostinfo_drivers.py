"""Graphics driver versions: recorded, joined to devices, and never hashed.

D1 wants the driver version recorded and explicitly *not* hashed, because it changes
often enough that hashing it would fragment a machine's history on every update — while
setting a tripwire for the day a driver update is seen to move results. Nothing in
llama.cpp reports it: the device list gives model and memory only, and even `-lv 10`
adds nothing (verified 2026-08-04, PROBE-2026-08-04-host-facts.md).

So it is probed per platform, best effort, and absent rather than guessed.
"""
from __future__ import annotations

import pathlib

from llmbench.hostinfo import (
    attach_drivers, parse_nvidia_smi, parse_windows_controllers,
)

_WINDOWS_CSV = (pathlib.Path(__file__).parent / "fixtures" /
                "video_controllers_windows.csv").read_text(encoding="utf-8")


def test_windows_controllers_are_read_from_real_output():
    got = parse_windows_controllers(_WINDOWS_CSV)
    assert got == {
        "AMD Radeon RX 7900 XTX": "32.0.31007.5012",
        "AMD Radeon(TM) Graphics": "32.0.21043.10005",
    }


def test_a_name_containing_a_comma_would_not_split_the_row():
    """The output is CSV, so it is parsed as CSV rather than split on commas."""
    csv = '"Name","DriverVersion"\n"Acme GPU, Special Edition","1.2.3"\n'
    assert parse_windows_controllers(csv) == {"Acme GPU, Special Edition": "1.2.3"}


def test_unparseable_output_yields_nothing_rather_than_raising():
    assert parse_windows_controllers("") == {}
    assert parse_windows_controllers("not csv at all") == {}


def test_nvidia_smi_output_is_read():
    """`nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`."""
    out = "NVIDIA GeForce RTX 4090, 550.54.14\nNVIDIA GeForce RTX 3090, 550.54.14\n"
    assert parse_nvidia_smi(out) == {
        "NVIDIA GeForce RTX 4090": "550.54.14",
        "NVIDIA GeForce RTX 3090": "550.54.14",
    }


def test_nvidia_smi_failure_text_yields_nothing():
    assert parse_nvidia_smi("command not found") == {}
    assert parse_nvidia_smi("") == {}


def test_drivers_attach_to_the_device_they_belong_to():
    devices = [
        {"id": "Vulkan0", "backend": "Vulkan", "name": "AMD Radeon RX 7900 XTX",
         "total_mib": 24560, "free_mib": 23749},
        {"id": "Vulkan1", "backend": "Vulkan", "name": "AMD Radeon(TM) Graphics",
         "total_mib": 16208, "free_mib": 15397},
    ]
    attached = attach_drivers(devices, parse_windows_controllers(_WINDOWS_CSV))
    assert attached[0]["driver"] == "32.0.31007.5012"
    assert attached[1]["driver"] == "32.0.21043.10005"


def test_a_device_with_no_matching_driver_says_nothing_rather_than_guessing():
    devices = [{"id": "Vulkan0", "backend": "Vulkan", "name": "Some Unlisted GPU",
                "total_mib": 8192, "free_mib": 8000}]
    attached = attach_drivers(devices, {"A Different GPU": "1.0"})
    assert attached[0].get("driver") is None


def test_attaching_does_not_mutate_the_devices_it_was_given():
    """The caller's list is evidence of what the binary reported; leave it alone."""
    devices = [{"id": "Vulkan0", "backend": "Vulkan", "name": "AMD Radeon RX 7900 XTX",
                "total_mib": 24560, "free_mib": 23749}]
    attach_drivers(devices, parse_windows_controllers(_WINDOWS_CSV))
    assert "driver" not in devices[0]


def test_a_driver_update_does_not_change_the_machine_identity():
    """D1's rule: hashing the driver would split a machine's history on every update."""
    from llmbench.models import HostFingerprint

    def host(driver):
        return HostFingerprint(
            os="Linux", arch="x86_64", cpu_count=8,
            devices=[{"id": "CUDA0", "backend": "CUDA", "name": "RTX 4090",
                      "total_mib": 24564, "free_mib": 24000, "driver": driver}])

    assert host("550.54.14").host_hash == host("560.00.01").host_hash
