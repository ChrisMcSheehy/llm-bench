"""What forks a host identity, and what deliberately does not."""
from __future__ import annotations

from llmbench.models import HostFingerprint


def _host(**overrides) -> HostFingerprint:
    base = dict(
        os="Linux", os_release="6.8.0", arch="x86_64", cpu_count=8,
        total_memory_bytes=33_454_276_608,
        devices=[{"id": "Vulkan0", "backend": "Vulkan", "name": "AMD Radeon RX 7900 XTX",
                  "total_mib": 24560, "free_mib": 23749}],
    )
    base.update(overrides)
    return HostFingerprint(**base)


def test_a_different_graphics_card_is_a_different_machine():
    other = [{"id": "Vulkan0", "backend": "Vulkan", "name": "AMD Radeon RX 6800",
              "total_mib": 16384, "free_mib": 16000}]
    assert _host().host_hash != _host(devices=other).host_hash


def test_a_different_compute_backend_is_a_different_machine():
    """The same card through Vulkan and through ROCm runs different implementations."""
    rocm = [{"id": "ROCm0", "backend": "ROCm", "name": "AMD Radeon RX 7900 XTX",
             "total_mib": 24560, "free_mib": 23749}]
    assert _host().host_hash != _host(devices=rocm).host_hash


def test_free_memory_never_forks_the_identity():
    """It differs between two runs on an idle machine. Hashing it would split history."""
    busy = [{"id": "Vulkan0", "backend": "Vulkan", "name": "AMD Radeon RX 7900 XTX",
             "total_mib": 24560, "free_mib": 2048}]
    assert _host().host_hash == _host(devices=busy).host_hash


def test_the_operating_system_release_never_forks_the_identity():
    """A point upgrade is not a new machine; the record still carries the version."""
    assert _host().host_hash == _host(os_release="6.9.1").host_hash


def test_the_operating_system_itself_does_fork_it():
    assert _host().host_hash != _host(os="Windows").host_hash


def test_a_memory_reading_that_moves_within_a_gibibyte_does_not_fork_it():
    """Firmware can reserve a little more or less. That is not a different machine."""
    assert _host().host_hash == _host(
        total_memory_bytes=33_454_276_608 - 200_000_000).host_hash


def test_a_real_memory_change_does_fork_it():
    assert _host().host_hash != _host(total_memory_bytes=64 * 1024 ** 3).host_hash


def test_a_declared_processor_name_is_recorded_but_not_hashed():
    assert _host().host_hash == _host(cpu_model="AMD Ryzen 7 7800X3D").host_hash


def test_the_label_names_the_card_and_the_backend():
    label = _host().label
    assert "7900 XTX" in label
    assert "Vulkan" in label


def test_a_machine_with_no_device_information_still_has_an_identity():
    """A server we merely connected to yields no devices; that is a machine too."""
    bare = _host(devices=[])
    assert bare.host_hash
    assert bare.host_hash != _host().host_hash, (
        "a machine whose cards are unknown is not the same as one whose cards are known")
    assert "no device information" in bare.label
