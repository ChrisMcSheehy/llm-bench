"""What machine is this, in facts that cost nothing to obtain.

Two sources, per design decision D2 as amended by D2a: the standard library for the
platform and its memory, and the model server's own binary for the graphics cards
(see `devices` below). The HTTP API is deliberately not one of them — no llama.cpp
endpoint reports any hardware fact at all, verified against a running server on
2026-08-04 (docs/ironclad/PROBE-2026-08-04-host-facts.md, Finding 1).

Nothing here raises. A fact that cannot be read is None, because "unknown" is an
honest answer and a wrong number is not.
"""
from __future__ import annotations

import csv
import ctypes
import io
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


def total_memory_bytes() -> Optional[int]:
    """Total physical memory, or None where neither method is available.

    Unix exposes it through sysconf. Windows has no sysconf, so this calls
    GlobalMemoryStatusEx through ctypes — still the standard library, still no
    dependency.
    """
    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in getattr(os, "sysconf_names", {}):
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError):
            return None
    if sys.platform == "win32":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        try:
            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.ullTotalPhys)
        except (OSError, AttributeError):
            return None
    return None


def machine_facts() -> dict[str, Any]:
    """The platform facts, free from the standard library.

    `platform.processor()` is deliberately absent: it returns a family/stepping string
    on Windows and frequently an empty string on Linux, so it is not a fact this can
    rely on. The processor *model* is a user declaration instead (D2a, decision 4).
    """
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "total_memory_bytes": total_memory_bytes(),
    }


#   "  Vulkan0: AMD Radeon RX 7900 XTX (24560 MiB, 23749 MiB free)"
#
# This matches llama.cpp's format rather than a list of backend names:
#
#     "  %s: %s (%zu MiB, %zu MiB free)\n"
#
# with the first field from ggml_backend_dev_name() and the second from
# ggml_backend_dev_description() (read from common_print_available_devices on
# 2026-08-04). The identifier is therefore whatever the backend called itself, and
# knowing the vocabulary is neither possible nor necessary: CUDA0, ROCm0, SYCL0 and a
# bare "Metal" all parse the same way. Requiring a trailing digit dropped every
# single-device backend silently, which is a wrong machine identity rather than a
# missing field.
#
# The description is taken lazily up to the final bracketed pair, so a name containing
# brackets - "AMD Radeon(TM) Graphics" - survives intact.
_DEVICE = re.compile(
    r"^\s*[-\s]*(?P<id>\S+?)\s*:\s+"
    r"(?P<name>.+?)\s+\((?P<total>\d+)\s*MiB,\s*(?P<free>\d+)\s*MiB free\)\s*$")

#: Trailing device index, e.g. the "0" of "Vulkan0". Absent on single-device backends.
_INDEX = re.compile(r"\d+$")


def parse_devices(text: str) -> list[dict[str, Any]]:
    """Every device in `--list-devices` output. Unrecognised text yields no devices."""
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = _DEVICE.match(line)
        if m:
            ident = m.group("id")
            out.append({
                "id": ident,
                "backend": _INDEX.sub("", ident) or ident,
                "name": m.group("name"),
                "total_mib": int(m.group("total")),
                "free_mib": int(m.group("free")),
            })
    return out


def devices(binary: Optional[str], timeout: float = 20.0) -> list[dict[str, Any]]:
    """Ask a llama.cpp binary what devices it can see.

    Returns an empty list when there is no binary to ask - which is the case whenever
    the user pointed llmbench at an address instead of letting it start the server.
    That is unknown, not zero devices, and the caller records the difference.

    The child inherits this process's environment unmodified. Do not narrow it: a
    scrubbed environment already cost this project a day when a child died on Windows
    for want of SYSTEMROOT (LESSONS.md,
    scrubbed-subprocess-env-needs-systemroot-on-windows).
    """
    if not binary:
        return []
    try:
        proc = subprocess.run([binary, "--list-devices"], capture_output=True,
                              text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return []
    # Which stream carries the list is not guaranteed across builds, and reading both
    # costs nothing.
    return parse_devices((proc.stdout or "") + (proc.stderr or ""))


def declared_path() -> Path:
    """Where the user's own statements about this machine live.

    Overridable with LLMBENCH_HOST_DECLARED, matching how LLMBENCH_DB and
    LLMBENCH_SERVERS already work.
    """
    override = os.environ.get("LLMBENCH_HOST_DECLARED")
    if override:
        return Path(override)
    return Path.home() / ".llmbench" / "host.json"


def load_declared() -> dict[str, Any]:
    """The user's declarations, or an empty dict. A missing file is not an error."""
    try:
        loaded = json.loads(declared_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def save_declared(values: dict[str, Any]) -> None:
    """Add to the declarations, keeping the ones already there."""
    path = declared_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = load_declared()
    merged.update({k: v for k, v in values.items() if v is not None})
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")


# ---- graphics driver versions ------------------------------------------------------
#
# D1 records the driver version and deliberately does not hash it: it changes often
# enough that hashing would split a machine's history on every update, while recording
# it lets an unexplained discrepancy be traced. That decision sets a tripwire - if a
# driver update is ever seen to move results beyond ordinary variation, the choice was
# wrong - and the tripwire needs this data to exist.
#
# No backend reports it. llama.cpp's device list carries model and memory only, and
# raising its verbosity adds nothing (verified 2026-08-04). So it is probed per
# platform, best effort: anything that fails is absent rather than guessed.


def parse_windows_controllers(csv_text: str) -> dict[str, str]:
    """Device name -> driver version, from Get-CimInstance Win32_VideoController CSV.

    Parsed as CSV rather than split on commas, because a device name may contain one.
    """
    try:
        rows = list(csv.DictReader(io.StringIO(csv_text)))
    except csv.Error:
        return {}
    return {r["Name"]: r["DriverVersion"] for r in rows
            if r.get("Name") and r.get("DriverVersion")}


def parse_nvidia_smi(text: str) -> dict[str, str]:
    """Device name -> driver version, from nvidia-smi's two-column CSV."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        name, sep, version = line.partition(",")
        if sep and name.strip() and version.strip():
            out[name.strip()] = version.strip()
    return out


def _run(cmd: list[str], timeout: float = 15.0) -> str:
    """Run a probe, returning its output or an empty string if it did not work."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout or ""


def driver_versions() -> dict[str, str]:
    """Graphics driver version per device name, as far as this platform will say.

    macOS is deliberately absent: its graphics driver ships with the operating system,
    whose version is already recorded as `os_release`, so probing would add a slow
    subprocess and no information.
    """
    if sys.platform == "win32":
        return parse_windows_controllers(_run([
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,DriverVersion | ConvertTo-Csv -NoTypeInformation"]))
    if sys.platform.startswith("linux"):
        found = parse_nvidia_smi(_run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"]))
        # The AMD kernel module reports one version for every card it drives, and the
        # file is absent on machines without it.
        try:
            amdgpu = Path("/sys/module/amdgpu/version").read_text().strip()
        except OSError:
            amdgpu = ""
        if amdgpu:
            found.setdefault("amdgpu", amdgpu)
        return found
    return {}


def attach_drivers(devices: list[dict[str, Any]],
                   drivers: dict[str, str]) -> list[dict[str, Any]]:
    """Copy of `devices` with a driver version on each one that has a known match.

    Matched on the device description, which is what both sources name it by. A device
    with no match carries no driver key at all rather than a placeholder, because
    "unknown" and "none" are different answers.

    Returns a copy: the caller's list records what the binary reported, and is left as
    it was.
    """
    out = []
    for d in devices:
        copy = dict(d)
        version = drivers.get(d.get("name", ""))
        if version:
            copy["driver"] = version
        out.append(copy)
    return out
