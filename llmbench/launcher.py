"""Starting and stopping model servers from a saved profile.

llama-server does not report the arguments it was started with unless it is run in
router mode (verified against a running server — see
docs/ironclad/PROBE-2026-08-04-model-shape.md). That leaves the bench unable to tell a
model with half its layers on the graphics card from one with all of them, because it
cannot see the setting that differs.

A bench that starts the server knows the arguments, because it supplied them. This module
is the only place in the project that knows about operating-system processes.

Security note: the profiles file is an allowlist. The dashboard may ask for a profile *by
name*; it must never be able to supply a binary path or an argument list, because that
would let a web page run an arbitrary program. See docs/ironclad/DESIGN-launcher.md, L1.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


class LaunchError(RuntimeError):
    """A server could not be started. The message carries the server's own output."""


@dataclass
class Profile:
    """One named way to start a model server.

    The file is a mapping of name to profile, and one entry looks like this::

        servers:
          vulkan-b10441:
            binary: C:/builds/llama-b10441/llama-server.exe
            model:  C:/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
            args:   ["-ngl", "99", "-c", "65536", "-fa", "on"]
            port:   8123          # optional; omitted means a free port is chosen

    `binary` and `model` are required and `args` is passed through verbatim, in order,
    because order is meaningful to llama.cpp and a later flag overrides an earlier one.
    Everything in `args` becomes part of the run's identity, which is the whole reason
    this module exists.

    **Every profile spells out its own arguments, and that is deliberate for now.** A
    profile is the literal command line, so what reaches the fingerprint is a resolved
    argument list rather than a template that has to be reasoned about. Shared defaults
    and `{path}` interpolation are designed (`DESIGN-benchmark-coverage.md`, B7) and not
    built; until they are, a large profile set is better generated than hand-maintained.
    See the README for a worked set covering the comparisons this bench is for.
    """

    name: str
    binary: str
    model: str
    args: list[str] = field(default_factory=list)
    port: Optional[int] = None


@dataclass
class RunningServer:
    profile: Profile
    port: int
    process: subprocess.Popen
    log_path: Path

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def launch_args(self) -> list[str]:
        """The arguments this server was actually started with, minus the binary.

        This is what makes the identity complete: the bench knows these because it
        supplied them, where a server it merely connected to would not report them.
        """
        return list(self.profile.args)


def profiles_path() -> Path:
    """Where profiles live unless told otherwise.

    Beside the results database, and overridable the same way, so a user who moves one
    can move the other with the same habit.
    """
    override = os.environ.get("LLMBENCH_SERVERS")
    if override:
        return Path(override)
    return Path.home() / ".llmbench" / "servers.yaml"


def load_profiles(path: Optional[str] = None) -> dict[str, Profile]:
    """Read the profiles file. A file that does not exist means no profiles, not an error.

    Most users have never created one, and starting the dashboard should not fail because
    of it.
    """
    p = Path(path) if path else profiles_path()
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    servers = data.get("servers") or {}
    if not isinstance(servers, dict):
        raise ValueError(f"{p}: 'servers' must be a mapping of name to profile")

    out: dict[str, Profile] = {}
    for name, body in servers.items():
        if not isinstance(body, dict):
            raise ValueError(f"{p}: profile {name!r} is not a mapping")
        for required in ("binary", "model"):
            if not body.get(required):
                raise ValueError(f"{p}: profile {name!r} has no {required!r}")
        args = body.get("args") or []
        if not isinstance(args, list):
            # Splitting a string on spaces would mangle any quoted path, and a path with
            # a space in it is the normal case on Windows.
            raise ValueError(
                f"{p}: profile {name!r} has 'args' as {type(args).__name__}; "
                "it must be a list, e.g. [\"-ngl\", \"99\"]")
        out[name] = Profile(
            name=name,
            binary=str(body["binary"]),
            model=str(body["model"]),
            args=[str(a) for a in args],
            port=int(body["port"]) if body.get("port") else None,
        )
    return out


def free_port() -> int:
    """A port nothing is listening on right now.

    Bind to port 0, let the operating system choose, then release it. There is a race
    between releasing and the server binding, which is why the caller reports a bind
    failure rather than silently retrying: a confusing "port in use" is better than a
    server that quietly started somewhere else.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _is_ready(port: int, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=timeout):
            return True
    except urllib.error.HTTPError as exc:
        # 503 is what llama-server answers while it is still loading a model: the HTTP
        # listener is bound long before the weights are in memory. Treating that as
        # ready is what made `llmbench launch` return early, so detection then ran
        # against a server that could not describe itself yet - and until the detection
        # path was fixed on 2026-08-05 it quietly filed results under an identity built
        # from defaults.
        #
        # Any other status still counts as ready: a 404 or a 401 means the server is up
        # and merely dislikes this particular request, which is a different thing from
        # being unable to serve at all.
        return exc.code != 503
    except Exception:
        return False


def _log_tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace")
                         .splitlines()[-lines:])
    except OSError:
        return "(no output captured)"


def build_argv(profile: Profile, port: int) -> list[str]:
    """The full command line, with the profile's own arguments last so they win."""
    return [profile.binary, "-m", profile.model,
            "--host", "127.0.0.1", "--port", str(port), *profile.args]


def start(profile: Profile, log_dir: Optional[str] = None,
          timeout: float = 300.0) -> RunningServer:
    """Start a server and return only once it answers.

    Raises LaunchError carrying the server's own output. Two of the three model loads
    attempted while designing this failed — an unsupported architecture and a context
    larger than the model was trained for — and in both cases the only useful information
    was in that output.
    """
    port = profile.port or free_port()
    directory = Path(log_dir) if log_dir else Path.home() / ".llmbench" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / f"{profile.name}-{port}.log"

    argv = build_argv(profile, port)
    log = open(log_path, "w", encoding="utf-8")
    try:
        # The environment is passed through unmodified. A scrubbed environment previously
        # broke a child process on Windows in this project (see LESSONS.md,
        # scrubbed-subprocess-env-needs-systemroot-on-windows), and a model server
        # legitimately needs library paths and device variables.
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    except OSError as exc:
        log.close()
        raise LaunchError(f"could not run {profile.binary!r}: {exc}") from exc

    server = RunningServer(profile=profile, port=port, process=process,
                           log_path=log_path)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log.close()
            raise LaunchError(
                f"{profile.name}: server exited with code {process.returncode} "
                f"before it was ready.\n--- its output ---\n{_log_tail(log_path)}")
        if _is_ready(port):
            log.close()
            return server
        time.sleep(0.25)

    stop(server)
    log.close()
    raise LaunchError(
        f"{profile.name}: no answer on port {port} within {timeout:.0f}s.\n"
        f"--- its output ---\n{_log_tail(log_path)}")


def stop(server: RunningServer, timeout: float = 20.0) -> None:
    """End the server, and do not return while it is still running.

    Terminate first so the server can close its files, then kill if it will not go. An
    orphaned model server holds gigabytes of graphics memory and occupies the port the
    next run wants.
    """
    process = server.process
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def registry_path() -> Path:
    """Where servers started by `llmbench launch` are recorded."""
    return profiles_path().parent / "running.json"


def _read_registry() -> dict[str, dict[str, Any]]:
    p = registry_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_registry(data: dict[str, dict[str, Any]]) -> None:
    p = registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def record_running(server: RunningServer) -> None:
    """Remember a server so a later, separate process can stop it.

    `llmbench launch` exits while the server keeps running, so `llmbench stop` has no
    handle to it — only what was written down here.
    """
    data = _read_registry()
    data[server.profile.name] = {
        "pid": server.pid, "port": server.port,
        "base_url": server.base_url, "log": str(server.log_path),
    }
    _write_registry(data)


def forget_running(name: str) -> None:
    data = _read_registry()
    data.pop(name, None)
    _write_registry(data)


def running() -> dict[str, dict[str, Any]]:
    """Servers this tool started and has not stopped, by profile name."""
    return _read_registry()


def is_listening(port: int) -> bool:
    """Whether something answers on a port. Used to tell 'running' from 'stale record'."""
    return _is_ready(port, timeout=0.5)


def stop_by_name(name: str) -> bool:
    """Stop a server recorded by `llmbench launch`. True if something was stopped.

    The recorded port is checked before the process is signalled. A process identifier
    can be reused by the operating system after the original exits, and killing whatever
    inherited it would be a far worse bug than failing to stop something already gone.
    """
    record = _read_registry().get(name)
    if not record:
        return False
    if not is_listening(int(record["port"])):
        forget_running(name)          # already gone; drop the stale record
        return False
    try:
        os.kill(int(record["pid"]), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        forget_running(name)
        return False
    for _ in range(40):
        if not is_listening(int(record["port"])):
            break
        time.sleep(0.25)
    forget_running(name)
    return True


class launched:
    """Run something against a freshly started server, and always stop it afterwards.

    Written as a context manager so that a failing run cannot leak a server: the stop
    happens on the way out whether the body succeeded or raised.
    """

    def __init__(self, profile: Profile, **kwargs: Any):
        self._profile = profile
        self._kwargs = kwargs
        self.server: Optional[RunningServer] = None

    def __enter__(self) -> RunningServer:
        self.server = start(self._profile, **self._kwargs)
        return self.server

    def __exit__(self, *exc: Any) -> None:
        if self.server is not None:
            stop(self.server)
