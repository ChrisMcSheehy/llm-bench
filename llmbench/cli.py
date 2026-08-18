"""llmbench CLI.

    llmbench detect  --engine llama.cpp --url http://localhost:8080
    llmbench run     suites/default.yaml
    llmbench serve   --port 8900
    llmbench evaluators
    llmbench runs
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

# Windows picks a legacy code page for a redirected stream, and rich does not always
# substitute for characters it cannot encode: printing the sweep's box-drawing rule to a
# cp1252 pipe raised UnicodeEncodeError and killed the run (observed 2026-08-04). Table
# borders and the label's middle dot happen to degrade to '?' instead, so the failure is
# selective rather than obvious — which is why this is fixed once at the entry point
# rather than per character. errors="replace" keeps an un-renderable glyph cosmetic.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):        # already-closed or exotic stream
            pass

from llmbench import hostinfo, launcher
from llmbench.gguf import read_shape
from llmbench.memory import (
    DEFAULT_UBATCH, derivation, has_sliding_window, kv_cache_bytes, swa_cells,
)
from llmbench.orchestrator import Orchestrator, current_host
from llmbench.registry import available
from llmbench.resources import data_path
from llmbench.stats import wilson
from llmbench.store import Store

app = typer.Typer(add_completion=False, help="Local LLM quality test harness")
console = Console()


@app.command()
def detect(engine: str = typer.Option(..., "--engine", "-e"),
           url: str = typer.Option(..., "--url", "-u"),
           model: str = typer.Option(None, "--model", "-m"),
           ubatch: int = typer.Option(
               None, "--ubatch", "-ub",
               help="Declare the server's physical batch size. No endpoint reports it, "
                    "and a sliding-window model's memory figure needs it."),
           kv_unified: bool = typer.Option(
               None, "--kv-unified/--no-kv-unified",
               help="Declare whether the server's cache is one shared pool. Needed for "
                    "the memory figure on a server with several slots.")):
    """Detect a deployed model and print its fingerprint (the run label).

    The two declarations describe a server llmbench did not start. They are recorded as
    stated rather than observed, and are used only for the memory estimate — never for
    the identity, which must record what was seen.
    """
    store = Store()
    orch = Orchestrator(store, log=console.print)
    declared = {k: v for k, v in (("ubatch", ubatch), ("kv_unified", kv_unified))
                if v is not None}
    spec = {"engine": engine, "base_url": url, "model": model, "declared": declared}
    fp = asyncio.run(orch.detect_only(spec))
    console.print(f"\n[bold]{fp.label}[/bold]")
    console.print(f"fingerprint: {fp.fingerprint_hash}")
    console.print_json(json.dumps(fp.model_dump(mode="json", exclude={"raw"}), default=str))
    store.close()


@app.command()
def memory(model: str = typer.Option(..., "--model", "-m",
                                     help="Path to a .gguf model file"),
           ctx: int = typer.Option(..., "--ctx", "-c",
                                   help="Context length to size the cache for"),
           cache_type_k: str = typer.Option("f16", "--cache-type-k", "-ctk"),
           cache_type_v: str = typer.Option("f16", "--cache-type-v", "-ctv"),
           ubatch: int = typer.Option(
               DEFAULT_UBATCH, "--ubatch", "-ub",
               help="Physical batch size. Sliding-window layers cache one batch "
                    "beyond their window, so it changes the answer for such models.")):
    """What a configuration would cost in memory, without loading it.

    Reads the model file's header only, so it answers before anything is running and
    works for a model you are considering rather than one you have started.
    """
    shape = read_shape(model)
    total = kv_cache_bytes(shape, ctx, cache_type_k, cache_type_v, n_ubatch=ubatch)
    if total is None:
        why = derivation(shape, ctx, cache_type_k, cache_type_v, n_ubatch=ubatch)
        console.print("[red]unknown[/red] - the KV-cache size could not be determined.")
        console.print(f"  reason: {why.get('reason', 'unrecognised cache type or shape')}")
        raise typer.Exit(code=1)

    table = Table(title="KV cache (estimate)")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("architecture", str(shape.architecture))
    table.add_row("layers", str(shape.block_count))
    table.add_row("sliding-window layers", str(sum(shape.sliding_window_pattern)))
    if has_sliding_window(shape):
        table.add_row("window cache (tokens)",
                      f"{swa_cells(shape, ctx, ubatch)}  (window "
                      f"{shape.sliding_window} + batch {ubatch})")
    table.add_row("context", str(ctx))
    table.add_row("cache type (K/V)", f"{cache_type_k}/{cache_type_v}")
    table.add_row("[bold]KV cache[/bold]", f"[bold]{_bytes_label(total)}[/bold]")
    console.print(table)
    console.print("[dim]Estimate, computed from the model's shape - not measured.[/dim]")


def _bytes_label(n: int) -> str:
    """MiB below a gigabyte, GiB above it. Two decimals either way."""
    mib = n / (1024 ** 2)
    return f"{mib / 1024:.2f} GiB" if mib >= 1024 else f"{mib:.2f} MiB"


@app.command()
def host(set_cpu_model: str = typer.Option(
             None, "--set-cpu-model",
             help="Record this machine's processor model, which the standard library "
                  "does not reliably report."),
         set_note: str = typer.Option(
             None, "--set-note", help="Any note to keep with this machine's record."),
         binary: str = typer.Option(
             None, "--binary",
             help="Path to a llama.cpp binary, to read the graphics cards from it.")):
    """Show what llmbench knows about this machine, and declare what it cannot read."""
    if set_cpu_model or set_note:
        hostinfo.save_declared({"cpu_model": set_cpu_model, "note": set_note})

    fp = current_host(binary)

    table = Table(title="This machine")
    table.add_column("field")
    table.add_column("value", justify="right")
    table.add_row("os", f"{fp.os} {fp.os_release or ''}".strip())
    table.add_row("arch", fp.arch)
    table.add_row("processors", str(fp.cpu_count))
    if fp.cpu_model:
        table.add_row("processor model", f"{fp.cpu_model}  [dim](declared)[/dim]")
    if fp.declared.get("note"):
        table.add_row("note", f"{fp.declared['note']}  [dim](declared)[/dim]")
    if fp.total_memory_bytes:
        table.add_row("memory", f"{fp.total_memory_bytes / 1024 ** 3:.1f} GiB")
    for d in fp.devices:
        driver = f"  [dim]driver {d['driver']}[/dim]" if d.get("driver") else ""
        table.add_row(d["id"], f"{d['name']}  ({d['total_mib']} MiB){driver}")
    if not fp.devices:
        table.add_row("devices", "[dim]unknown - pass --binary to read them[/dim]")
    table.add_row("[bold]host[/bold]", f"[bold]{fp.host_hash}[/bold]")
    console.print(table)


@app.command()
def run(suite: str = typer.Argument(
            None, help="Path to a suite YAML. Omit to use the bundled default suite."),
        server: list[str] = typer.Option(
            None, "--server", "-s",
            help="Launch profile to run against. Repeat to sweep several builds.")):
    """Run a suite against every target it defines, or against launched servers.

    With --server the suite is run against each named profile in turn: llmbench starts
    the server, runs the whole suite, stops it, then moves to the next. Sequentially,
    never at once — two servers sharing one graphics card would contend for it and
    corrupt the speed figures.
    """
    suite = suite or str(data_path("suites", "default.yaml"))
    store = Store()
    orch = Orchestrator(store, log=console.print)
    try:
        if not server:
            results = asyncio.run(orch.run_suite(suite))
        else:
            results = _run_against_servers(orch, suite, list(server))
    finally:
        store.close()
    partial = [r for r in results if r.status == "partial"]
    if partial:
        console.print(f"[yellow]{len(partial)} run(s) completed with a failed test "
                      f"module.[/yellow]")
    console.print(f"\n[green]Completed {len(results)} run(s).[/green] "
                  f"View: [bold]llmbench serve[/bold]")


def _run_against_servers(orch: Orchestrator, suite: str, names: list[str]):
    """Start each profile, run the suite against it, stop it, then the next.

    Every profile is checked before anything is started, so a typo in the fourth name
    fails immediately rather than after three full suite runs.
    """
    profiles = launcher.load_profiles()
    unknown = [n for n in names if n not in profiles]
    if unknown:
        console.print(f"[red]No such profile(s): {', '.join(unknown)}.[/red] "
                      f"Known: {', '.join(sorted(profiles)) or '(none)'}")
        raise typer.Exit(code=1)

    results = []
    for name in names:
        console.print(f"\n[bold]── {name} ──[/bold]")
        try:
            with launcher.launched(profiles[name]) as server:
                console.print(f"[dim]{name} listening on {server.base_url}[/dim]")
                # The binary travels with the spec so the run can record which
                # graphics cards it ran on: only the binary reports them, and we have
                # it precisely because we started this server ourselves.
                spec = {"engine": "llama.cpp", "base_url": server.base_url,
                        "known_args": server.launch_args,
                        "binary": server.profile.binary}
                results.extend(asyncio.run(orch.run_suite(suite, target_specs=[spec])))
        except launcher.LaunchError as exc:
            # One build failing to start must not abandon the builds after it — that is
            # the normal case when testing a pull request that does not work yet.
            console.print(f"[red]{name} did not start:[/red]\n{exc}")
    return results


@app.command()
def servers():
    """List the model servers llmbench is allowed to start."""
    profiles = launcher.load_profiles()
    if not profiles:
        console.print(f"No profiles. Create [bold]{launcher.profiles_path()}[/bold]:")
        console.print('\n[dim]servers:\n  my-build:\n    binary: /path/to/llama-server\n'
                      '    model:  /path/to/model.gguf\n'
                      '    args:   ["-ngl", "99", "-c", "16384"][/dim]')
        return
    live = launcher.running()
    table = Table(title="Launch profiles")
    table.add_column("name")
    table.add_column("status")
    table.add_column("model")
    table.add_column("args")
    for name, profile in sorted(profiles.items()):
        record = live.get(name)
        if record and launcher.is_listening(int(record["port"])):
            status = f"[green]running[/green] {record['base_url']}"
        else:
            status = "[dim]stopped[/dim]"
        table.add_row(name, status, Path(profile.model).name, " ".join(profile.args))
    console.print(table)


@app.command()
def launch(name: str = typer.Argument(..., help="Profile name from servers.yaml")):
    """Start a model server and leave it running."""
    profiles = launcher.load_profiles()
    if name not in profiles:
        console.print(f"[red]No profile named {name!r}.[/red] "
                      f"Known: {', '.join(sorted(profiles)) or '(none)'}")
        raise typer.Exit(code=1)
    console.print(f"Starting [bold]{name}[/bold]…")
    try:
        server = launcher.start(profiles[name])
    except launcher.LaunchError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    launcher.record_running(server)
    console.print(f"[green]{name} is listening on {server.base_url}[/green]")
    console.print(f"[dim]log: {server.log_path}[/dim]")
    console.print(f"[dim]stop it with: llmbench stop {name}[/dim]")


@app.command()
def stop(name: str = typer.Argument(..., help="Profile name to stop")):
    """Stop a server started by `llmbench launch`."""
    if launcher.stop_by_name(name):
        console.print(f"[green]Stopped {name}.[/green]")
    else:
        console.print(f"[yellow]{name} was not running.[/yellow]")


@app.command()
def evaluators():
    """List discovered test modules."""
    for name in available():
        console.print(f"  • {name}")


@app.command()
def runs():
    """List stored runs with headline metrics, each with what it rests on."""
    store = Store()
    board = store.leaderboard()
    if not board:
        console.print("No runs yet. Try: llmbench run")
        raise typer.Exit()
    t = Table(title="Runs")
    # Reading and writing speed are two columns because they are two measurements with
    # different bottlenecks (design B3). One blended column used to sit here and moved
    # mostly with the prompt size it never mentioned.
    # `answered` sits beside the recall it qualifies (design B2). Two configurations at
    # 0.85 recall are not the same result if one of them answered nine questions in ten
    # and the other answered every one, and no other column here says so.
    for col in ("label", "engine", "kv", "machine", "needle.recall", "needle.answered",
                "coding.pass@1", "speed.prefill", "speed.decode"):
        t.add_column(col)
    for r in board:
        t.add_row(
            str(r.get("label", "")), str(r.get("engine", "")), str(r.get("kv", "")),
            str(r.get("host_label") or "unknown"),
            _fmt(r.get("needle.score_mean"), r.get("needle.score_mean.n"),
                 r.get("needle.score_mean.successes")),
            _fmt(r.get("needle.answer_rate"), r.get("needle.answer_rate.n"),
                 r.get("needle.answer_rate.successes")),
            # pass@1 passes no numerator and shows no interval, deliberately: its value
            # is a mean over problems of the unbiased estimator while its n counts
            # attempts, so Wilson does not describe it (design C1). The two speed
            # columns pass none for the simpler reason that a throughput is not a rate.
            _fmt(r.get("coding.pass@1"), r.get("coding.pass@1.n"),
                 r.get("coding.pass@1.successes")),
            _fmt(r.get("speed.prefill_tps"), r.get("speed.prefill_tps.n"),
                 r.get("speed.prefill_tps.successes")),
            _fmt(r.get("speed.decode_tps"), r.get("speed.decode_tps.n"),
                 r.get("speed.decode_tps.successes")),
        )
    console.print(t)
    store.close()


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8900):
    """Launch the dashboard."""
    import uvicorn
    uvicorn.run("llmbench.dashboard.app:app", host=host, port=port, log_level="info")


def _fmt(v, n=None, successes=None):
    """A figure, what qualifies it, and never one without the other.

    This table gets pasted into places where the reader cannot ask how many questions
    were behind an accuracy (design D7). An unstated count prints as a dash, because a
    count of zero would be a different and false claim.

    A proportion also carries its 95% interval (design C8): 0.833 over six items is one
    question away from 0.667, and the interval is what says so on the page rather than
    leaving the reader to work it out. Everything else carries none — a range around a
    throughput or a count would be arithmetic on the wrong kind of number, and printing
    one would be worse than printing nothing.

    The interval goes on a second line because this table already has nine columns, and
    the bounds are separated by a comma rather than a dash so that every character is
    ASCII — the console encoding on Windows is not guaranteed to carry anything else
    (LESSONS: rich-substitutes-for-some-unencodable-characters-but-not-all).
    """
    if v is None:
        return "—"
    text = f"{v:.3f}" if isinstance(v, float) else str(v)
    count = f"n={n}" if n is not None else "n=—"
    bounds = wilson(successes, n) if successes is not None and n else None
    interval = f"[{bounds[0]:.2f}, {bounds[1]:.2f}] " if bounds else ""
    return f"{text}\n[dim]{interval}{count}[/dim]"


if __name__ == "__main__":
    app()
