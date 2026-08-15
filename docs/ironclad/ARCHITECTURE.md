# Architecture

## What this project is

A test bench for locally-run AI language models. It detects what a model server is
actually running, executes test modules against it, stores every graded interaction in
SQLite, and presents the results in a web dashboard.

## Layers

Dependencies point downward. Nothing lower may import from anything higher.

| Layer | Responsibility | Must not |
|---|---|---|
| `cli.py` | Command-line entry point; wires everything together | — |
| `dashboard/` | Web API and static frontend | Write test results (it writes only human votes) |
| `orchestrator.py` | Runs a suite: detect → evaluate → aggregate → persist | Contain grading logic |
| `store.py` | SQLite persistence and read queries | Know what any individual test means |
| `evaluators/` | Test modules producing graded samples | Touch the store or the dashboard |
| `targets/` | HTTP adapters for model backends | Import evaluators |
| `registry.py` | Auto-discovery of evaluator plugins | — |
| `resources.py` | Locates bundled data files | — |
| `launcher.py` | Starts and stops model servers from saved profiles | Import anything else in this project |
| `hostinfo.py` | Machine facts: standard library, plus the binary's device list | Import anything else in this project |
| `memory.py` | KV-cache cost arithmetic: shape + cache settings → bytes, or unknown | Import anything else in this project except `gguf.py` |
| `gguf.py` | Reads a model file's header | Import anything else in this project |
| `models.py` | Shared data types and identity hashing | Import anything else in this project |

## Rules

1. A new test module is one file in `evaluators/`. It is discovered automatically; there
   is no registration list to update anywhere else.
2. Bundled data (question sets, coding problems, suite files) is reached only through
   `resources.py`. Never a path relative to the current directory.
3. `models.py` is the single source of truth for what a run, a sample, and a metric are.
4. Identity is a hash of the settings that determine results. If a setting changes the
   numbers, it belongs in that hash.

## Layout

```
llmbench/
  cli.py orchestrator.py store.py registry.py models.py config.py resources.py
  targets/      backend adapters
  evaluators/   test modules
  dashboard/    web API + static frontend
  data/         bundled datasets, coding problems, suite files
tests/          pytest suite
docs/ironclad/  design docs, plans, lessons
```

## Structural decisions

- **2026-07-26 — flattened the package to one level.** The project previously nested
  `llmbench/llmbench/llmbench/`, with an empty middle level, which made the documented
  install command fail. See `plans/2026-07-26-hardware-agnostic-foundation.md`.
- **2026-07-26 — bundled data moved inside the package.** Data files previously sat
  outside the installable package and were found by relative path, so an installed copy
  had none of them and running from the wrong directory silently substituted toy data.
- **2026-07-26 — line endings pinned to LF via `.gitattributes`.** The project is
  developed on Windows and targets macOS and Linux; without this, normalisation depended
  on each contributor's local git configuration.
- **2026-08-04 — the launch-profiles file is an allowlist.** `launcher.py` is the only
  module that starts processes, and the dashboard may reference a profile by name only.
  Allowing the browser to supply a binary path and arguments would make the dashboard a
  way to run arbitrary programs on its host. See `DESIGN-launcher.md`, decision L1.
- **2026-08-04 — machine facts come from the binary and the standard library, never from
  the model server's API.** No llama.cpp endpoint reports any hardware fact, verified
  against a running server (`PROBE-2026-08-04-host-facts.md`). `hostinfo.py` is the only
  module that knows how a device list is obtained or parsed, so a second backend adds a
  function there and changes nothing else. Composing a host from those facts lives in
  `orchestrator.py`, because `hostinfo.py` and `models.py` are both leaf modules that
  import nothing else in this project and the composition needs both.
- **2026-08-04 — model shape is read from the GGUF file, not from the server.** The
  llama.cpp HTTP API reports no layer count, key/value head count or head dimension
  (verified against a running server; see `PROBE-2026-08-04-model-shape.md`). `gguf.py`
  reads them from the model file's header, which costs no model load. The format is known
  to exactly one module so that swapping in the `gguf` package later stays a contained
  change.
- **2026-08-04 — the two ladder evaluators share their stopping rules.**
  `evaluators/_ladder.py` holds `context_ladder` and `climb`; `needle` and
  `long_context` both walk their ladders through it so they can never drift into
  stopping for different reasons. It is a helper module inside the evaluators package,
  like `_extract.py`, and registers nothing. A third ladder evaluator adds no rules.
- **2026-08-04 — a run may finish `partial`.** A test module that raises no longer ends
  the run or the targets after it (design D3d), so `run.status` is now
  `running | ok | partial | error`: not `ok`, because part of the suite did not run, and
  not `error`, because the rest of it did. A run whose ladder merely stopped early stays
  `ok` — a machine's honest limit is not a fault.
- **2026-08-05 — the identity hashes the server executable.** `ModelFingerprint.binary_sha`
  is a short SHA-256 of the `llama-server` binary, set only when llmbench started the
  server and therefore knows the path (design D6b). Two builds of one commit — a Vulkan
  build against a ROCm one, a fork before it is rebased — were previously one
  configuration and got pooled. Hashed rather than named because a path both splits one
  build across machines and merges successive rebuilds at one path. Adding it changed
  every future fingerprint hash; stored rows keep theirs.
- **2026-08-05 — a figure's sample count is produced with the figure, not derived later.**
  `Metric.n` is set by the aggregator in the same expression that computes the value
  (design D7a). The rejected alternative — counting the samples in SQL at read time —
  re-implements each aggregator's filter, and the two can then disagree without anyone
  noticing. `None` means the aggregator did not say, and is displayed as a dash, never as
  a zero.
- **2026-08-05 — accumulated effort is a query over `run` and `sample`.**
  `Store.configuration_effort()` counts runs, failures, graded samples, the date span and
  the machines for each configuration. Nothing is incremented and nothing is cached, so
  the figures cannot be inflated and cannot go stale (design D9a). It is two queries
  merged in Python rather than one join, because joining runs to samples multiplies each
  run by its own sample count.
- **2026-08-03 — the store can add columns after its first release.** `Store.__init__`
  ran `CREATE TABLE IF NOT EXISTS` only, which does nothing to a database that already
  exists, so no column added after the first release would ever have reached one. New
  columns are declared in `_ADDED_COLUMNS` in `store.py`, never by editing `SCHEMA`.
- **2026-08-15 — a test module supplies the grading verdict and nothing else.**
  `Evaluator.run_case` in `evaluators/base.py` makes the model call, turns a failure into
  an `error` sample and an ungradable response into a `skipped` one, and transfers all six
  measurements onto the result (design E1). The transfer had been copied into ten places
  and the copies had drifted: two carried the server's own prefill and decode speeds, one
  carried a single figure, and eight carried neither — so two modules reported no
  server-side speed at all. A grading exception is deliberately not caught: recording a
  defect in our own grading code as `error=` on a sample would file a bench bug as a model
  result. A grader may be a plain function or a coroutine, because `coding` grades by
  running pytest in a subprocess and the other nine are pure.
- **2026-08-15 — per-category figures are declared, not looped.** An evaluator lists
  `Breakdown(metric, by)` entries and the default aggregator produces one figure per
  category, each carrying its own `n` (design E2). Five near-identical group-and-average
  loops are gone. Each breakdown rests on `score`, which every module that wants one sets
  on every sample it grades. The metric name is part of the declaration rather than
  generated, because `store.py`'s `QUALITY_METRICS` reads those names to decide what may
  be pooled across machines. `coding` keeps its own loop: its buckets already exist for
  `pass@k`, and deriving the same figures twice by two rules would let them disagree.
- **2026-08-15 — speed is two measured figures, and the blended one is gone.** The default
  aggregator no longer emits `tok_per_sec_mean` (design B3). It was output tokens over
  *total* wall time, so it included prompt processing, and averaging it across a context
  ladder produced a headline dominated by the variable it hid. `evaluators/speed.py`
  measures prefill and decode separately at stated prompt sizes, from the server's own
  `timings` block — figures that were already being captured and read by nothing. A
  backend that publishes no timings gets **no figure**, never a wall-clock substitute,
  because the substitute is the blend this decision removes. Kept alongside was rejected:
  the wrong number is the one already pasted into existing tables. Each evaluator's raw
  `tok_per_sec` stays on the sample as the cross-check, and `speed` reports it under
  `wallclock_tps` so it cannot be mistaken for the decode figure. The new metric names are
  deliberately absent from `store.QUALITY_METRICS`, so they group by host and never pool
  across machines (design D1).
- **2026-08-15 — prompt sizing is calibrated once, in one place.**
  `evaluators/_sizing.py` holds `chars_per_token`, which `needle`, `long_context` and
  `speed` all use. It was two copies before `speed` needed a third. Sizing by character
  budget from a ratio measured once is what keeps a million-token rung to a single
  tokenize call; the resulting size is approximate, and the count that gets recorded is
  always the server's own.
- **2026-08-15 — a test module may live in a separately installed package.**
  `registry.discover()` reads the `llmbench.evaluators` entry-point group after scanning
  the built-in folder (design E4), so extending the tool no longer means editing it. The
  group name is a public contract: renaming it unregisters every plugin anyone has
  published. Loading the entry point is the whole mechanism, because `@register` is what
  registers — one rule, whether the entry point names a module or a class. Built-ins load
  first so a name clash reads as the plugin being the newcomer. A plugin that fails to
  import stops discovery with an error naming it rather than being skipped: a silently
  absent test module is indistinguishable from one never installed, and the blast radius
  is bounded because only commands that need evaluators call discovery. This executes
  third-party code by design — installing a package already grants that, and nothing is
  discovered that the user did not choose to install.
- **2026-08-15 — the quantisation scheme is part of the quant label.** `parse_quant` keeps
  an Unsloth `UD-` prefix instead of matching the plain token inside the longer name, so
  `UD-Q4_K_M` and `Q4_K_M` are two configurations rather than one. They are the same
  nominal size and different programs — Dynamic quants choose the type per layer. Kept in
  the existing `quant` string rather than given a field of its own: it is the label the
  files actually ship under, and a new field would mean a store migration to record
  something the string already says. This changes the hash of any future detection of a
  `UD-` model; stored rows keep theirs, as with `binary_sha`.
- **2026-08-15 — question files load through `resources.load_jsonl`.** Reading sits beside
  `resolve_data_file` rather than in the evaluators package (design E5), because that
  module is already the only place that knows where bundled data lives, and a module
  author should not have to find two files to read one. It names the file and the line
  when a line does not parse — four hand-written copies raised an error naming neither.
