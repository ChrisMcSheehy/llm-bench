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
| `stats.py` | Interval arithmetic on counts: successes + n → a range, or unknown | Import anything else in this project |
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
- **2026-08-16 — the dashboard may name a suite to run; it may never supply one.**
  `POST /api/run` takes a suite name and an optional profile name and nothing else
  (design B6). `resources.resolve_suite` checks the name against `[A-Za-z0-9_-]+` and then
  looks it up in a listing of files that already exist, so the set of things the web layer
  can run is exactly the set the user wrote to disk. The request model sets
  `extra="forbid"`, so an unexpected field is refused rather than dropped — silently
  ignoring one is how a request that meant to smuggle something gets a success back. This
  is decision L1 restated for a second verb: L1 stops the browser choosing which binary
  runs, and a trigger accepting a suite *body* would hand that back, because a suite names
  targets and a target is an address and an argument list. One run at a time, for the
  reason the sweep never starts two servers at once: two servers sharing a graphics card
  contend for it and corrupt the speed figures. The background task records its own
  failure rather than raising, because nothing awaits it and the dashboard would otherwise
  show "running" for ever.
- **2026-08-16 — a test module declares its chart; the dashboard renders it generically.**
  `View(kind, title, x, y, value)` on an evaluator, `Store.view_data` to group its samples
  by one or two dimensions, and one `/api/run/{id}/views` endpoint serving all of them
  (design E3). This replaces `needle_heatmap`, `coding_breakdown` and
  `throughput_by_context` — three queries with an evaluator's name written into the SQL —
  and the three hand-written Plotly calls that consumed them. The plugin claim was only
  ever half true before: a module was discovered and run automatically, then stopped at
  the generic leaderboard, and giving it a chart meant editing three files its author was
  never told about. Five renderers, because they are what the dashboard already drew —
  this is the existing charts described as data, not a speculative abstraction. `kind` and
  `value` are validated in `View.__post_init__` so a typo fails when the module is
  imported rather than when someone opens the dashboard. A cell nobody probed reports
  `None` and a count of zero, where the old heatmap wrote `0.0` and drew a confident dark
  square meaning "this failed".
- **2026-08-16 — a multi-round exchange is a sibling of `run_case`, not a special case of
  it.** `Evaluator.run_conversation` drives a bounded tool-calling conversation and
  returns one Sample (design B4). It sums tokens and latency across rounds and records
  **no per-token speed at all**: a rate over a conversation divides generated tokens by a
  duration that includes reading every intermediate tool result, which is exactly the
  blended figure B3 removed, and taking one round's rate instead would report whichever
  round happened to be last. `tests/test_shared_result_path.py` asserts the absence rather
  than tolerating it, so it stays a decision. Hitting the round bound is not an error: the
  exchange is graded as it stands, because a model that talked itself out of a scenario
  has told you something about the model.
- **2026-08-16 — tool use is measured against a simulated company with a frozen clock.**
  `evaluators/_office.py` holds an invented company and pure-function tools; `agency.py`
  holds the protocol, the scenarios and the scoring (design B4). Everything here rests on
  results being comparable across runs, and a tool benchmark touching a real calendar,
  clock or network is comparable with nothing, including itself an hour later. **The unit
  is the check**, settled at sign-off: the scenario is a dimension, so a per-skill figure
  falls out of the shared aggregation, and a four-check scenario is worth four. Tools are
  described in the prompt and answered in JSON rather than through a native tool API, so
  this measures the model on any backend rather than measuring which backends have
  function calling. **Restraint and focus are scored explicitly** because the expensive
  failure of a tool-using model in practice is not failing to act, it is acting when it
  should not have.
- **2026-08-16 — HumanEval is 164 committed problem directories, not a second harness.**
  Converted once into this project's existing `problem.yaml` + `tests.py` + `solution.py`
  format and committed; the converter is not shipped (design B1). The existing harness
  already samples completions, extracts the code block, runs pytest in a killed-on-timeout
  subprocess and scores with the unbiased pass@k estimator HumanEval itself defines, so a
  second harness would duplicate all of it to gain nothing. Committed rather than
  downloaded because bundled data ships in the wheel and a downloaded corpus is one silent
  upstream edit away from making last month's results incomparable. Attribution and the
  MIT notice live in `problems/coding/HUMANEVAL.md`. `tests.py` imports the solution
  module with a wildcard as well as by name, because three problems' checks call a helper
  the prompt defines rather than only the entry point. Every one of the 164 reference
  solutions was executed through the real harness before being committed, which is how
  those three were found. **Documented difference from HumanEval's own protocol:** it asks
  a model to *complete* the prompt, so the prompt's own code is always present; this bench
  asks for a complete solution, which makes those three problems harder here. A score from
  this bench is not interchangeable with a published HumanEval number, which is true of
  every suite here.
- **2026-08-15 — a third ladder evaluator, and the first with resolution below pass/fail.**
  `evaluators/reassembly.py` plants three labelled fragments of a generated hexadecimal key
  at three depths and grades the reassembled answer in four tiers (design B5). It adds no
  ladder rules, as `_ladder.py`'s entry anticipated. The overlap with `long_context` is
  real and is not the justification: multi-hop retrieval under distractors already exists
  there. What is new is retrieving several items and emitting them *together*, scoring
  assembly separately from retrieval, and **bit accuracy as a gradient** — every other
  long-context figure here is 1.0 or 0.0, which says a line was crossed and nothing about
  how far. `score` is the bit accuracy so the per-rung breakdown inherits the gradient;
  `passed` is the exact match so a wrong-length answer still counts as the failure it is.
  Hexadecimal because it is exactly four bits per character, which is what makes a
  bit-level comparison meaningful; base64 would measure transcription luck. The key is
  generated per cell from a seed and never taken from anything published — a real key may
  sit in training data, and a model reciting a memorised one would score perfectly while
  retrieving nothing. A wrong-length answer reports bit accuracy as **None**, because
  comparing bits across lengths measures alignment and would turn a structural failure
  into a plausible-looking ~50%.
- **2026-08-15 — the haystack is built in one place.** `_sizing.py` holds the filler
  vocabulary and `build_filler`, shared by `needle` and `reassembly`, so the document a
  fact is buried in is the same kind of text in each and results from the two are
  comparable.
- **2026-08-15 — launch profiles resolve their defaults and variables at load time.**
  `load_profiles` merges `defaults.args` ahead of each profile's own and substitutes
  `{name}` from `defaults.vars` (design B7), so what leaves the loader is a real path and
  a complete argument list and nothing downstream ever sees a template. This is what keeps
  criterion 9 true: a profile inheriting `-fa on` and one stating it are the same
  configuration and must hash identically. A restated flag appears twice and is left that
  way — the resolved list is the command line that really ran, llama.cpp takes the last
  occurrence and `_parse_args` overwrites on each, so the two agree. Deduplicating would
  need a table of which flags carry a value and which stand alone, and a wrong entry there
  drops a setting silently. An undefined `{name}` raises rather than passing through: as a
  literal it would be reported later as a missing file, blaming the disk for a typo. Only
  `{identifier}` counts as a placeholder, so a Jinja chat template survives being passed
  as an argument.
- **2026-08-15 — how often the model answered is recorded as a fact, not inferred from a
  string.** `Sample.answered` is True when a gradable response arrived, False when the
  model was asked and produced nothing usable, and None when it was never successfully
  asked (design B2). The default aggregator turns it into `answer_rate`, reported beside
  every accuracy, because an accuracy over an unstated subset is the naked figure D7
  forbids. A separate field rather than parsing `skipped`: that one string carries two
  situations a denominator must tell apart — "never attempted" and "attempted, said
  nothing" — and only the second is a fact about the model. Distinguishing them by the
  `not attempted:` / `no answer:` prefixes the reasons happen to use would make a display
  convention load-bearing. `answer_rate` **is** in `QUALITY_METRICS`: a model that spends
  its budget thinking and returns nothing does so on any machine. Rows written before the
  column existed are NULL, never backfilled — nobody recorded whether they answered.
- **2026-08-15 — an evaluator that overrides `aggregate` calls `super()`.** `ifeval`,
  `human` and `speed` did not, and so silently reported no `skipped_count`, and would have
  reported no `answer_rate`. A module that opts out of the shared aggregator opts out of
  everything added to it later. `ifeval` now publishes `instruction_acc` and `prompt_acc`
  as names for the `score_mean` and `pass_rate` the default aggregator already computes,
  rather than computing them twice.
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
