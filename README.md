# llmbench

A local-LLM quality test bench. Point it at a llama.cpp / Ollama server (or
OpenRouter), and it will:

1. **Fingerprint** what's actually deployed — model, quant, KV-cache type,
   flash-attention, speculative/MTP config, sampling defaults, **and the
   llama.cpp build commit** — and use that as the run label.
2. **Run pluggable evaluators** against it. Ships with:

   *Capability probes (deterministic, no judge, quant-sensitive):*
   - `mcqa` — multiple-choice family in one harness: MMLU, ARC, HellaSwag,
     TruthfulQA-MC, GPQA, CommonsenseQA, OpenBookQA, WMDP. Point `data_file`
     at a JSONL; grades by exact letter.
   - `math_qa` — GSM8K / MATH numeric, graded via `\boxed{}` / `####` / last
     number.
   - `ifeval` — instruction following with programmatically verifiable
     constraints (bullet counts, word ranges, keyword counts, casing, JSON,
     ending phrases). Reports instruction- and prompt-level accuracy.
   - `structured` — JSON-schema adherence + function/tool-call correctness
     (BFCL flavoured), no native tool API needed.
   - `text2sql` — Spider/BIRD-style, graded by **execution match** against a
     bundled SQLite database.

   *Long context (KV-cache stress — the TurboQuant probes):*
   - `needle` — single-needle NIAH across a context-length × depth grid,
     ladder auto-derived from `n_ctx` (a 1M model → ~128k/256k/512k/1M).
   - `long_context` — RULER/MRCR-style multi-key retrieval (with distractors)
     and variable-tracking chains across the same ladder.

   *Code & fidelity:*
   - `coding` — generates solutions to pre-solved problems, grades against
     held-out unit tests (`pass@k`).
   - `perplexity` — exact teacher-forced PPL via the native `llama-perplexity`
     binary over the detected GGUF (opt-in; the HTTP API can't do this). Run it
     for F16 and each quant of a model to see precisely what the quant costs.

   *Human evaluation:*
   - `human` — generates one response per arena prompt (no auto-score) so you
     can rate configs yourself in the dashboard Arena (blind A/B → Elo, or star
     ratings). See below.
   - `oneshot` — one-shot "build me an app" prompts (Snake, a call-centre
     dashboard, an animated hero, a bar-chart race, a Pomodoro timer, a
     calculator). Each asks for a self-contained HTML artifact; captures
     time-to-complete, tok/s, output size, and a heuristic "did it build" score,
     and stores the artifact so you can render and rate it in the Gallery.
3. **Store** everything in SQLite (one row per graded interaction).
4. **Visualise** it in a dashboard: recall heatmaps, throughput-vs-context,
   coding pass rates, and cross-run trends.

Because the fingerprint captures quant + KV-cache + spec-type + commit + how much
of the model sits on the GPU (`-ngl`) + batch and slot settings (`-b`, `-ub`,
`-np`), swapping
a quant, toggling KV-cache quantisation (TurboQuant), or bumping llama.cpp each
produce a **distinct, comparable run** — which is the point when you're hunting
the best config for limited hardware.

> **One caveat on the launch settings.** `llama-server` only reports the arguments it
> was started with when it runs in **router mode** (`--models-dir`). Started the ordinary
> way (`-m model.gguf`), it reports no argv at all, so `-ngl`, `-b` and `-ub` cannot be
> read and a run is filed as having *unobserved* launch settings — its label carries
> `launch:unreported`. Such a run is never pooled with one whose settings are known, but
> two plain-mode runs differing only in `-ngl` cannot be told apart. The slot count is
> the exception: it comes from the server's own `total_slots`, so it is always accurate.
>
> **Letting llmbench start the server removes this caveat entirely** — see below.

### What a configuration costs

`llmbench memory --model path/to/model.gguf --ctx 131072 --cache-type-k q8_0` reports the
key/value cache size for a configuration **without loading the model** — it reads only the
file's header. That is the other half of every comparison in this tool: compressing the
cache harder saves memory and may cost accuracy, and you cannot weigh the trade without
both numbers.

The figure is an **estimate**, computed from the model's shape rather than measured, and
it is stored with the numbers it was derived from. Where the shape is not recognised, or
the cache type is one this tool does not have a size for, it reports **unknown** rather
than a number.

The estimate is checked against llama-server's own reported allocation, not only against
its own arithmetic: for two real configurations of Gemma 4 12B it reproduces the server's
figures exactly (1088 MiB for a two-slot split cache, 1568 MiB for a unified one).

> **When it answers "unknown", and how to fix it.** Two facts change the answer and no
> llama.cpp endpoint reports either: whether a multi-slot server's cache is unified, and
> the physical batch size (a sliding-window layer caches `window + batch` tokens, not the
> window). Both follow from the launch arguments, so **a server llmbench started always
> gets a number**.
>
> For a server you merely pointed it at, tell it:
>
> ```bash
> llmbench detect --engine llamacpp --url http://localhost:8080 --ubatch 512
> ```
>
> Declared values are recorded as *declared*, feed only the memory estimate, and never
> touch the identity hash — a claim about a server is not an observation of it. Without
> one, a multi-slot server or a sliding-window model reports **unknown** rather than a
> figure that could be wrong by the slot count. See
> `docs/ironclad/PROBE-2026-08-04-host-facts.md`.

### Which machine a result came from

`llmbench host` shows what the bench knows about this machine and its identity hash.
Machine facts come from the standard library and, when llmbench knows which binary to
ask, from `llama-server --list-devices` — no vendor tools and no extra dependency. What
it cannot read, you can declare:

```bash
llmbench host --set-cpu-model "AMD Ryzen 7 7800X3D"
```

Each card also records its **driver version**, which nothing in llama.cpp reports and
which is deliberately *not* part of the machine's identity — drivers change too often to
hash without splitting a machine's history, but an unexplained difference between two
runs cannot be traced without them.

Quality figures pool across machines, because the same model answers the same questions
equally well anywhere. **Speed figures never pool** — they are grouped by machine, and a
run recorded before this existed shows its machine as unknown rather than being averaged
into one that is known. A metric the tool does not recognise is treated as
machine-dependent, because pooling the wrong figure hides two computers inside one
number while refusing to pool costs nothing but statistical power.

### When the machine cannot hold the whole ladder

The long-context tests climb a ladder of context lengths from the bottom. The climb stops
at the first rung the server refuses, or when the rung just finished projects the next one
past a time budget — thirty minutes per test module by default, `time_budget_s` in the
suite file to change it, `null` to remove it:

```yaml
evaluators:
  needle:
    time_budget_s: 3600
```

Rungs that were never attempted are recorded as **skipped, with the reason**, which is not
the same thing as an error: a machine that cannot hold a 512k context has an honest limit,
not a fault. `effective_ctx` still means the largest rung still answering correctly, and
now also tells you the largest one this machine could attempt. A run whose ladder stopped
early is a **completed** run, and the dashboard says which rungs were not attempted and
why rather than leaving a hole where a figure would be.

A test module that fails costs only itself. The modules after it still run, the servers
after it are still tested, and the run is recorded as `partial` with the module named.

> The time projection assumes a rung costs at least linear time in its length, which is
> argued rather than measured — see `docs/ironclad/PROBE-2026-08-04-ladder-timing.md`.

### Every figure says what it rests on

The bundled test sets are small. An accuracy of 0.83 over six questions is one question
away from 0.67, and printed on its own it reads exactly as solidly as an accuracy over six
hundred — so no figure is displayed without the number of graded items behind it. The
dashboard tiles, the chart labels, the heatmap tooltips and the `llmbench runs` table all
carry it, and a figure whose count was never recorded shows a dash rather than a zero.

The memory figure is labelled **estimate** wherever it appears. It is computed from the
model file's header and has never been checked against what a server really allocates.

### How much testing a configuration has had

`http://127.0.0.1:8900/configs` lists every configuration with the evidence behind it: how
many runs, how many of those failed or ran only partly, how many graded samples, over what
span of dates, and on how many machines. All of it is derived from the stored runs, so it
cannot be inflated and never goes stale.

Quality figures there are pooled across machines, because the same model answers the same
questions equally well anywhere. Speed figures are never pooled across machines and are
listed per machine instead.

## Launching servers

If llmbench starts the model server, it knows the arguments, because it supplied them.
That is the only way `-ngl 40` and `-ngl 99` become two comparable configurations on an
ordinary llama.cpp build, and it makes comparing builds a click instead of a ritual.

Describe the servers you want to be able to start in `~/.llmbench/servers.yaml`:

```yaml
servers:
  vulkan-b10148:
    binary: C:/builds/llama-b10148-bin-win-vulkan-x64/llama-server.exe
    model:  C:/models/gemma-4-12B-it-heretic-Q8_0.gguf
    args:   ["-ngl", "99", "-c", "16384", "-fa", "on"]
  turboquant:
    binary: C:/projects/llama-cpp-turboquant/build/bin/Release/llama-server.exe
    model:  C:/models/gemma-4-12B-it-heretic-Q8_0.gguf
    args:   ["-ngl", "99", "-c", "16384", "-ctk", "q8_0", "-ctv", "q8_0"]
```

```bash
llmbench servers          # what is defined, and what is running
llmbench launch vulkan-b10148
llmbench stop vulkan-b10148
```

### Sweeping a suite across builds

```bash
llmbench run suites/default.yaml --server vulkan-b10148 --server turboquant
```

Each profile is started, benched with the whole suite, and stopped before the next
begins — never two at once, because two servers sharing one graphics card contend for it
and corrupt the speed figures. A build that fails to start is reported with the server's
own error and the remaining builds still run, which is the normal case when the pull
request under test does not work yet.

When sweeping, the suite file does not need a `targets:` section; the targets come from
`--server`.

The dashboard has the same controls at **/servers**. Because each build reports its own
commit, running the same model through two builds files them as two configurations that
sit side by side in the leaderboard — which is the point when you are testing a
pull request.

> **The profiles file is an allowlist.** The dashboard can ask to start a profile *by
> name*; it can never supply a binary path or arguments. A web page that could post an
> executable and its arguments would be a way to run anything on the machine hosting the
> dashboard, so the set of things that may run lives in a file only you can edit.

## Install

```bash
pip install -e ".[exec]"      # [exec] pulls pytest for the coding harness
```

Run that from the repository root. Test data, coding problems and the default suite
are bundled inside the package, so every command below works from any directory.

## Use

```bash
# 1. See what a server is running (no test, just the fingerprint):
llmbench detect --engine llama.cpp --url http://localhost:8080

# 2. Run the bundled default suite (or pass your own YAML):
llmbench run
llmbench run my-suite.yaml

# 3. Open the dashboard:
llmbench serve                # http://127.0.0.1:8900

# List discovered test modules / stored runs:
llmbench evaluators
llmbench runs
```

`detect` prints the label, e.g.:

```
Qwen3-8B · Q4_K_M · kv:q8_0 · draft-mtp · a1b2c3d
fingerprint: 4b668a430521d703
```

## Configuring a run

A suite lists targets and evaluator overrides. The bundled default lives inside the
package at `llmbench/data/suites/default.yaml`; copy it out to make your own:

```yaml
targets:
  - engine: llama.cpp
    base_url: http://localhost:8080     # model omitted -> uses the loaded one
  # - engine: ollama
  #   base_url: http://localhost:11434
  #   model: qwen3:14b
  # - engine: openrouter
  #   model: qwen/qwen3-235b-a22b
  #   api_key_env: OPENROUTER_API_KEY

evaluators:
  needle:
    # context_lengths: [131072, 262144, 524288, 1000000]   # or let it auto-ladder
    depths: [0, 25, 50, 75, 100]
  coding:
    n_samples: 1          # raise (+ temperature ~0.8) for real pass@k
    execute: true         # false = grade code extraction only, no execution
```

## Detection: where each field comes from

| Field | Source |
|---|---|
| model, quant, `n_ctx`, sampling, **build commit** | llama.cpp `GET /props` (`model_path`, `build_info` = `b<num>-<commit>`) |
| KV-cache type, flash-attn, `--spec-type` (MTP), draft model | `GET /v1/models` → `status.args` (launch argv) |
| exact token counts (needle sizing) | llama.cpp `POST /tokenize` |
| Ollama model/quant/params/ctx | `GET /api/tags`, `POST /api/show` |
| throughput | response `timings.predicted_per_second`, plus wall-clock tok/s |

Ollama exposes no tokenizer endpoint, so needle sizing there uses a chars/token
heuristic (noted on the samples).

## Loading real benchmark data

`mcqa` and `math_qa` ship tiny **original** sample sets so they run with no
download; they're for wiring and relative comparison between configs. Omit
`data_file` and you get the bundled set. For real numbers, convert a public dataset
to the JSONL schema in each module's docstring and point `data_file` at it — a
`data_file` that doesn't exist is now an error, never a silent fallback to the
samples:

```
mcqa:    {"id","question","options":[...],"answer":"B","subject":"..."}
math_qa: {"id","question","answer":"42","type":"gsm8k"}
```

`ifeval`, `structured`, `long_context`, `text2sql` and `coding` are
self-contained (constraints / synthetic context / bundled DB / your own
problems), so no dataset wrangling is needed.

## How the modules map to the public benchmark zoo

| Module | Covers |
|---|---|
| `mcqa` | MMLU(-Pro/Redux), ARC-C/E, HellaSwag, TruthfulQA-MC, GPQA, CommonsenseQA, OpenBookQA, WMDP, WinoGrande |
| `math_qa` | GSM8K, MATH, MGSM, AMC/AIME-style (final-answer) |
| `ifeval` | IFEval, Multi-IF, COLLIE, RobustIF |
| `structured` | BFCL (function calling), JSON-mode / structured-output evals |
| `text2sql` | Spider, BIRD, LiveSQLBench (execution accuracy) |
| `needle` + `long_context` | NIAH, RULER, MRCR, NoLiMa, GraphWalks-style retrieval |
| `coding` | HumanEval(+), MBPP(+), LiveCodeBench-style (your own problems) |
| `perplexity` | raw quant fidelity (KL/PPL) — the number `llama-perplexity` reports |

Deliberately **out of scope** (need a browser, GUI, vision, or human/LLM
judge): SWE-bench, τ-bench, Terminal-Bench, OSWorld, BrowseComp, LMArena /
Arena-Hard, AlpacaEval, EQ-Bench, and all multimodal suites. A judge-based
evaluator (scoring open-ended output with your Anthropic/OpenRouter key) is the
natural next module if you want the judged ones — the interface already
supports it.

## Human evaluation (Arena)

Auto-graders can't score writing quality, coherence, helpfulness, or tone — so
`llmbench` includes a blind rating game, the same method LMArena uses.

1. Run a suite with the `human` evaluator across **two or more configs** (the
   ones you want to compare — different quants, KV-cache settings, samplers,
   commits). Each generates a response to every arena prompt.
   ```bash
   llmbench run                         # human is enabled by default
   ```
2. `llmbench serve`, then open the dashboard and click **→ human arena**.
3. **Compare (A/B):** you get a prompt and two anonymised, randomly-ordered
   responses. Pick the better one (or Tie / Both weak). Keys `1 2 3 0`, `s` to
   skip. Votes feed an **Elo** leaderboard (K=32, ties = draws).
4. **Rate (stars):** score single responses 1–5 (keys `1`–`5`) for an absolute
   signal when you don't want a head-to-head.

Responses are blind (the config label is never shown until the leaderboard), and
the arena only offers a matchup when a prompt has responses from ≥2 configs.
Votes persist in the same SQLite DB (`hvote` table), so the Elo board is
cumulative across sessions. Add your own prompts to
`datasets/arena_prompts.jsonl` (`{"id","category","prompt"}`).

### One-shot builds & the Gallery

The `oneshot` evaluator asks each config to build a self-contained HTML artifact
(a game, a call-centre dashboard, an animated page, a small app). At **Gallery**
in the dashboard, each config's artifact is rendered live in a sandboxed iframe
with its time-to-complete, tok/s, output size, and heuristic build score — and a
1–5 star rating. Artifacts also flow into the blind A/B arena (rendered
side-by-side), so "which config designs the better UI?" feeds the same Elo board
as text prompts.

Rendered artifacts run in `sandbox="allow-scripts allow-pointer-lock"` iframes:
scripts execute (so games/animations work) but with an opaque origin, so they
can't touch the dashboard, its storage, or the network origin. Add build prompts
to `datasets/oneshot_prompts.jsonl` (`{"id","kind","category","features","prompt"}`).
These are large, slow generations (4k output tokens), so run `oneshot` on a
focused set of configs.

## Adding a test module

Drop one file in `llmbench/evaluators/`. It's discovered automatically — no
registration elsewhere:

```python
from llmbench.evaluators.base import Evaluator, EvalContext
from llmbench.models import Sample
from llmbench.registry import register

@register
class LatencyEval(Evaluator):
    name = "latency"
    version = "1"
    default_config = {"prompts": ["Hello", "Explain TCP in one line."]}

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        out = []
        for i, p in enumerate(ctx.config["prompts"]):
            r = await ctx.generate([{"role": "user", "content": p}], max_tokens=64)
            out.append(Sample(evaluator=self.name, case_id=str(i),
                              latency_ms=r.latency_ms, tok_per_sec=r.tok_per_sec,
                              score=1.0))
        return out
```

Reference it in a suite under `evaluators:` and it runs. The default
`aggregate()` gives you mean score / pass-rate / throughput; override it for
bespoke metrics (see `needle.py`'s heatmap aggregation).

## Architecture

```
targets/      backend adapters — detect() + generate() + count_tokens()
              (llama.cpp, ollama, openrouter; shared OpenAI-compat client)
evaluators/   pluggable test modules (needle, coding, + your own)
registry.py   @register + autoloading discovery
orchestrator  detect -> run evaluators -> aggregate -> persist
store.py      SQLite schema + dashboard queries (DuckDB/Tableau-friendly)
dashboard/    FastAPI JSON API + single-file Plotly frontend
launcher.py   starts and stops model servers from saved profiles
gguf.py       reads a model file's header (layer count, head dimensions)
memory.py     KV-cache cost: shape + cache settings -> bytes, or unknown
hostinfo.py   machine facts: stdlib, plus the binary's device list
cli.py        detect | memory | host | run | servers | launch | stop |
              serve | runs | evaluators
```

"Microservice-style" here means **in-process plugins with a clean, uniform
interface** — each module is self-contained and independently registrable. For
a genuine out-of-process service, wrap an evaluator's `evaluate()` behind
FastAPI and have a thin client implement the same `Evaluator` interface; the
orchestrator doesn't care which side of a socket the module lives on.

## Notes / caveats

- **Coding harness runs model-generated code.** It executes in a temp dir with a
  scrubbed environment and a wall-clock timeout that kills the whole process tree,
  so nothing the code spawned outlives it. That is **all** it is, on every
  operating system — there are no memory, CPU or filesystem limits anywhere, and
  it is **not** a sandbox. Run it against models you trust, set `execute: false`
  to grade extraction only, or run the bench inside a container if the output
  could be adversarial.
- Results are stored at `~/.llmbench/llmbench.db`. Set `LLMBENCH_DB` to override —
  the bench and the dashboard both read it, so they always open the same database.
- Needle depth is split by character proportion (calibrated against the real
  tokenizer once per run) rather than re-tokenising every rung — so a 1M-token
  rung costs one tokenize call, not a million tokens of round-trips.
- `effective_ctx` = the largest ladder rung still at ≥ 0.66 recall — which is also the
  largest rung this machine managed to attempt, since the climb stops where the machine
  does.

## Requirements

Python ≥ 3.10. Runtime deps: httpx, pydantic, pyyaml, typer, rich, fastapi,
uvicorn. The coding evaluator additionally needs pytest (`[exec]` extra).

## The test suite

`llmbench` has 286 automated checks across 43 files. Almost every one of them
guards a real defect that really happened — most test files open by describing
it, with the date.

**[docs/TESTS-EXPLAINED.md](docs/TESTS-EXPLAINED.md)** walks through all of them
in plain English, grouped by what they protect: does the bench know what it is
testing, is the number honest, can it start and stop servers, does it know the
machine, what will a configuration cost in memory, does the data survive, does
the display match what happened, and does it survive the real world.

```bash
pip install -e ".[exec]"
pytest -q
```

## Licence

MIT — see [LICENSE](LICENSE).
