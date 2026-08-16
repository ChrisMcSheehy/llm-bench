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
   - `reassembly` — three labelled fragments of a generated hex key are planted at
     three depths; the model must find all three and return them joined. Graded in
     four tiers — parts found, order correct, **bit accuracy**, exact match — so a
     degrading rung reports *how far*, not just that a line was crossed. A key of
     the wrong length reports bit accuracy as a dash, never as a number.

   *Tool use:*
   - `agency` — a simulated company, eight simulated tools and a **frozen clock**,
     so "book it tomorrow at 14:00" resolves to the same instant on every run
     forever. Scored per *check*, not per scenario. Two of the checks are the
     reason it exists: **restraint** (asked something no tool can answer, the model
     must decline without calling anything) and **focus** (a plausible,
     always-wrong tool is visible throughout; using it fails). The expensive
     failure of a tool-using model isn't failing to act — it's acting when it
     shouldn't have, confidently.

   *Speed (two figures, because it is two operations):*
   - `speed` — reading the prompt and writing the answer, measured separately at
     stated prompt sizes (~64 / 512 / 2k / 4k / 8k). One warm-up discarded, three
     timed trials, reported as medians. Figures come from the server's own
     timings; a backend that reports none gets a dash rather than a guess.

   *Code & fidelity:*
   - `coding` — generates solutions to pre-solved problems, grades against
     held-out unit tests (`pass@k`). **168 problems**: OpenAI's HumanEval (164,
     MIT, converted and bundled — no download) plus four written for this project.
     Four problems moved the pass rate in steps of 25%; 168 can separate two
     quantisations of one model.
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

### Why speed is two numbers

A model server does two different jobs. It **reads** your prompt — one pass over
every token, limited by how fast the machine computes — and then it **writes** the
answer one token at a time, limited by how fast it moves the weights through
memory. Different bottlenecks, different responses to the same setting.

This bench used to publish a single blend of the two: output tokens divided by
total wall-clock time, averaged across every rung of a context ladder. So the
headline "speed" moved mostly with the prompt size it never mentioned, and two
configurations differing only in context length looked like different models.

`speed` reports `prefill` and `decode` separately, and the leaderboard has a
column for each. Prompt sizes are approximate — prompts are padded using a ratio
measured once — so what gets recorded is the token count the **server** reported,
not the one that was asked for. Prefill scenarios generate exactly one token and
publish no decode figure at all: timing a process that has barely started is noise
wearing a number's clothes.

The old blended figure is gone rather than kept alongside. Two speed numbers where
one is known to be wrong is worse than one correct pair, because the wrong one is
the one already pasted into every existing table.

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

The same rule applies to the *denominator*. Every generating module reports an
**answer rate** beside its accuracy: of the questions the model was actually
asked, how many produced a gradable response at all. A configuration answering
99% of questions at 85% accuracy and one answering 80% at 85% are very different,
and nothing else on the row tells them apart.

A response that never arrived is excluded from the accuracy rather than scored
zero — but it counts against the answer rate, because being asked and saying
nothing is a fact about the model. A test the machine could never run counts
against neither: an honest limit is not a question ducked.

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

`binary` and `model` are required; `port` is optional and a free one is chosen
without it. `args` is passed through **verbatim and in order** — order is
meaningful to llama.cpp, and a later flag beats an earlier one.

### Example profiles, one per comparison

Each pair below changes exactly one thing, because that is what makes two rows
worth putting side by side. Copy the pair you care about; you do not need them all.

```yaml
servers:
  # Two builds, identical everything else — the pull-request case. Each build
  # reports its own commit, and the binary itself is hashed, so a Vulkan build and
  # a ROCm build of the same commit stay two rows instead of being averaged.
  b10441-vulkan:
    binary: C:/builds/llama-b10441-vulkan/llama-server.exe
    model:  C:/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
    args:   ["-ngl", "99", "-c", "65536", "-fa", "on"]
  b10441-pr-9912:
    binary: C:/builds/llama-pr9912-vulkan/llama-server.exe
    model:  C:/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
    args:   ["-ngl", "99", "-c", "65536", "-fa", "on"]

  # Cache compression on and off — what `needle` and `long_context` are for. Pair
  # this with `llmbench memory` to see what the compression bought in bytes.
  kv-f16:
    binary: C:/builds/llama-b10441-vulkan/llama-server.exe
    model:  C:/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
    args:   ["-ngl", "99", "-c", "65536", "-fa", "on"]
  kv-q8:
    binary: C:/builds/llama-b10441-vulkan/llama-server.exe
    model:  C:/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
    args:   ["-ngl", "99", "-c", "65536", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]

  # How much sits on the card. This comparison is *only* possible when llmbench
  # starts the server — a server you merely connect to reports no `-ngl` at all,
  # and both runs would be filed as `launch:unreported`.
  offload-partial:
    binary: C:/builds/llama-b10441-vulkan/llama-server.exe
    model:  C:/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
    args:   ["-ngl", "24", "-c", "65536", "-fa", "on"]

  # Two quantisations of one model. `UD-` is a dynamic quant, which picks its type
  # per layer — the bench keeps the prefix, so this is not the same configuration
  # as a stock Q4_K_M of the same size.
  q4-stock:
    binary: C:/builds/llama-b10441-vulkan/llama-server.exe
    model:  C:/models/Qwen3.6-35B-A3B-Q4_K_M.gguf
    args:   ["-ngl", "99", "-c", "65536", "-fa", "on"]

  # A long-context profile: bigger window, one slot so the whole cache serves it.
  long-256k:
    binary: C:/builds/llama-b10441-vulkan/llama-server.exe
    model:  C:/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
    args:   ["-ngl", "99", "-c", "262144", "-fa", "on", "-np", "1",
             "-ctk", "q8_0", "-ctv", "q8_0"]
```

### Shared defaults and path variables

A set of any size is mostly the same eight lines over and over, so a `defaults`
block carries the parts that repeat:

```yaml
defaults:
  args: ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]
  vars:
    models: C:/models
    builds: C:/builds

servers:
  vulkan-b10441:
    binary: "{builds}/llama-b10441/llama-server.exe"
    model:  "{models}/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    args:   ["-ngl", "99", "-c", "65536"]

  partial-offload:                       # the usual settings, except this one
    binary: "{builds}/llama-b10441/llama-server.exe"
    model:  "{models}/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    args:   ["-ngl", "24", "-c", "65536"]
```

`{name}` is replaced from `defaults.vars`, so moving your model folder is one
edit rather than three hundred. A name with no value is an error naming it and
the profile — left as a literal it would surface later as a missing file, which
blames the disk for a typo. Braces that aren't a plain `{identifier}` are left
alone, so a Jinja chat template can be passed as an argument.

**A profile's own arguments come last and win**, because llama.cpp takes the
later of two conflicting flags — that is what "the usual settings, except this"
has to mean. A profile that restates a default carries the flag twice; that is
the command line that really ran, and both llama.cpp and the fingerprint read the
last occurrence.

Both are resolved when the file is read, so what reaches the launcher is a real
path and what reaches the fingerprint is a complete argument list — never a
template. **A profile that inherits `-fa on` and one that states it are the same
configuration and file under the same identity**, so tidying your profile file
never forks a configuration's history.

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

The quant is read from the model filename **including a dynamic-quantisation
prefix**: `UD-Q4_K_M` and `Q4_K_M` are two configurations, not one. They land at
much the same file size but pick their per-layer types differently, so pooling
them would average away exactly the comparison you were making.

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
from llmbench.evaluators.base import Breakdown, EvalContext, Evaluator, Verdict, View
from llmbench.models import Sample
from llmbench.registry import register

@register
class ShoutEval(Evaluator):
    name = "shout"
    version = "1"
    default_config = {"prompts": {"greeting": "Say hello.", "network": "Explain TCP."}}
    breakdowns = [Breakdown("accuracy", ("topic",))]   # one figure per topic
    views = [View("bar", "accuracy by topic", x="topic")]  # and a chart of it

    async def evaluate(self, ctx: EvalContext) -> list[Sample]:
        cfg = self.resolve_config(ctx.config)

        def grade(res) -> Verdict:            # the only part only you can write
            ok = res.text.strip().isupper()
            return Verdict(score=1.0 if ok else 0.0, passed=ok,
                           meta={"answer": res.text[:120]})

        return [await self.run_case(
                    ctx, case_id=topic, group=topic, dims={"topic": topic},
                    messages=[{"role": "user", "content": f"{p} Reply in capitals."}],
                    grade=grade, max_tokens=64)
                for topic, p in cfg["prompts"].items()]
```

Reference it in a suite under `evaluators:` and it runs. Three things are worth
knowing, and all three exist so that a module is grading logic and nothing else:

- **`run_case` does the call and the record.** A failed call becomes a sample
  carrying the error; a response that never reached an answer becomes a *skipped*
  sample carrying the reason, never a score of zero; and all six measurements —
  tokens in and out, latency, throughput, and the server's own prefill and decode
  speeds — are transferred for you. Your grader returns the verdict.
- **`breakdowns` replaces the group-and-average loop.** Declare the dimensions and
  the metric name; the default `aggregate()` produces one figure per category, each
  stating how many items it rests on. Override `aggregate()` only for something
  genuinely bespoke, and call `super()` when you do — `needle.py`'s effective-context
  figure is the example.
- **`views` gets you a chart.** Declare `View("bar", "accuracy by subject",
  x="subject")` and the dashboard draws it — no endpoint, no SQL, no HTML. Five
  kinds: bar, line, heatmap, table, artifact. Each cell carries its own count, and
  a cell nobody probed is drawn as a gap rather than as a zero.
- **`load_jsonl` reads question files**, resolving "the bundled set" versus "the one
  the suite configured", and naming the file and line number when a line is malformed.

### Shipping a test module as its own package

You do not have to fork this project to extend it. Publish a package that declares
one entry point, and an `llmbench` that has it installed discovers it like a
built-in:

```toml
# in your own pyproject.toml
[project.entry-points."llmbench.evaluators"]
mytest = "llmbench_mytest"
```

Point it at the module or at the class inside it — either works, because
importing is what fires `@register`. `llmbench evaluators` will list it.

Two rules worth knowing before you publish. A module reusing a built-in name is
**rejected**, not silently preferred — two things answering to `needle` would mean
your suite quietly ran the other one. And a plugin that fails to import **stops
discovery with an error naming it**, rather than being skipped: a test module that
is silently absent looks exactly like one that was never installed, and a bench
running fewer tests than you asked for without saying so is the failure this
project takes most seriously.

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
  **With 168 problems this is 41× the execution exposure it was with four.** The
  posture has not changed and neither has the advice; there is simply a lot more
  of it. `problem_ids:` in the suite file runs a subset.
- **A full coding run is 168 model calls and 168 pytest subprocesses.** That is the
  point — four problems could not separate two quantisations — but it is not quick.
  Use `problem_ids:` while you are wiring things up.
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

`llmbench` has 593 automated checks across 50 files. Almost every one of them
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
