# Probe — does a context rung really cost at least linear time?

Date: 2026-08-04
Runs: Phase 5's task C10
(`plans/2026-08-04-graceful-degradation-phase-5.md`)
Design: `DESIGN-hardware-agnostic.md`, **D3c**

> **Status: MEASURED 2026-08-05.** Against `llama-server` build `b10144-d73c1d6b2` running
> gemma4-heretic on a Radeon RX 7900 XTX. See **Results** below.
>
> **Verdict: the claim is half right, and right where it matters.** The linear projection is
> a lower bound at the two upper steps (ratios 1.21 and 1.43, the quadratic term appearing as
> argued) and an *over*-estimate at the bottom step (ratio 0.58), because a rung costs fixed
> overhead plus a term proportional to length, and scaling through the origin ignores the
> fixed part. The rule is unchanged: the over-estimate occurs three orders of magnitude below
> the budget, where it cannot cause a skip. D3c's wording is corrected.
>
> The assumed ~850 tok/s behind the 1800-second default also held: measured 641.8–1223.2,
> and 641.8 at the top rung. The default stands.
>
> *(Superseded: this document previously read "NOT MEASURED — no llama-server was available
> on the machine where Phase 5 was executed." One became available on 2026-08-05.)*

---

## Why this document exists

Phase 5 made the context ladder stop climbing when the rung just completed projects the
next one past a time budget. The projection is linear:

```
projected_seconds = seconds_taken × (next_length / this_length)
```

D3c defends that with an argument rather than a measurement:

> Processing a prompt costs *at least* linear time in its length — attention adds a
> quadratic term on top — so the linear projection is a **lower bound** on what the next
> rung will really cost. Stopping only when even the optimistic projection overshoots
> means the rule never skips a rung that would have fitted.

The argument is sound in principle and **has never been checked against a server.** That
is the exact shape of the mistake this project has already paid for once: Phase 3's KV
cache estimate was verified three separate ways, every one of them comparing the
arithmetic against its own inputs, and it was wrong by a factor of 2.4 against what
llama-server actually allocated. See LESSONS.md,
`validate-an-estimate-against-the-thing-it-estimates`.

## What is at risk if the claim is false

| If the truth is… | Then the rule… | Severity |
|---|---|---|
| Actual ≥ projected at every step (the claim) | never skips a rung that would have fitted | correct as designed |
| Actual a little below projected | is conservative — may skip a rung that would have fitted | a weakness worth recording, not a defect |
| Actual far below projected | skips rungs needlessly, and the bench under-reports `effective_ctx` on capable machines | **defect — redesign the rule** |
| Actual far above projected | lets through a rung that runs for hours, which is what the budget exists to prevent | **defect — the budget does not bind** |

The second column is why the direction of the error matters more than its size.

## A second, smaller unknown rides along

The chosen default budget — **1800 seconds per ladder evaluator** — was derived from an
assumed prompt-processing rate of **~850 tokens/second** for an 8B model on a 24 GB card.
That figure is the constant the mock backend in `selftest.py` reports. Nobody measured it.
The same run that settles the projection settles this: if the real rate is far from 850,
the default budget should move, not the rule.

## How to settle it

1. Start a real `llama-server` — a launch profile, or by hand — with a model whose context
   comfortably exceeds the top rung below.

2. Run `needle` alone, with no budget, over a ladder of at least four rungs. Suite:

```yaml
name: ladder-timing
evaluators:
  needle:
    context_lengths: [4096, 8192, 16384, 32768]
    depths: [50]
    time_budget_s: null
```

3. Read the per-rung wall time out of the database rather than off a stopwatch. Write this
   to `rung_times.py` at the repository root, run it, then delete it:

```python
"""Seconds spent on each rung of the most recent needle run, and the projection."""
import json

from llmbench.store import Store

store = Store()
run_id = store.runs()[0]["run_id"]
rows = store.conn.execute(
    "SELECT dims_json, latency_ms FROM sample "
    "WHERE run_id=? AND evaluator='needle' AND latency_ms IS NOT NULL",
    (run_id,)).fetchall()
store.close()

seconds: dict[int, float] = {}
for row in rows:
    length = int(json.loads(row["dims_json"])["context_len"])
    seconds[length] = seconds.get(length, 0.0) + row["latency_ms"] / 1000.0

ladder = sorted(seconds)
print(f"run {run_id}")
for i, length in enumerate(ladder):
    print(f"  {length:>7} tokens: {seconds[length]:8.1f}s actual")
    if i + 1 < len(ladder):
        nxt = ladder[i + 1]
        projected = seconds[length] * (nxt / length)
        actual = seconds[nxt]
        verdict = "lower bound holds" if actual >= projected else "OVER-ESTIMATE"
        print(f"          -> {nxt}: projected {projected:8.1f}s, "
              f"actual {actual:8.1f}s, ratio {actual / projected:.2f}  {verdict}")
```

4. Fill in the table below, and update D3c to say what was found — including the measured
   tokens/second, so the budget default rests on a number rather than on the mock's.

## Results — measured 2026-08-05

| Server | Build | Model | Machine | Rungs | Filled in |
|---|---|---|---|---|---|
| `llama-server` | `b10144-d73c1d6b2` (Clang 20.1.8, Windows x86_64) | gemma4-heretic, `Q8_0`, 11.9 B params, 48 layers (40 sliding-window), `-ngl 99 -c 32768 -fa on`, 4 slots, unified cache | Windows 11, Radeon RX 7900 XTX 24560 MiB (Vulkan), driver 32.0.31007.5012 | 4096 / 8192 / 16384 / 32768, one depth, no budget | **yes** |

Per rung, one sample each, `max_tokens` 32:

| Rung | Prompt tokens | Wall seconds | Server prompt tok/s |
|---|---|---|---|
| 4096 | 3842 | 6.3 | 676.3 |
| 8192 | 7878 | 7.4 | 1223.2 |
| 16384 | 15985 | 17.9 | 949.4 |
| 32768 | 31960 | 51.2 | 641.8 |

| From | To | Projected | Actual | Ratio | Lower bound holds? |
|---|---|---|---|---|---|
| 4096 | 8192 | 12.6s | 7.4s | **0.58** | **no — over-estimate** |
| 8192 | 16384 | 14.8s | 17.9s | 1.21 | yes |
| 16384 | 32768 | 35.8s | 51.2s | 1.43 | yes |

### The finding, in one sentence

**The projection is not a lower bound at the bottom of the ladder, and is a lower bound
everywhere it matters.**

The claim in D3c holds for the two upper steps, and the ratios grow — 1.21 then 1.43 — which
is the quadratic attention term appearing exactly as argued. It fails at the first step, and
the reason is structural rather than noise: a rung's cost is not proportional to its length,
it is **fixed overhead plus a term proportional to length**. At 4096 tokens the fixed part —
request handling, sampling, and the 32 generated tokens, which cost the same at every rung —
is a large share of 6.3 seconds, so doubling the context does not double the time.

Scaling a small rung's time through the origin therefore over-states the next one. The error
shrinks as the fixed part becomes negligible, which is why the ratio crosses 1.0 between 8k
and 16k and keeps climbing.

### Why this is an accepted weakness and not a defect

The over-estimate can only cost something if it fires, and it fires only when
`elapsed + projected > budget`. At the bottom of the ladder the numbers are three orders of
magnitude below the 1800-second default: the 4096 rung projects 12.6s against a budget of
1800s. For the rule to wrongly refuse a rung, the over-estimate would have to occur *near the
budget* — and near the budget the rungs are large, where the projection under-estimates
instead.

So the direction of the error is wrong precisely where the error cannot matter, and right
where it can. D3c's rule stands unchanged; D3c's *justification* is corrected to say
"a lower bound once the rung is large enough for its fixed cost to be negligible" rather
than "a lower bound".

### The second unknown, settled

The default budget was derived from an assumed **~850 tokens/second** for an 8B model on a
24 GB card — the mock backend's constant, never measured.

Measured on this machine: **641.8 to 1223.2 tokens/second**, and 641.8 at the top rung where
it matters most. The assumption was sound: 850 sits inside the measured range, and the
figure at the largest context is within 25% of it. **The 1800-second default needs no
change.** Working it through at the measured 641.8 tok/s, a five-depth needle rung at 131072
tokens costs roughly 1020 seconds — still inside the budget, which is the behaviour the
thirty-minute figure was chosen to get.

### Caveats on this measurement, stated rather than buried

1. **One sample per rung.** Enough to establish the shape and the order of magnitude, not
   enough to put an error bar on any single figure. A repeat run would strengthen it.
2. **The model answered nothing.** Every rung scored 0.00, because this model emits
   `reasoning_content` and spends a 32-token budget thinking — see the defect recorded on
   2026-08-05 in LESSONS.md, `a-truncated-answer-is-not-a-wrong-answer`. This does **not**
   invalidate the timing: the prompt was processed in full at every rung, and generation was
   pinned at 32 tokens for all four, so the differences across rungs are prompt processing
   almost undiluted. It does mean the absolute seconds here are lower than a real needle run
   with a full answer, so the fixed-cost share is, if anything, *understated* — the true
   over-estimate at the bottom is a little worse than 0.58.
3. **Prompt caching was off** (`cached_tokens: 0` observed). A server reusing a cached prefix
   would break the projection in the other direction entirely; that is not tested here.
4. **One backend, one model, one machine.** A sliding-window model on a Vulkan backend. A
   dense model, or CUDA, may sit differently.

## What must not be done instead

**Do not substitute the mock backend.** `MockTarget.generate` returns a hard-coded
`latency_ms=100.0` regardless of prompt length, so every projection it produces will agree
with every actual to within rounding. It would turn this document green while proving
nothing at all — which is the self-agreeing check the whole lesson is about.
