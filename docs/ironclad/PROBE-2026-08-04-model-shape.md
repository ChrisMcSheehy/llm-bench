# Probe — what the backend actually reports about a model's shape

Date: 2026-08-04
Runs: Phase 3's mandated first task ("Phase 3 opens with a probe, not with code" —
`plans/2026-07-26-hardware-agnostic-foundation.md`, line 1134)
Design: `DESIGN-hardware-agnostic.md`, **D8** and **D6**

---

## Why this document exists

The design says the D8 memory formula must be written against what a backend really
reports, not against anyone's recollection of the API, and names an unresolved unknown:

> There is a promising lead — the llama.cpp adapter already fetches an `architecture`
> block and stores it unused […] but the exact field names have not been verified
> against a running server.

They have now. **The lead was wrong**, and the probe turned up a second, more urgent
problem in work already shipped. Both are recorded here so the Phase 3 plan is written
against evidence.

## What was probed

| | |
|---|---|
| Binary | `llama-server.EXE`, build `b10144-d73c1d6b2` (winget `ggml.llamacpp`) |
| Plain mode | `-m nomic-embed-text-v1.5.Q4_K_M.gguf --port 8099 -c 2048 -ngl 99 --embeddings` |
| Router mode | `--models-dir <dir> --models-max 1 --port 8098` |
| Endpoints | `/props`, `/v1/models`, `/slots`, `/metrics` |
| GGUF headers | `nomic-embed-text-v1.5` (nomic-bert), `gemma-4-12B-it-heretic` (gemma4, dense), `gemma-4-26B-A4B-it-UD` (gemma4, MoE) |

Raw captures: `probe-llamacpp.json`, `probe-router.json` in the session scratchpad. They
are throwaway; every number they contain that matters is reproduced below.

---

## Finding 1 — the `architecture` block contains no shape data

It is absent entirely in plain mode. In router mode it exists and is:

```json
{"input_modalities": ["text"], "output_modalities": ["text"]}
```

Modalities. Not layers, not heads, not dimensions. **The design's D8 lead is dead.**

The only shape-ish field anywhere in the HTTP API is `/v1/models` → `data[0].meta`:

```json
{"vocab_type": 3, "n_vocab": 30522, "n_ctx": 2048, "n_ctx_train": 2048,
 "n_embd": 768, "n_params": 136727040, "size": 83349984, "ftype": "Q4_K - Medium"}
```

`n_embd` alone cannot produce a KV-cache size: it does not give the layer count, and it
gives the *attention* width rather than the *key/value* width — which are different
numbers on every grouped-query model, i.e. on essentially every current model.

**Consequence for D8:** the formula cannot be computed from llama.cpp's HTTP API. Per the
design's own rule ("Where the shape is not recognised, the tool reports **unknown**"),
HTTP-only deployments must report unknown. A second source is required for a real answer.

## Finding 2 — the GGUF file has everything, and reading it is nearly free

The shape lives in the model file's header, which is a key/value block at the very start.
Reading it requires **no model load** — the 12.67 GB and 16.95 GB files below were both
read in well under a second, because only the header is touched.

Confirmed present, on a validated parser (its `embedding_length` 768 and `context_length`
2048 match what the running server independently reported for the same model, so the
parser is reading real fields rather than plausible garbage):

| Field | nomic-bert | gemma4 12B dense | gemma4 26B MoE |
|---|---|---|---|
| `block_count` | 12 | 48 | 30 |
| `attention.head_count` | 12 | 16 | 16 |
| `attention.head_count_kv` | *absent* | **array[48]** | **array[30]** |
| `attention.key_length` / `value_length` | *absent* | 512 | 512 |
| `attention.key_length_swa` / `value_length_swa` | — | 256 | 256 |
| `attention.sliding_window` | — | 1024 | 1024 |
| `attention.sliding_window_pattern` | — | **array[48] bool** | **array[30] bool** |
| `context_length` | 2048 | 131072 | 262144 |

`model_path` is reported by `/props`, so the file is locatable whenever the server shares
a filesystem with the bench. When it does not, the answer is unknown — which is the
correct answer, not a failure.

## Finding 3 — the design's D8 formula is wrong by ~41× on a real model

D8 states the formula as:

```
layers × key/value heads × head dimension × 2 × bytes per element × context length
```

Every term of that is a scalar. On `gemma-4-12B-it-heretic` **none of the first three is**:

- `head_count_kv` is a **per-layer array**: `[8,8,8,8,8,1, 8,8,8,8,8,1, …]` — 40 layers
  with 8 KV heads, 8 layers with 1.
- `sliding_window_pattern` is a **per-layer boolean array**: 40 layers are sliding-window
  (they cache only the last 1024 tokens, never the full context), 8 are full-attention.
- Head dimension **differs between those two kinds of layer**: 512 for full-attention,
  256 for sliding-window (`key_length_swa`).

Worked at `n_ctx` 131072 with an f16 cache (2 bytes/element):

| | Bytes | |
|---|---|---|
| Naive scalar formula (48 × 8 × 512 × 2 × 2 × 131072) | 103,079,215,104 | **96 GiB** |
| Per-layer sum: 40 SWA (8 × 256 × 2 × 2 × 1024) | 335,544,320 | 0.31 GiB |
| Per-layer sum: 8 full (1 × 512 × 2 × 2 × 131072) | 2,147,483,648 | 2.00 GiB |
| **Correct total** | **2,483,027,968** | **2.31 GiB** |

The naive formula overstates by a factor of **41.5**. This is not a rounding concern; it
is the difference between "this configuration fits on your card" and "it does not."

D8 already anticipated this in prose — it warns that grouped-query attention, sliding
windows and hybrid layers each break a naive formula, and requires *unknown* over a
confident wrong number. The probe confirms the warning was right and the stated formula
does not implement it. **Phase 3 must specify a per-layer sum, not a single product.**

The MoE model shows the same structure (30 layers, KV head counts drawn from `{2, 8}`,
mixed sliding-window pattern), so this is the architecture's normal shape rather than one
odd file.

---

## Finding 4 — Phase 2's identity fix cannot fire on a plain llama-server

**This is a defect in shipped work and is the most urgent thing in this document.**

`targets/llamacpp.py` reads the launch arguments from `/v1/models` →
`data[0].status.args`. The probe shows `status` exists **only in router mode**:

| Mode | `data[0]` keys |
|---|---|
| Plain (`-m model.gguf`) | `aliases, created, id, meta, object, owned_by, tags` — **no `status`** |
| Router (`--models-dir`) | `aliases, architecture, can_remove, created, id, object, owned_by, source, status, tags` |

Run end to end against the plain-mode server, which had been started with `-ngl 99 -c 2048`:

```
label       : 'nomic-embed-text-v1.5.Q4_K_M · Q4_K_M · kv:f16 · d73c1d6'
launch_args : []
n_gpu_layers: None
n_batch     : None
n_ubatch    : None
n_parallel  : None
```

All four fields Phase 2 added are `None`, so two deployments differing only in `-ngl`
still hash identically. **Design criterion 7 — "the same model run with half its layers
on the graphics card and with all of them produces two identities, not one" — is not met
in plain mode.** Phase 2's tests all pass, and none of them could have caught this: they
exercise `_parse_args` and `ModelFingerprint` directly, never `detect()` against a server.

What is salvageable without launch arguments:

| Field | Plain-mode source | Note |
|---|---|---|
| `n_parallel` | `/props` → `total_slots` (reported 4) | Direct hit |
| `n_ctx` | `/props` → `default_generation_settings.n_ctx` | Already used |
| `n_gpu_layers` | none found | Not exposed |
| `n_batch`, `n_ubatch` | none found | Not exposed; server also silently rewrote `n_batch` to 512 to match `n_ubatch`, so the launch value would have been wrong anyway |

That last row is worth keeping: the server's log recorded
`embeddings enabled with n_batch (2048) > n_ubatch (512) … setting n_batch = n_ubatch = 512`.
**The launch argument is a request, not a fact.** Even where argv is readable, it can
disagree with what the server actually did — which argues for preferring server-reported
values over parsed argv wherever both exist.

---

## What this means for the plans

1. **Phase 3 is now writable**, against GGUF-header fields rather than the HTTP API, with
   a per-layer sum and an explicit *unknown* path when the file is unreachable.
2. **Phase 2 needs a follow-up** to close Finding 4. It is a real gap in shipped work, not
   a new feature. Sizing and sequencing are the project owner's call — it is recorded here
   rather than fixed silently.
3. **Detection needs a test against a real server.** Phase 2 shipped 21 green tests over a
   code path that does not work in the common deployment. That gap is a test-shape problem,
   recorded as a lesson in [`LESSONS.md`](LESSONS.md).
