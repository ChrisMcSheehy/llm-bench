# Probe — what the backend reports about the machine, and what `n_ctx` actually means

Date: 2026-08-04
Runs: evidence for **Phase 4** (host as an entity) before any of it is planned
Design: `DESIGN-hardware-agnostic.md`, **D1** and **D2**;
turns up a defect against **D8**

---

## Why this document exists

D2 says the machine facts come from three places: Python's standard library, "the backend
**if it reports them**", and otherwise the user's own declaration. Only the first of those
is certain without looking. The phrase "if it reports them" was never checked against a
running server, and Phase 3's probe already showed that this API reports less than the
design assumed — so the same assumption is checked here before Phase 4 is planned rather
than after it ships.

The probe answered that question and, incidentally, settled an open question left in the
Phase 3 plan. The answer to that one is bad: **shipped code returns a memory figure that
is too small by the slot count on a multi-slot server.** See Finding 4.

## What was probed

| | |
|---|---|
| Binary | `C:/tools/llama.cpp/llama-server.exe`, build `b10148-ddfc2288e`, Clang 20.1.8, Windows x86_64 |
| Machine | AMD Radeon RX 7900 XTX (24 GB) + Radeon integrated (16 GB), Windows 11 |
| Run A | `-m nomic-embed-text-v1.5.Q4_K_M.gguf --port 8099 -c 2048 -ngl 99 --embeddings` |
| Run B | `-m gemma-4-12B-it-heretic-Q8_0.gguf --port 8098 -c 8192 -np 2 -ngl 0` |
| Endpoints | `/props`, `/health`, `/slots`, `/v1/models`, `/metrics`, `/api/devices`, `/devices` |
| Binary flags | `--list-devices`, `--help` (device and KV-cache sections) |

Run B is a **non-embedding model with more than one slot**, which is exactly the run the
Phase 3 plan named as needed to settle its open question 1 and did not have.

---

## Finding 1 — the HTTP API reports nothing at all about the machine

Every endpoint the server exposes was checked on build `b10148`, one build newer than the
Phase 3 capture. `/props` carries 19 keys; none of them describes hardware:

```
bos_token, build_info, chat_template, chat_template_caps, cors_proxy_enabled,
default_generation_settings, endpoint_metrics, endpoint_props, endpoint_slots,
eos_token, is_sleeping, media_marker, modalities, model_alias, model_ftype,
model_path, total_slots, ui, ui_settings
```

Searching the whole `/props` document for `gpu`, `vulkan`, `cuda`, `device`, `vram`,
`memory`, `cpu` and `offload` returns nothing. The single hit for `backend` is
`default_generation_settings.params.backend_sampling`, a sampling flag.

`/metrics` answers **501 Not Implemented** unless the server is started with `--metrics`.
`/api/devices` and `/devices` do not exist (404).

> **Consequence for D2.** Read literally — "read from the backend if it reports them" —
> the graphics-card branch is dead code for llama.cpp. Nothing in the HTTP API will ever
> fill it.

## Finding 2 — the *binary* reports the graphics cards exactly, for free

The information D2 wants is one command away, and needs no server, no model, and no
vendor-specific tool:

```
$ llama-server.exe --list-devices
Available devices:
  Vulkan0: AMD Radeon RX 7900 XTX (24560 MiB, 23749 MiB free)
  Vulkan1: AMD Radeon(TM) Graphics (16208 MiB, 15397 MiB free)
```

That is device name, total memory, free memory, and the compute backend (`Vulkan`),
per device — which is more than D2 asked for and avoids the `nvidia-smi` parsing the
design explicitly rejected.

Three things about this matter for planning:

1. **The backend name is identity-relevant and the design does not mention it.** The same
   card driven through Vulkan and through ROCm runs separately-written implementations of
   the same mathematics — which is precisely the reason D1 hashes the operating system.
2. **Free memory must never be hashed.** It changes between two runs on an idle machine.
   Total memory is stable; free memory is a reading, not a fact about the machine.
3. **This is knowable only when llmbench knows the binary.** That is true when it launched
   the server from a profile (`launcher.py` has the path) and false when the user pointed
   it at an address. The same asymmetry as Finding 4 of the model-shape probe: what we
   started, we know.

## Finding 3 — the startup log has the devices too, but only at raised verbosity

A tempting third source, since `launcher.py` already captures the child's output. At the
default verbosity it is useless: the entire startup log is 13 lines and mentions no
device, no backend and no memory.

At `-lv 10` it carries more than `--list-devices` does:

```
common_param:   - Vulkan0 : AMD Radeon RX 7900 XTX (24560 MiB, 23749 MiB free)
common_param:   - Vulkan1 : AMD Radeon(TM) Graphics (16208 MiB, 15397 MiB free)
common_param:   - CPU     : AMD Ryzen 7 7800X3D 8-Core Processor (31904 MiB, 4757 MiB free)
```

— including the processor model and system memory, which D2 otherwise expects the user to
declare. It is still log-scraping of a format with no stability promise, and it requires
raising a verbosity the user did not ask for. Recorded as an option, not a recommendation:
`--list-devices` (Finding 2) gets the same graphics-card facts from a documented flag.

## Finding 5 — the server prints its own cache allocation, and ours is 2.4× too small

Raising verbosity for Finding 3 exposed something more valuable: llama-server states
exactly what it allocated. For `-c 8192 -np 2` on `gemma-4-12B` (`-ub 512` by default):

```
llama_kv_cache_iswa: creating non-SWA KV cache, size = 4096 cells
llama_kv_cache: size =  128.00 MiB ( 4096 cells,  8 layers, 2/2 seqs), K (f16): 64.00 MiB, V (f16): 64.00 MiB
llama_kv_cache_iswa: creating     SWA KV cache, size = 1536 cells
llama_kv_cache: size =  960.00 MiB ( 1536 cells, 40 layers, 2/2 seqs), K (f16): 480.00 MiB, V (f16): 480.00 MiB
```

**1088 MiB.** `llmbench` estimated **448 MiB** for the same server. Three separate things
account for the gap, and only the first was known before this probe:

1. **Cells are per sequence.** `4096 cells … 2/2 seqs` is 4096 *each*; the arithmetic
   checks out exactly at 8 layers × 1 KV head × 512 dim × 4096 × 2 bytes × 2 seqs = 64 MiB
   of keys. This is Finding 4, and it is now fixed — the slot count multiplies the
   **bytes**, never the token count.
2. **A sliding-window layer caches `n_swa + n_ubatch`, not the window.** 1536 = 1024 + 512.
   Confirmed with a second run at `-ub 256`, which produced **1280 cells** = 1024 + 256.
   `memory.py` uses `min(sliding_window, n_ctx)` = 1024, so it understates every window
   layer by 1.5× at the default batch — and window layers are 40 of this model's 48.
3. **Consequently the headline 2.31 GiB figure for this model at 131072 is also low**, by
   the same 1.5× on the window layers. It was validated against the model file's shape,
   which was the right check for D8a's per-layer sum, and against nothing that knew how
   llama.cpp actually sizes a window cache.

> **Both are now fixed** (design D8b and D8c). Three further runs pinned the window rule
> down: `cells = min(n_ctx, window × sequences_sharing + n_ubatch)`, confirmed at
> `-c 512` (capped to 512), `-c 2048` (1536), `-ub 256` (1280) and auto-slots unified
> (1024×4+512 = 4608). Detection now reproduces the server's own allocation **exactly**
> for both a split cache (1088 MiB) and a unified one (1568 MiB). Where the batch size
> was never observed and the model has window layers, the answer is unknown rather than
> assumed.

The deeper lesson is that a validation existed all along and was never used: the server
states the answer, so any estimate this project makes can be checked against a real
allocation rather than against its own arithmetic.

---

## Finding 4 — `n_ctx` is **per slot**, and a shipped figure is wrong because of it

The Phase 3 plan left this open question:

> Which context length is the right one to size against? […] If `-c` is divided among
> slots on a non-unified server, a detected estimate would be **too large** by the slot
> count.

Run B settles it, and the error runs the other way.

| Launched with | `total_slots` | `default_generation_settings.n_ctx` | log says |
|---|---|---|---|
| `-c 2048` (no `-np`) — run A | 4 | **2048** | `n_ctx_slot = 2048, kv_unified = 'true'` |
| `-c 8192 -np 2` — run B | 2 | **4096** | `n_ctx_slot = 4096, kv_unified = 'false'` |

`/slots` agrees: two slots, `n_ctx` 4096 each.

So the reported `n_ctx` is the **per-slot** context, not the total. In both runs the total
cache is the `-c` value:

- unified (run A): one shared pool of 2048 — total is the reported `n_ctx`
- non-unified (run B): 2 × 4096 — total is the reported `n_ctx` **times `total_slots`**

`llmbench` currently sizes the cache from the reported `n_ctx` alone
(`targets/llamacpp.py`, `detect()`). That is right for run A and **half the true figure**
for run B — in general, too small by a factor of `total_slots` whenever the cache is not
unified.

### Why it cannot simply be fixed by multiplying

Whether the cache is unified decides the factor, and **`kv_unified` is not in `/props`,
`/slots` or `/v1/models`** — it appears only in the startup log. From HTTP alone the two
cases are indistinguishable, so the total is ambiguous by a factor of `total_slots`.

The flag's default explains both observations:

```
-kvu, --kv-unified, -no-kvu, --no-kv-unified
        use single unified KV buffer shared across all sequences
        (default: enabled if number of slots is auto)
```

Run A passed no `-np`, so the slot count was auto and unification was on. Run B passed
`-np 2`, so it was off. **The rule is derivable from the launch arguments — which llmbench
has whenever it started the server, and never otherwise.**

### The narrow good news

For a single-slot server the two cases coincide, so the shipped figure is correct there.
The defect bites exactly the multi-slot configurations a bench exists to compare.

---

## What this means for the plans

1. **Phase 4's D2 needs an amendment before it is planned.** The graphics-card facts come
   from `--list-devices` on the binary, not from the HTTP API; the compute backend belongs
   in the record and probably in the hash; free memory belongs in neither.
2. **Phase 3 needs a correction, and it is a design decision, not a patch.** D8 says a
   confident wrong number is worse than no number, and the current figure is confidently
   wrong on multi-slot servers. The options are on the table in the report accompanying
   this document; the choice between "multiply when we launched it, unknown otherwise" and
   "always label the figure per-slot" changes what the tool claims, so it wants sign-off.
3. **Neither is a reason to distrust the per-layer sum.** The arithmetic in `memory.py` is
   unaffected; what is wrong is the number of tokens handed to it.
