"""llama.cpp (llama-server) target.

Detection sources, confirmed against the current server README:
  GET /props        -> n_ctx, model_path (quant in filename), build_info
                       ("b<num>-<commit>"), default sampling, speculative flag,
                       chat_template.
  GET /v1/models    -> id (routing alias), path, status.args (the launch argv:
                       we parse -c, -ctk/-ctv KV-quant, -fa, --spec-type MTP,
                       -md draft model).
  POST /tokenize    -> exact token counts for context sizing.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

from llmbench.gguf import read_shape
from llmbench.memory import (
    DEFAULT_UBATCH, derivation, has_sliding_window, kv_cache_bytes,
)
from llmbench.models import (
    ModelFingerprint, model_name_from_path, parse_params, parse_quant,
)
from llmbench.targets.base import Target, approx_tokens


def _parse_args(args: list[str]) -> dict[str, Any]:
    """Pull the fields we care about out of the launch argv."""
    out: dict[str, Any] = {}
    aliases = {
        "-c": "ctx", "--ctx-size": "ctx", "-ctk": "ctk",
        "--cache-type-k": "ctk", "-ctv": "ctv", "--cache-type-v": "ctv",
        "-md": "draft_model", "--model-draft": "draft_model",
        "--spec-draft-model": "draft_model", "--spec-type": "spec_type",
        "-ngl": "ngl", "--n-gpu-layers": "ngl", "--gpu-layers": "ngl",
        # Batch and slot settings change how many calculations are bundled together,
        # which changes the order they are summed in — so they change the answers.
        "-b": "n_batch", "--batch-size": "n_batch",
        "-ub": "n_ubatch", "--ubatch-size": "n_ubatch",
        "-np": "n_parallel", "--parallel": "n_parallel",
    }
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in aliases and i + 1 < len(args):
            out[aliases[tok]] = args[i + 1]
            i += 2
            continue
        # Whether the key/value cache is one shared pool or one per slot. No HTTP
        # endpoint reports it, so the flag is the only source — and it decides whether
        # the server's per-slot context has to be multiplied by the slot count.
        if tok in ("-kvu", "--kv-unified"):
            out["kv_unified"] = True
            i += 1
            continue
        if tok in ("-no-kvu", "--no-kv-unified"):
            out["kv_unified"] = False
            i += 1
            continue
        if tok in ("-fa", "--flash-attn"):
            # value form (-fa on) or bare flag
            if i + 1 < len(args) and args[i + 1] in ("on", "off", "auto"):
                out["flash_attn"] = args[i + 1]
                i += 2
            else:
                out["flash_attn"] = "on"
                i += 1
            continue
        i += 1
    return out


def binary_sha(path: Optional[str]) -> Optional[str]:
    """A short hash of the executable, or None if it cannot be read.

    Read in chunks because a llama-server build is tens to hundreds of megabytes and
    this runs once per run, not once per sample.

    An unreadable path is **unknown, not an error**: a profile may point at a binary
    that has since moved, and the server is up and answering regardless. A run that
    cannot say which build produced it is worth less than one that can, and is worth far
    more than no run at all.
    """
    if not path:
        return None
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]
    except OSError:
        return None


def _as_int(value: Optional[str]) -> Optional[int]:
    """Launch arguments arrive as text. Anything non-numeric is recorded as absent.

    Recording it as absent rather than guessing matters because these values reach the
    identity hash: a wrong number files results under a label that does not describe
    them, whereas a missing one is merely missing.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_ubatch(parsed: dict, observed: bool,
                    declared: dict) -> tuple[Optional[int], str]:
    """The physical batch size, and where it came from.

    Order: what the server was started with, then the documented default when we saw
    the arguments at all, then what the user declared, then unknown.
    """
    from_args = _as_int(parsed.get("n_ubatch"))
    if from_args is not None:
        return from_args, "observed"
    if observed:
        return DEFAULT_UBATCH, "default (arguments were observed)"
    stated = _as_int(str(declared.get("ubatch"))) if declared.get("ubatch") else None
    if stated is not None:
        return stated, "declared"
    return None, "unknown"


def _resolve_unified(parsed: dict, observed: bool,
                     declared: dict) -> tuple[Optional[bool], str]:
    """Whether the cache is one shared pool, and where that came from."""
    if "kv_unified" in parsed:
        return bool(parsed["kv_unified"]), "observed"
    if observed:
        # -kvu defaults on only when the slot count is left automatic.
        return "n_parallel" not in parsed, "inferred from the observed arguments"
    if "kv_unified" in declared:
        return bool(declared["kv_unified"]), "declared"
    return None, "unknown"


def _cache_pools(n_ctx: Optional[int], slots: Optional[int],
                 unified: Optional[bool]) -> tuple[Optional[int], dict[str, Any]]:
    """How many independent caches of `n_ctx` tokens the server allocates.

    llama-server reports the context **per slot**: started `-c 8192 -np 2` it reports
    4096, and allocates one cache of that size per sequence. Its own log says so —
    `size = 128.00 MiB (4096 cells, 8 layers, 2/2 seqs)` — cells being per sequence and
    the buffer holding both (measured 2026-08-04, see
    docs/ironclad/PROBE-2026-08-04-host-facts.md).

    So the slot count multiplies the **bytes**, not the token count. The two are the
    same thing on a dense model and are not on a sliding-window one, where a window
    layer caches the same amount however long the context is.

    A unified cache is one pool however many slots use it. Whether it is unified is
    reported by no endpoint, so `unified` is resolved by the caller from the launch
    arguments or a declaration; None means neither was available, and the answer is then
    unknown (design D8b) rather than one of the two candidates.

    Returns the pool count and the notes to record beside the figure. The notes carry
    ``known: False`` and a reason when it cannot be determined, so they override
    whatever the memory module concluded from the shape alone.
    """
    note: dict[str, Any] = {"n_ctx_per_slot": n_ctx, "total_slots": slots}
    if not n_ctx:
        return None, note

    if not slots or slots <= 1:
        # One slot: one cache, so nothing is ambiguous.
        note.update(cache_pools=1, kv_unified=None)
        return 1, note

    if unified is None:
        note.update(
            known=False, kv_unified=None, cache_pools=None,
            reason=(f"context is reported per slot and this server has {slots} slots; "
                    "whether its cache is unified is not reported over HTTP and the "
                    "launch arguments were not observed, so the cache is either one "
                    f"pool of {n_ctx} tokens or {slots} of them"),
        )
        return None, note

    pools = 1 if unified else slots
    note.update(kv_unified=unified, cache_pools=pools)
    return pools, note


class LlamaCppTarget(Target):
    engine = "llama.cpp"

    #: Arguments the bench itself started this server with, when it started it.
    #: llama-server reports its own argv only in router mode, so for an ordinary
    #: deployment this is the only way the work-split settings are knowable at all.
    known_args: list[str] = []

    #: Settings the user states about a server llmbench did not start. No endpoint
    #: reports the physical batch size or whether the cache is unified, and both change
    #: the memory figure, so without these a sliding-window model or a multi-slot server
    #: has no figure at all. A declaration is a claim: it feeds the derived memory
    #: figure, which records its own inputs, and never the identity hash.
    declared: dict[str, Any] = {}

    #: The executable this server was started from, when llmbench started it. Hashed
    #: into the identity so two builds of one commit are told apart.
    binary: Optional[str] = None

    def __init__(self, *args: Any, known_args: Optional[list[str]] = None,
                 declared: Optional[dict[str, Any]] = None,
                 binary: Optional[str] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.known_args = list(known_args or [])
        self.declared = dict(declared or {})
        self.binary = binary

    async def detect(self) -> ModelFingerprint:
        # /props carries the identity - the build, the context, the model path - so a
        # server that cannot answer it has none, and detection stops rather than
        # inventing one out of defaults. /v1/models only adds detail, so it stays
        # optional and its absence is not fatal.
        props = await self._get_required("/props")
        models = await self._get("/v1/models") or await self._get("/models") or {}

        dgs = props.get("default_generation_settings") or {}
        params = dgs.get("params") or {}
        model_path = props.get("model_path")

        # Resolve the routing id + launch args from /v1/models.
        model_id, args, arch = self.model, [], {}
        data = models.get("data") or []
        if data:
            entry = self._match_entry(data)
            model_id = model_id or entry.get("id")
            status = entry.get("status") or {}
            args = status.get("args") or []
            arch = entry.get("architecture") or {}
            model_path = model_path or entry.get("path")
        model_id = model_id or (model_path or "unknown")

        # Arguments we supplied ourselves are used only where the server reported none.
        # A server that does report its argv is describing what it is actually running,
        # which beats what we asked for.
        effective_args = args or self.known_args
        parsed = _parse_args(effective_args)

        # build_info: "b6543-abcdef1"
        build_number, build_commit = None, None
        binfo = props.get("build_info")
        if binfo and "-" in binfo:
            head, _, tail = binfo.partition("-")
            try:
                build_number = int(head.lstrip("b"))
            except ValueError:
                pass
            build_commit = tail or None

        spec_type = parsed.get("spec_type")
        if not spec_type:
            spec_type = "draft" if dgs.get("speculative") else "none"
        mtp = "mtp" in (spec_type or "").lower()

        chat_template = props.get("chat_template")
        tmpl_sha = (
            hashlib.sha256(chat_template.encode()).hexdigest()[:12]
            if chat_template else None
        )

        n_ctx = dgs.get("n_ctx") or _as_int(parsed.get("ctx"))

        # The server's own figure beats the launch argument wherever both exist: -np
        # defaults to auto, and llama-server also rewrites values it cannot honour, so
        # the argument records what was asked for rather than what is running.
        n_parallel = props.get("total_slots") or _as_int(parsed.get("n_parallel"))

        sampling = {
            k: params.get(k) for k in
            ("temperature", "top_k", "top_p", "min_p", "repeat_penalty",
             "samplers", "dry_multiplier", "mirostat")
            if k in params
        }

        name = model_name_from_path(model_path) or model_id
        quant = parse_quant(model_path) or parse_quant(model_id)
        n_params = parse_params(name) or parse_params(model_path)

        # The model file is reachable whenever the server shares a filesystem with the
        # bench. When it is not, the shape is unknown and so is the memory figure.
        shape = read_shape(model_path) if model_path else None
        cache_k = parsed.get("ctk", "f16")
        cache_v = parsed.get("ctv", "f16")
        # Neither of these is reported by any endpoint, and both change the figure, so
        # each is resolved in order of authority and records where it came from.
        observed = bool(effective_args)
        n_ubatch, ubatch_source = _resolve_ubatch(parsed, observed, self.declared)
        unified, unified_source = _resolve_unified(parsed, observed, self.declared)

        pools, ctx_note = _cache_pools(n_ctx, n_parallel, unified)
        ctx_note.update(n_ubatch_source=ubatch_source,
                        kv_unified_source=unified_source)
        # A unified cache is one pool that several sequences share, so it holds room for
        # a window each; a split cache is one pool per sequence, each holding one.
        seqs_sharing = (n_parallel or 1) if ctx_note.get("kv_unified") else 1

        per_pool = kv_cache_bytes(shape, n_ctx, cache_k, cache_v,
                                  n_ubatch=n_ubatch, seqs_sharing=seqs_sharing)
        kv_bytes = per_pool * pools if (per_pool is not None and pools) else None
        kv_derivation = derivation(shape, n_ctx, cache_k, cache_v,
                                   n_ubatch=n_ubatch, seqs_sharing=seqs_sharing)
        kv_derivation.update(ctx_note)
        kv_derivation["kv_cache_bytes_per_pool"] = per_pool
        if kv_bytes is None and kv_derivation.get("known") and has_sliding_window(shape):
            kv_derivation.update(
                known=False,
                reason=("this model has sliding-window layers, whose cache size depends "
                        "on the physical batch size; no endpoint reports it and the "
                        "launch arguments were not observed"),
            )

        return ModelFingerprint(
            engine=self.engine,
            build_number=build_number,
            build_commit=build_commit,
            base_url=self.base_url,
            model_id=model_id,
            model_name=name,
            quant=quant,
            n_params=n_params,
            n_ctx=n_ctx,
            binary_sha=binary_sha(self.binary),
            kv_cache_k=parsed.get("ctk", "f16"),
            kv_cache_v=parsed.get("ctv", "f16"),
            flash_attn=parsed.get("flash_attn"),
            n_gpu_layers=parsed.get("ngl"),
            n_batch=_as_int(parsed.get("n_batch")),
            n_ubatch=_as_int(parsed.get("n_ubatch")),
            n_parallel=n_parallel,
            # Only a served argv makes the three fields above meaningful. llama-server
            # reports it in router mode only, so in an ordinary `-m model.gguf`
            # deployment they are unknown rather than unset.
            launch_settings_observed=bool(effective_args),
            kv_cache_bytes=kv_bytes,
            kv_cache_derivation=kv_derivation,
            spec_type=spec_type,
            draft_model=parsed.get("draft_model"),
            mtp=mtp,
            sampling=sampling,
            chat_template_sha=tmpl_sha,
            launch_args=effective_args,
            raw={"props": props, "architecture": arch},
        )

    def _match_entry(self, data: list[dict]) -> dict:
        if self.model:
            for e in data:
                if e.get("id") == self.model:
                    return e
        # Prefer a loaded model, else the first.
        for e in data:
            status = (e.get("status") or {}).get("value")
            if status == "loaded":
                return e
        return data[0]

    async def count_tokens(self, text: str) -> int:
        try:
            r = await self._client.post(
                f"{self.base_url}/tokenize",
                headers=self._headers,
                json={"content": text, "add_special": False},
            )
            r.raise_for_status()
            toks = r.json().get("tokens") or []
            return len(toks)
        except Exception:
            return approx_tokens(text)
