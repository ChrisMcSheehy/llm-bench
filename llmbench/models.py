"""Domain models shared across the framework.

Everything the orchestrator, evaluators, store and dashboard pass around is
defined here so there is a single source of truth for the schema. Keep these
serialisable (they hit SQLite as JSON) and cheap to construct.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _canonical(obj: Any) -> str:
    """Stable JSON for hashing — sorted keys, no whitespace jitter."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class ModelFingerprint(BaseModel):
    """The identity of a deployed model *as configured right now*.

    This is the label a run is filed under. Two deployments of the same GGUF
    with a different KV-cache quant, a different llama.cpp commit, or different
    sampling defaults produce *different* fingerprints — which is exactly what
    you want when comparing quant mixes or TurboQuant on/off.
    """

    engine: str                      # llama.cpp | ollama | openrouter
    engine_version: Optional[str] = None
    build_number: Optional[int] = None
    build_commit: Optional[str] = None

    base_url: str
    model_id: str                    # the id/alias the API routes on
    model_name: Optional[str] = None  # human name parsed from path/id
    quant: Optional[str] = None       # Q4_K_M, IQ4_XS, MXFP4, ...
    n_params: Optional[str] = None    # "8B", "235B-A22B", ...

    n_ctx: Optional[int] = None
    kv_cache_k: Optional[str] = None  # f16, q8_0, q4_0 (TurboQuant lives here)
    kv_cache_v: Optional[str] = None
    flash_attn: Optional[str] = None  # on | off | auto

    # How the work is divided up. All four move the numbers: layers on the card versus
    # the processor run separately-written implementations of the same mathematics, and
    # batching changes the order values are summed in.
    n_gpu_layers: Optional[str] = None   # -ngl. Text, because llama.cpp also takes
                                         # "auto" and "all", which int() would destroy.
    n_batch: Optional[int] = None        # -b   logical batch size
    n_ubatch: Optional[int] = None       # -ub  physical batch size
    n_parallel: Optional[int] = None     # -np  server slots

    # Which executable produced these results, as a short hash of the file itself.
    # /props reports the build commit, and two builds of one commit are routinely
    # different programs - a Vulkan build against a ROCm one, or a fork before it is
    # rebased. Without this they share a fingerprint and get pooled, averaging away the
    # difference the run was measuring. Hashed rather than named because a path is not
    # an identity: the same build lives in different folders on different machines, and
    # one path is overwritten by every rebuild. None means llmbench did not start this
    # server and so never saw its binary - unknown, which is its own state (design D6a).
    binary_sha: Optional[str] = None

    # Whether the backend told us its launch settings at all. Without this, the four
    # fields above are ambiguous: None reads as "not set" when it may mean "not
    # reported", and two deployments with genuinely different layer splits then hash
    # identically. False is the honest default — absent a statement, we do not know.
    launch_settings_observed: bool = False

    # What this configuration costs in memory. Derived, not measured, and deliberately
    # NOT part of _identity_dict: a better formula must not fork a configuration's
    # history. None means unknown and must never be displayed as 0.
    kv_cache_bytes: Optional[int] = None
    kv_cache_derivation: dict[str, Any] = Field(default_factory=dict)

    spec_type: Optional[str] = None   # none | draft-mtp | draft-eagle3 | ngram-* ...
    draft_model: Optional[str] = None
    mtp: bool = False                 # convenience flag: spec_type involves MTP

    sampling: dict[str, Any] = Field(default_factory=dict)
    chat_template_sha: Optional[str] = None
    launch_args: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)  # full /props etc. for forensics

    detected_at: datetime = Field(default_factory=_utcnow)

    # ---- derived identity -------------------------------------------------
    def _identity_dict(self) -> dict[str, Any]:
        """The subset that defines *sameness*. Deliberately excludes raw dumps,
        timestamps and base_url so the same config on a different port collides."""
        return {
            "engine": self.engine,
            "build_commit": self.build_commit,
            "binary_sha": self.binary_sha,
            "model_id": self.model_id,
            "quant": self.quant,
            "n_params": self.n_params,
            "n_ctx": self.n_ctx,
            "kv_cache_k": self.kv_cache_k,
            "kv_cache_v": self.kv_cache_v,
            "flash_attn": self.flash_attn,
            "n_gpu_layers": self.n_gpu_layers,
            "n_batch": self.n_batch,
            "n_ubatch": self.n_ubatch,
            "n_parallel": self.n_parallel,
            "launch_settings_observed": self.launch_settings_observed,
            "spec_type": self.spec_type,
            "draft_model": self.draft_model,
            "sampling": self.sampling,
            "chat_template_sha": self.chat_template_sha,
        }

    @property
    def fingerprint_hash(self) -> str:
        return hashlib.sha256(_canonical(self._identity_dict()).encode()).hexdigest()[:16]

    @property
    def label(self) -> str:
        """Short human label for tables/legends."""
        bits = [self.model_name or self.model_id]
        if self.quant:
            bits.append(self.quant)
        kv = self._kv_label()
        if kv:
            bits.append(f"kv:{kv}")
        if self.n_gpu_layers:
            bits.append(f"ngl:{self.n_gpu_layers}")
        work = self._work_split_label()
        if work:
            bits.append(work)
        if not self.launch_settings_observed:
            # Says why the settings above are missing rather than leaving the reader to
            # assume they were at their defaults.
            bits.append("launch:unreported")
        if self.mtp or (self.spec_type and self.spec_type != "none"):
            bits.append(self.spec_type or "spec")
        if self.build_commit:
            bits.append(self.build_commit[:7])
        elif self.engine_version:
            bits.append(f"v{self.engine_version}")
        if self.binary_sha:
            # Shown because the commit alone does not identify the program: two builds
            # of one commit are two rows that would otherwise read identically, and a
            # reader comparing forks could not tell which was which (design D7).
            bits.append(f"bin:{self.binary_sha[:6]}")
        return " · ".join(str(b) for b in bits if b)

    def _kv_label(self) -> Optional[str]:
        if not self.kv_cache_k and not self.kv_cache_v:
            return None
        if self.kv_cache_k == self.kv_cache_v:
            return self.kv_cache_k
        return f"{self.kv_cache_k}/{self.kv_cache_v}"

    def _work_split_label(self) -> Optional[str]:
        """Batch and slot settings, shown only where the launch actually set them.

        Almost no launch sets these, so listing them unconditionally would add three
        empty tokens to every label on every screen.
        """
        parts = [f"{flag}:{value}" for flag, value in
                 (("b", self.n_batch), ("ub", self.n_ubatch), ("np", self.n_parallel))
                 if value is not None]
        return " ".join(parts) or None


class HostFingerprint(BaseModel):
    """The machine a run happened on.

    Separate from ModelFingerprint on purpose (design D1): one machine runs many
    configurations and one configuration runs on many machines, so the host is an
    entity joined at the run rather than a field on the model's identity. The model's
    fingerprint hash keeps its exact existing meaning, so every comparison and stored
    vote that already exists keeps working.
    """

    os: str
    os_release: Optional[str] = None      # recorded, never hashed: a point upgrade is
                                          # not a different machine
    arch: str
    cpu_count: int = 1
    cpu_model: Optional[str] = None       # declared by the user; see D2a decision 4
    total_memory_bytes: Optional[int] = None
    devices: list[dict[str, Any]] = Field(default_factory=list)
    declared: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime = Field(default_factory=_utcnow)

    def _identity_dict(self) -> dict[str, Any]:
        """Only what makes two machines genuinely different.

        Free memory is excluded because it differs between two runs on an idle
        machine, and total memory is rounded to whole gibibytes because firmware can
        reserve a little more or less without the machine having changed.
        """
        gib = (round(self.total_memory_bytes / 1024 ** 3)
               if self.total_memory_bytes else None)
        return {
            "os": self.os,
            "arch": self.arch,
            "cpu_count": self.cpu_count,
            "total_memory_gib": gib,
            "devices": [
                {"backend": d.get("backend"), "name": d.get("name"),
                 "total_mib": d.get("total_mib")}
                for d in self.devices
            ],
        }

    @property
    def host_hash(self) -> str:
        return hashlib.sha256(_canonical(self._identity_dict()).encode()).hexdigest()[:16]

    @property
    def label(self) -> str:
        """One line, e.g. 'Linux x86_64 - AMD Radeon RX 7900 XTX (Vulkan)'."""
        head = f"{self.os} {self.arch}"
        if not self.devices:
            return f"{head} - no device information"
        first = self.devices[0]
        extra = f" +{len(self.devices) - 1}" if len(self.devices) > 1 else ""
        return f"{head} - {first.get('name')} ({first.get('backend')}){extra}"


class Sample(BaseModel):
    """One graded interaction. The atomic row in the store."""

    evaluator: str
    case_id: str
    group: Optional[str] = None                 # coarse bucket for aggregation
    dims: dict[str, Any] = Field(default_factory=dict)  # context_len, depth_pct, problem...

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    tok_per_sec: Optional[float] = None
    server_prompt_tps: Optional[float] = None   # llama.cpp reported prompt-eval speed
    server_gen_tps: Optional[float] = None      # llama.cpp reported gen speed

    score: Optional[float] = None               # 0..1
    passed: Optional[bool] = None
    error: Optional[str] = None
    # Why this was never attempted, or None if it was. A skip is not a failure: a
    # machine that cannot hold a 512k context has an honest limit, not a broken run
    # (design D3). It is also never a score of zero - there is no measurement here.
    skipped: Optional[str] = None
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class Metric(BaseModel):
    """An aggregate produced by an evaluator over its samples."""

    evaluator: str
    name: str
    value: float
    unit: Optional[str] = None
    # How many graded items this figure rests on, set by the same expression that
    # computed the value. Deriving it later would re-implement the aggregator's filter
    # in SQL, and the two could then disagree without anyone noticing (design D7a).
    # None means the aggregator did not say, and is displayed as a dash - never as a
    # zero, because an unstated count is not a count of nothing.
    n: Optional[int] = None
    dims: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    run_id: str
    fingerprint: ModelFingerprint
    suite: str
    status: str = "running"           # running | ok | partial | error
    started_at: datetime = Field(default_factory=_utcnow)
    finished_at: Optional[datetime] = None
    samples: list[Sample] = Field(default_factory=list)
    metrics: list[Metric] = Field(default_factory=list)
    error: Optional[str] = None


# ---- shared parsing helpers (used by target adapters) --------------------

_QUANT_RE = re.compile(
    r"(IQ\d+_[A-Z]+(?:_[A-Z]+)?|Q\d+_K_[A-Z]+|Q\d+_K|Q\d+_\d+|Q\d+|MXFP4|BF16|F16|F32)",
    re.IGNORECASE,
)
_PARAMS_RE = re.compile(r"(\d+(?:\.\d+)?[BbMm](?:-A\d+(?:\.\d+)?[BbMm])?)")


def parse_quant(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = _QUANT_RE.search(text)
    return m.group(1).upper() if m else None


def parse_params(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = _PARAMS_RE.search(text)
    return m.group(1).upper() if m else None


def model_name_from_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    base = re.split(r"[\\/]", path)[-1]
    base = re.sub(r"\.gguf$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"-\d+-of-\d+$", "", base)  # strip shard suffix
    return base
