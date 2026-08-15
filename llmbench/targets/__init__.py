"""Target factory. `build_target(spec)` -> a ready Target instance."""
from __future__ import annotations

import os
from typing import Any

from llmbench.targets.base import GenResult, Target
from llmbench.targets.llamacpp import LlamaCppTarget
from llmbench.targets.ollama import OllamaTarget
from llmbench.targets.openrouter import OpenRouterTarget

_ENGINES = {
    "llama.cpp": LlamaCppTarget,
    "llamacpp": LlamaCppTarget,
    "ollama": OllamaTarget,
    "openrouter": OpenRouterTarget,
}


def build_target(spec: dict[str, Any]) -> Target:
    """spec keys: engine, base_url, model (optional), api_key / api_key_env."""
    engine = spec["engine"].lower()
    if engine not in _ENGINES:
        raise ValueError(f"Unknown engine {engine!r}. Known: {sorted(_ENGINES)}")
    cls = _ENGINES[engine]

    api_key = spec.get("api_key")
    if not api_key and spec.get("api_key_env"):
        api_key = os.environ.get(spec["api_key_env"])

    kw: dict[str, Any] = {"model": spec.get("model")}
    if api_key:
        kw["api_key"] = api_key
    if spec.get("timeout"):
        kw["timeout"] = float(spec["timeout"])
    if spec.get("declared") and cls is LlamaCppTarget:
        # What the user states about a server we did not start: the physical batch size
        # and whether the cache is unified, neither of which any endpoint reports. Used
        # for the memory figure only, never for the identity.
        kw["declared"] = dict(spec["declared"])
    if spec.get("known_args") and cls is LlamaCppTarget:
        # Set only when llmbench started this server itself. llama-server does not
        # report its own argv outside router mode, so without this the settings that
        # decide the identity are simply unknown.
        kw["known_args"] = list(spec["known_args"])
    if spec.get("binary") and cls is LlamaCppTarget:
        # Also set only when llmbench started the server, and for the same reason: the
        # executable is knowable precisely because we ran it. Its hash goes into the
        # identity so two builds of one commit are two configurations.
        kw["binary"] = spec["binary"]

    if engine == "openrouter":
        return OpenRouterTarget(model=spec["model"], api_key=api_key or "",
                                base_url=spec.get("base_url", "https://openrouter.ai/api"),
                                timeout=kw.get("timeout", 600.0))
    return cls(base_url=spec["base_url"], **kw)


__all__ = ["Target", "GenResult", "build_target",
           "LlamaCppTarget", "OllamaTarget", "OpenRouterTarget"]
