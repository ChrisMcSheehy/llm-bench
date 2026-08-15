"""Ollama target.

Ollama serves an OpenAI-compatible endpoint at /v1 (used for generation) plus
its own management API:
  GET  /api/version -> engine version
  GET  /api/tags    -> installed models with details.{quantization_level,parameter_size,family}
  POST /api/show    -> model_info (context length, arch), template, details

Ollama has no /tokenize endpoint, so context sizing falls back to a heuristic.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

from llmbench.models import ModelFingerprint, parse_params, parse_quant
from llmbench.targets.base import Target, approx_tokens


class OllamaTarget(Target):
    engine = "ollama"

    async def _post(self, path: str, payload: dict) -> Optional[dict]:
        try:
            r = await self._client.post(
                f"{self.base_url}{path}", headers=self._headers, json=payload)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    async def detect(self) -> ModelFingerprint:
        # The tag list is what names the model, so an unreachable server has no identity
        # and detection stops. The version string is a nicety and may be absent.
        tags = await self._get_required("/api/tags")
        version = (await self._get("/api/version") or {}).get("version")

        model_id = self.model
        if not model_id:
            names = [m.get("name") for m in tags.get("models", []) if m.get("name")]
            model_id = names[0] if names else "unknown"

        show = await self._post("/api/show", {"model": model_id}) or {}
        details = show.get("details") or {}
        model_info = show.get("model_info") or {}
        template = show.get("template")

        # context length key is arch-scoped, e.g. "qwen3.context_length"
        n_ctx = None
        for k, v in model_info.items():
            if k.endswith(".context_length"):
                n_ctx = v
                break

        quant = details.get("quantization_level") or parse_quant(model_id)
        n_params = details.get("parameter_size") or parse_params(model_id)
        family = details.get("family")

        tmpl_sha = (
            hashlib.sha256(template.encode()).hexdigest()[:12] if template else None
        )

        return ModelFingerprint(
            engine=self.engine,
            engine_version=version,
            base_url=self.base_url,
            model_id=model_id,
            model_name=model_id.split(":")[0],
            quant=quant,
            n_params=n_params,
            n_ctx=int(n_ctx) if n_ctx else None,
            chat_template_sha=tmpl_sha,
            raw={"details": details, "model_info": model_info, "family": family},
        )

    async def count_tokens(self, text: str) -> int:
        return approx_tokens(text)
