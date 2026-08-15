"""OpenRouter target (the "if it comes to it" cloud fallback / baseline).

Detection is thin — the point of OpenRouter here is to give you a reference
line on the same charts as your local runs. Token counts come from the API's
usage field after the fact, or a heuristic when sizing prompts up front.
"""
from __future__ import annotations

from typing import Optional

from llmbench.models import ModelFingerprint, parse_params, parse_quant
from llmbench.targets.base import Target, approx_tokens

OPENROUTER_BASE = "https://openrouter.ai/api"


class OpenRouterTarget(Target):
    engine = "openrouter"

    def __init__(self, model: str, api_key: str, base_url: str = OPENROUTER_BASE,
                 **kw):
        super().__init__(base_url=base_url, model=model, api_key=api_key, **kw)

    async def detect(self) -> ModelFingerprint:
        meta = {}
        catalog = await self._get("/v1/models") or {}
        for m in catalog.get("data", []):
            if m.get("id") == self.model:
                meta = m
                break
        ctx = meta.get("context_length")
        return ModelFingerprint(
            engine=self.engine,
            base_url=self.base_url,
            model_id=self.model,
            model_name=meta.get("name") or self.model,
            quant=parse_quant(self.model),
            n_params=parse_params(meta.get("name") or self.model),
            n_ctx=int(ctx) if ctx else None,
            raw={"catalog": meta},
        )

    async def count_tokens(self, text: str) -> int:
        return approx_tokens(text)
