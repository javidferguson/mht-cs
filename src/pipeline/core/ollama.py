"""Thin async Ollama client.

Only the pieces the pipeline needs: model listing, a schema-constrained generate,
and token accounting. Ollama's `format` parameter takes a JSON Schema and
constrains decoding to it, so extraction output is parseable by construction
rather than by regex salvage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from pipeline.core.config import Config


class OllamaError(RuntimeError):
    pass


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_duration_ns: int

    @property
    def parsed(self) -> Any:
        return json.loads(self.text)


class OllamaClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=cfg.ollama_base_url,
            timeout=httpx.Timeout(cfg.ollama_timeout_s, connect=10.0),
        )

    async def __aenter__(self) -> "OllamaClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def list_models(self) -> list[dict[str, Any]]:
        r = await self._client.get("/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])

    async def has_model(self, name: str) -> bool:
        # Ollama reports "llama3.1:8b"; tolerate the user omitting the ":latest" tag.
        wanted = name if ":" in name else f"{name}:latest"
        return any(m.get("name") == wanted for m in await self.list_models())

    async def generate(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> Generation:
        payload: dict[str, Any] = {
            "model": self.cfg.extraction_model,
            "prompt": prompt,
            "stream": False,
            # Reasoning models (qwen3, deepseek-r1, ...) otherwise route all output
            # into a separate "thinking" field and return an EMPTY "response",
            # which surfaces as a confusing JSONDecodeError. Extraction wants the
            # answer, not the reasoning. Ignored by non-reasoning models.
            "think": False,
            # temperature 0: extraction should be as reproducible as the model allows.
            # num_ctx: the model default (131072) reserves ~22GB of KV cache per
            # slot, which makes OLLAMA_NUM_PARALLEL>1 impossible on a 64GB machine.
            # num_predict bounds generation. Two documents in the full corpus put
            # the model into a loop that never terminated - 420s with no response.
            # Extraction output runs 100-250 tokens, so this cap is generous, and a
            # truncated (unparseable) result recorded in seconds beats a hang that
            # burns the whole timeout and blocks a worker slot.
            "options": {
                "temperature": 0,
                "num_ctx": self.cfg.ollama_num_ctx,
                "num_predict": self.cfg.ollama_num_predict,
                **(options or {}),
            },
        }
        if system:
            payload["system"] = system
        if schema is not None:
            payload["format"] = schema

        r = await self._client.post("/api/generate", json=payload)
        if r.status_code == 404:
            raise OllamaError(
                f"model {self.cfg.extraction_model!r} not found on the Ollama server. "
                f"Run: make pull-model"
            )
        r.raise_for_status()
        body = r.json()
        return Generation(
            text=body.get("response", ""),
            prompt_tokens=body.get("prompt_eval_count", 0),
            completion_tokens=body.get("eval_count", 0),
            total_duration_ns=body.get("total_duration", 0),
        )
