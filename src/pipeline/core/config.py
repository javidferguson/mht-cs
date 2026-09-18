"""Single source of truth for runtime configuration.

Everything is env-driven so the same image runs unchanged on a Mac (host-native
Ollama) and on EC2 (containerised Ollama). OLLAMA_BASE_URL is the only knob that
differs between the two.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _opt_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else None


@dataclass(frozen=True)
class Config:
    ollama_base_url: str
    extraction_model: str
    ollama_num_parallel: int
    ollama_num_ctx: int
    ollama_num_predict: int
    ollama_timeout_s: int
    input_dir: Path
    out_dir: Path
    pg_dsn: str
    bench_n: int
    limit: int | None

    @classmethod
    def from_env(cls) -> "Config":
        user = os.getenv("POSTGRES_USER", "mht")
        pwd = os.getenv("POSTGRES_PASSWORD", "mht")
        host = os.getenv("POSTGRES_HOST", "postgres")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "mht")
        return cls(
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
            ).rstrip("/"),
            extraction_model=os.getenv("EXTRACTION_MODEL", "llama3.1:8b"),
            ollama_num_parallel=_int("OLLAMA_NUM_PARALLEL", 8),
            ollama_num_ctx=_int("OLLAMA_NUM_CTX", 4096),
            ollama_num_predict=_int("OLLAMA_NUM_PREDICT", 1536),
            ollama_timeout_s=_int("OLLAMA_TIMEOUT_S", 180),
            input_dir=Path(os.getenv("INPUT_DIR", "/data")),
            out_dir=Path(os.getenv("OUT_DIR", "/out")),
            pg_dsn=f"postgresql://{user}:{pwd}@{host}:{port}/{db}",
            bench_n=_int("BENCH_N", 20),
            limit=_opt_int("LIMIT"),
        )

    @property
    def pg_dsn_redacted(self) -> str:
        user = os.getenv("POSTGRES_USER", "mht")
        host = os.getenv("POSTGRES_HOST", "postgres")
        port = os.getenv("POSTGRES_PORT", "5432")
        db = os.getenv("POSTGRES_DB", "mht")
        return f"postgresql://{user}:***@{host}:{port}/{db}"
