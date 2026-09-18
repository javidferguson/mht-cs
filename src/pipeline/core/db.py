"""Postgres access. The job queue lives here too (claimed with SKIP LOCKED),
which is why there is no Redis or Celery in this stack.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib import resources
from typing import Iterator

import psycopg

from pipeline.core.config import Config


@contextmanager
def connect(cfg: Config) -> Iterator[psycopg.Connection]:
    with psycopg.connect(cfg.pg_dsn, autocommit=True) as conn:
        yield conn


def server_version(cfg: Config) -> str:
    with connect(cfg) as conn:
        return conn.execute("SELECT version()").fetchone()[0]


def init_schema(cfg: Config) -> None:
    """Apply schema.sql. Idempotent - every statement is IF NOT EXISTS."""
    ddl = resources.files("pipeline.core").joinpath("schema.sql").read_text()
    with connect(cfg) as conn:
        conn.execute(ddl)


def table_exists(cfg: Config, name: str) -> bool:
    with connect(cfg) as conn:
        return bool(
            conn.execute("SELECT to_regclass(%s) IS NOT NULL", (f"public.{name}",)).fetchone()[0]
        )
