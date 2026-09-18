"""Throughput measurement. Replaces runtime guesses with a number.

Samples documents proportionally to the real corpus mix (20% posts / 80% replies)
so the extrapolation isn't skewed by post length, and runs them at the configured
parallelism without touching the job queue.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass

from pipeline.core.config import Config
from pipeline.core.db import connect
from pipeline.core.models import extraction_schema
from pipeline.core.ollama import OllamaClient
from pipeline.core.prompts import render_document, system_prompt

SAMPLE_SQL = """
(SELECT d.doc_id, d.doc_type, d.author_id, d.title, d.body,
        NULL::text AS parent_title
   FROM documents d WHERE d.doc_type='post' ORDER BY md5(d.doc_id) LIMIT %(posts)s)
UNION ALL
(SELECT d.doc_id, d.doc_type, d.author_id, d.title, d.body,
        p.title
   FROM documents d JOIN documents p ON p.doc_id = d.parent_id
  WHERE d.doc_type='reply' ORDER BY md5(d.doc_id) LIMIT %(replies)s)
"""


@dataclass
class BenchResult:
    n: int
    parallel: int
    wall_s: float
    prompt_tokens: int
    completion_tokens: int
    latencies_ms: list[float]
    corpus_total: int

    @property
    def docs_per_s(self) -> float:
        return self.n / self.wall_s if self.wall_s else 0.0

    @property
    def completion_tps(self) -> float:
        return self.completion_tokens / self.wall_s if self.wall_s else 0.0

    @property
    def eta_s(self) -> float:
        return self.corpus_total / self.docs_per_s if self.docs_per_s else 0.0

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[min(int(0.95 * len(s)), len(s) - 1)]


async def run(cfg: Config, n: int | None = None) -> BenchResult:
    n = n or cfg.bench_n
    # Match the corpus mix: 2001 posts / 7990 replies ~= 1:4.
    posts = max(1, round(n * 0.2))
    replies = max(1, n - posts)

    with connect(cfg) as conn:
        cur = conn.execute(SAMPLE_SQL, {"posts": posts, "replies": replies})
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        corpus_total = conn.execute("SELECT count(*) FROM documents").fetchone()[0]

    schema = extraction_schema()
    sem = asyncio.Semaphore(cfg.ollama_num_parallel)
    lat: list[float] = []
    pt = ct = 0

    async def one(row: dict) -> None:
        nonlocal pt, ct
        prompt = render_document(
            doc_type=row["doc_type"], author_id=row["author_id"], title=row["title"],
            body=row["body"], parent_title=row["parent_title"],
        )
        async with sem:
            t0 = time.perf_counter()
            g = await client.generate(prompt, schema=schema, system=system_prompt())
            lat.append((time.perf_counter() - t0) * 1000)
            pt += g.prompt_tokens
            ct += g.completion_tokens

    async with OllamaClient(cfg) as client:
        # One warm-up call so model load time doesn't land in the measurement.
        await one(rows[0])
        lat.clear()
        pt = ct = 0
        t0 = time.perf_counter()
        await asyncio.gather(*(one(r) for r in rows))
        wall = time.perf_counter() - t0

    return BenchResult(
        n=len(rows), parallel=cfg.ollama_num_parallel, wall_s=wall,
        prompt_tokens=pt, completion_tokens=ct, latencies_ms=lat, corpus_total=corpus_total,
    )
