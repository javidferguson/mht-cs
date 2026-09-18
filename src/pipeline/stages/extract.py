"""Stage 2: documents -> extractions, via schema-constrained Ollama calls.

Concurrency model: claim a batch of jobs with FOR UPDATE SKIP LOCKED, run them
through an asyncio semaphore sized to OLLAMA_NUM_PARALLEL, write results, repeat.
Multiple worker containers can run this safely against the same queue.

Resumability comes from two places: the jobs table (status/attempts) and the
extractions primary key, which doubles as a content-hash cache.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from dataclasses import dataclass, field

from pipeline.core.config import Config
from pipeline.core.db import connect
from pipeline.core.models import extraction_schema
from pipeline.core.ollama import OllamaClient
from pipeline.core.prompts import PROMPT_VERSION, render_document, system_prompt

MAX_ATTEMPTS = 3
STAGE = "extract"


@dataclass
class ExtractStats:
    processed: int = 0
    cached: int = 0
    failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_s: float = 0.0
    latencies_ms: list[int] = field(default_factory=list)


CLAIM_SQL = """
WITH candidate AS (
    SELECT doc_id FROM jobs
     WHERE stage = %(stage)s AND status = 'pending' AND attempts < %(max_attempts)s
     -- md5 rather than doc_id: plain doc_id ordering claims every 'post_...'
     -- before any 'reply_...', so a LIMIT run would only ever see posts - and
     -- replies are four fifths of the corpus and carry the hard attribution
     -- cases. (No literal percent signs here: psycopg reads them as
     -- placeholders, even inside a SQL comment.)
     -- Deterministic, so resumability is unaffected.
     ORDER BY md5(doc_id)
     LIMIT %(n)s
     FOR UPDATE SKIP LOCKED
)
UPDATE jobs j
   SET status = 'running', locked_at = now(), locked_by = %(worker)s,
       attempts = j.attempts + 1, updated_at = now()
  FROM candidate c
 WHERE j.doc_id = c.doc_id AND j.stage = %(stage)s
RETURNING j.doc_id
"""

FETCH_SQL = """
SELECT d.doc_id, d.doc_type, d.author_id, d.title, d.body, d.content_hash,
       p.title AS parent_title
  FROM documents d
  LEFT JOIN documents p ON p.doc_id = d.parent_id
 WHERE d.doc_id = ANY(%s)
"""


def _cached_ids(conn, doc_ids: list[str], model: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT e.doc_id FROM extractions e
          JOIN documents d ON d.doc_id = e.doc_id AND d.content_hash = e.content_hash
         WHERE e.doc_id = ANY(%s) AND e.prompt_version = %s AND e.model = %s
           AND e.parsed_ok
        """,
        (doc_ids, PROMPT_VERSION, model),
    ).fetchall()
    return {r[0] for r in rows}


async def _one(client: OllamaClient, sem: asyncio.Semaphore, row: dict, schema: dict) -> dict:
    prompt = render_document(
        doc_type=row["doc_type"],
        author_id=row["author_id"],
        title=row["title"],
        body=row["body"],
        parent_title=row["parent_title"],
    )
    async with sem:
        t0 = time.perf_counter()
        try:
            g = await client.generate(prompt, schema=schema, system=system_prompt())
            latency = int((time.perf_counter() - t0) * 1000)
            try:
                parsed = g.parsed
            except json.JSONDecodeError:
                # Greedy decoding occasionally walks into a repetition loop - one
                # document emitted "\u00a0" thousands of times until num_predict
                # truncated it mid-escape. temperature alone does not reliably
                # break that; repeat_penalty targets it directly.
                g = await client.generate(
                    prompt, schema=schema, system=system_prompt(),
                    options={"temperature": 0.4, "repeat_penalty": 1.3},
                )
                latency = int((time.perf_counter() - t0) * 1000)
                try:
                    parsed = g.parsed
                except json.JSONDecodeError as e:
                    return {**row, "ok": False, "error": f"unparseable JSON after retry: {e}",
                            "raw": g.text, "pt": g.prompt_tokens, "ct": g.completion_tokens,
                            "latency": latency}
            return {**row, "ok": True, "error": None, "raw": parsed,
                    "pt": g.prompt_tokens, "ct": g.completion_tokens, "latency": latency}
        except Exception as e:  # noqa: BLE001
            return {**row, "ok": False, "error": f"{type(e).__name__}: {e}", "raw": None,
                    "pt": 0, "ct": 0, "latency": int((time.perf_counter() - t0) * 1000)}


def _persist(conn, res: dict, model: str) -> None:
    conn.execute(
        """
        INSERT INTO extractions (doc_id, content_hash, prompt_version, model, raw_json,
                                 parsed_ok, error, prompt_tokens, completion_tokens, latency_ms)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (doc_id, content_hash, prompt_version, model) DO UPDATE SET
            raw_json = EXCLUDED.raw_json, parsed_ok = EXCLUDED.parsed_ok,
            error = EXCLUDED.error, prompt_tokens = EXCLUDED.prompt_tokens,
            completion_tokens = EXCLUDED.completion_tokens,
            latency_ms = EXCLUDED.latency_ms, created_at = now()
        """,
        (res["doc_id"], res["content_hash"], PROMPT_VERSION, model,
         json.dumps(res["raw"]) if res["ok"] else None,
         res["ok"], res["error"], res["pt"], res["ct"], res["latency"]),
    )
    conn.execute(
        """
        UPDATE jobs SET status = %s, error = %s, locked_at = NULL, updated_at = now()
         WHERE doc_id = %s AND stage = %s
        """,
        ("done" if res["ok"] else "error", res["error"], res["doc_id"], STAGE),
    )


async def run(cfg: Config, limit: int | None = None, progress=None,
              doc_ids: list[str] | None = None) -> ExtractStats:
    """Run N independent worker coroutines against the queue.

    Each worker claims ONE job, processes it and immediately claims another. An
    earlier version claimed a batch of N and awaited the whole batch before
    claiming again - a convoy effect where every slot idled waiting for the
    slowest document in its batch. With p50 8.5s and p95 16.7s latency that
    halved sustained throughput (0.27 docs/s vs 0.55 for the same work).
    """
    stats = ExtractStats()
    schema = extraction_schema()
    worker_id = f"{socket.gethostname()}:{id(stats)}"
    n_workers = max(cfg.ollama_num_parallel, 1)
    claimed_n = 0
    sem = asyncio.Semaphore(n_workers)
    lock = asyncio.Lock()
    t_start = time.perf_counter()

    reclaim_stale(cfg)

    if doc_ids:
        # Targeted re-run for debugging a specific document.
        with connect(cfg) as conn:
            conn.execute(
                "UPDATE jobs SET status='pending', attempts=0, locked_at=NULL"
                " WHERE stage=%s AND doc_id = ANY(%s)",
                (STAGE, doc_ids),
            )

    async def claim_one() -> dict | None:
        """Claim a single job and return its document row, or None when drained."""
        nonlocal claimed_n
        async with lock:
            # Counted at CLAIM time. Checking processed+cached instead lets all
            # N workers pass the test before any of them has finished.
            if limit is not None and claimed_n >= limit:
                return None
            with connect(cfg) as conn:
                sql = CLAIM_SQL if not doc_ids else CLAIM_SQL.replace(
                    "AND status = 'pending'",
                    "AND status = 'pending' AND doc_id = ANY(%(docs)s)",
                )
                got = conn.execute(
                    sql,
                    {"stage": STAGE, "n": 1, "worker": worker_id,
                     "max_attempts": MAX_ATTEMPTS, "docs": doc_ids},
                ).fetchone()
                if not got:
                    return None
                doc_id = got[0]
                claimed_n += 1

                if _cached_ids(conn, [doc_id], cfg.extraction_model):
                    conn.execute(
                        "UPDATE jobs SET status='done', error=NULL, locked_at=NULL,"
                        " updated_at=now() WHERE stage=%s AND doc_id=%s",
                        (STAGE, doc_id),
                    )
                    stats.cached += 1
                    return {"__cached__": True}

                cur = conn.execute(FETCH_SQL, ([doc_id],))
                cols = [c.name for c in cur.description]
                row = cur.fetchone()
                return dict(zip(cols, row)) if row else None

    async def worker() -> None:
        while True:
            row = await claim_one()
            if row is None:
                return
            if row.get("__cached__"):
                if progress:
                    progress(stats)
                continue
            res = await _one(client, sem, row, schema)
            with connect(cfg) as conn:
                _persist(conn, res, cfg.extraction_model)
            if res["ok"]:
                stats.processed += 1
                stats.prompt_tokens += res["pt"]
                stats.completion_tokens += res["ct"]
                stats.latencies_ms.append(res["latency"])
            else:
                stats.failed += 1
            stats.wall_s = time.perf_counter() - t_start
            if progress:
                progress(stats)

    async with OllamaClient(cfg) as client:
        await asyncio.gather(*(worker() for _ in range(n_workers)))

    stats.wall_s = time.perf_counter() - t_start
    return stats


def reclaim_stale(cfg: Config, older_than_s: int = 900) -> int:
    """Return jobs abandoned by a crashed worker to the queue.

    Without this a killed worker leaves its in-flight jobs 'running' forever and
    they are never retried.
    """
    with connect(cfg) as conn:
        return conn.execute(
            """UPDATE jobs SET status='pending', locked_at=NULL, locked_by=NULL,
                      updated_at=now()
                WHERE stage=%s AND status='running'
                  AND locked_at < now() - make_interval(secs => %s)""",
            (STAGE, older_than_s),
        ).rowcount


def queue_counts(cfg: Config) -> dict[str, int]:
    with connect(cfg) as conn:
        rows = conn.execute(
            "SELECT status, count(*) FROM jobs WHERE stage=%s GROUP BY status", (STAGE,)
        ).fetchall()
    return {s: n for s, n in rows}
