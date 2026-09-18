"""Stage 1: JSONL -> documents + seeded job queue.

Idempotent by design. A document is keyed on doc_id and carries a content_hash;
re-running over unchanged input is a no-op, and a document whose body changed has
its downstream jobs reset to pending so only the affected work is redone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from pipeline.core.config import Config
from pipeline.core.db import connect

CHUNK = 1000
STAGES = ("extract",)


@dataclass
class IngestResult:
    files: list[str] = field(default_factory=list)
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    jobs_seeded: int = 0
    jobs_reset: int = 0

    @property
    def total(self) -> int:
        return self.inserted + self.updated + self.unchanged


def content_hash(title: str | None, body: str) -> str:
    return hashlib.sha256(f"{title or ''}\x00{body}".encode()).hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}:{lineno} is not valid JSON: {e}") from e


def _to_row(rec: dict[str, Any], source: str) -> tuple:
    doc_type = rec["type"]
    title = rec.get("title")
    body = rec["body"]
    return (
        rec["id"],
        doc_type,
        rec["author_id"],
        rec.get("parent_id"),
        title,
        body,
        rec["created_at"],
        content_hash(title, body),
        source,
    )


def ingest(cfg: Config) -> IngestResult:
    files = sorted(cfg.input_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no *.jsonl under {cfg.input_dir} (check DATA_DIR)")

    rows: list[tuple] = []
    for path in files:
        rows.extend(_to_row(rec, path.name) for rec in _read_jsonl(path))

    # Posts must land before replies: documents.parent_id is a self-referencing FK,
    # so the database rejects an orphan reply rather than silently storing one.
    rows.sort(key=lambda r: 0 if r[1] == "post" else 1)

    res = IngestResult(files=[p.name for p in files])
    changed: list[str] = []

    upsert = """
        INSERT INTO documents
            (doc_id, doc_type, author_id, parent_id, title, body,
             created_at, content_hash, source_file)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (doc_id) DO UPDATE SET
            doc_type     = EXCLUDED.doc_type,
            author_id    = EXCLUDED.author_id,
            parent_id    = EXCLUDED.parent_id,
            title        = EXCLUDED.title,
            body         = EXCLUDED.body,
            created_at   = EXCLUDED.created_at,
            content_hash = EXCLUDED.content_hash,
            source_file  = EXCLUDED.source_file,
            ingested_at  = now()
        -- Rows whose content is identical are not touched and not returned,
        -- which is what makes a re-run a true no-op.
        WHERE documents.content_hash IS DISTINCT FROM EXCLUDED.content_hash
        RETURNING doc_id, (xmax = 0) AS was_insert
    """

    with connect(cfg) as conn, conn.cursor() as cur:
        for i in range(0, len(rows), CHUNK):
            for row in rows[i : i + CHUNK]:
                cur.execute(upsert, row)
                out = cur.fetchone()
                if out is None:
                    res.unchanged += 1
                    continue
                doc_id, was_insert = out
                changed.append(doc_id)
                if was_insert:
                    res.inserted += 1
                else:
                    res.updated += 1

        for stage in STAGES:
            seeded = cur.execute(
                """
                INSERT INTO jobs (doc_id, stage)
                SELECT doc_id, %s FROM documents
                ON CONFLICT (doc_id, stage) DO NOTHING
                """,
                (stage,),
            ).rowcount
            res.jobs_seeded += max(seeded, 0)

            # A document whose text changed must be re-extracted.
            if changed:
                reset = cur.execute(
                    """
                    UPDATE jobs
                       SET status = 'pending', attempts = 0, error = NULL, updated_at = now()
                     WHERE stage = %s AND doc_id = ANY(%s) AND status <> 'pending'
                    """,
                    (stage, changed),
                ).rowcount
                res.jobs_reset += max(reset, 0)

    return res


def stats(cfg: Config) -> dict[str, Any]:
    """Numbers the Phase 1 gate checks."""
    q = {
        "documents": "SELECT count(*) FROM documents",
        "posts": "SELECT count(*) FROM documents WHERE doc_type = 'post'",
        "replies": "SELECT count(*) FROM documents WHERE doc_type = 'reply'",
        "authors": "SELECT count(DISTINCT author_id) FROM documents",
        "threads": "SELECT count(DISTINCT parent_id) FROM documents WHERE parent_id IS NOT NULL",
        "orphan_replies": (
            "SELECT count(*) FROM documents r WHERE r.doc_type = 'reply' AND NOT EXISTS "
            "(SELECT 1 FROM documents p WHERE p.doc_id = r.parent_id)"
        ),
        "jobs_pending": "SELECT count(*) FROM jobs WHERE status = 'pending'",
        "date_min": "SELECT min(created_at)::date FROM documents",
        "date_max": "SELECT max(created_at)::date FROM documents",
    }
    with connect(cfg) as conn:
        return {k: conn.execute(sql).fetchone()[0] for k, sql in q.items()}
