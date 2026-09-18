"""A/B test: does parent-post context earn the complexity it costs?

Writes ONLY to `extractions`, under its own prompt_version. It never touches
`jobs` (which tracks the v5 run) and never calls `normalize` - normalize
truncates `mentions` and rebuilds from whatever PROMPT_VERSION is set, so running
it here would replace 35k mentions with 300 documents' worth.

Comparison is done directly on extractions.raw_json, the same technique the
named-entity recall metric in eval/score.py uses.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

from pipeline.core.config import Config
from pipeline.core.db import connect
from pipeline.rules.lexicon import load as load_lexicon
from pipeline.core.models import extraction_schema
from pipeline.core.ollama import OllamaClient
from pipeline.core.prompts import PARENT_CONTEXT_CHARS, PROMPT_VERSION, render_document, system_prompt

BASELINE = "v5"
PER_GROUP = 150


def _named_entity_tokens() -> list[str]:
    """Tokens whose presence in text is unambiguous, from the lexicon itself."""
    lex = load_lexicon()
    out: set[str] = set()
    for e in lex.entries.values():
        if not e.name_must_appear:
            continue
        for a in (e.id.replace("_", " "), *e.aliases):
            if len(a) >= 6 and " " not in a:   # single distinctive words only
                out.add(a.lower())
    return sorted(out)


SELECT_SQL = """
WITH toks AS (SELECT unnest(%(toks)s::text[]) AS tok),
     r AS (
  SELECT d.doc_id,
         d.author_id,
         d.title,
         d.body,
         p.title  AS parent_title,
         p.body   AS parent_body,
         EXISTS (SELECT 1 FROM toks WHERE d.body ILIKE '%%'||tok||'%%')  AS in_own,
         EXISTS (SELECT 1 FROM toks WHERE p.body ILIKE '%%'||tok||'%%')  AS in_parent
    FROM documents d
    JOIN documents p ON p.doc_id = d.parent_id
    JOIN extractions e ON e.doc_id = d.doc_id
                      AND e.prompt_version = %(base)s AND e.parsed_ok
   WHERE d.doc_type = 'reply'
)
(SELECT *, 'parent_only' AS grp FROM r WHERE in_parent AND NOT in_own ORDER BY doc_id LIMIT %(n)s)
UNION ALL
(SELECT *, 'own_text'    AS grp FROM r WHERE in_own                  ORDER BY doc_id LIMIT %(n)s)
"""


@dataclass
class Side:
    label: str
    docs: int = 0
    mentions: int = 0
    self_unattested: int = 0     # PRIMARY   - contamination
    self_attested: int = 0   # GUARDRAIL - real signal
    bare_refs: int = 0

BARE = {"the patch", "patch", "it", "the gel", "the pill", "this", "that", "the ring"}


def _tally(label: str, rows: list[tuple], toks: list[str]) -> Side:
    """Split self-attributed named-entity mentions by whether the name is actually in
    the reply's own text.

      self_attested -> the member named the drug themselves. Real signal.
      self_unattested   -> attributed to the member, but the name appears only in
                         the post they were answering. Contamination.
    """
    s = Side(label=label)
    for own_text, raw in rows:
        s.docs += 1
        low = (own_text or "").lower()
        for t in (raw or {}).get("treatments", []):
            s.mentions += 1
            surface = (t.get("surface") or "").strip().lower()
            if surface in BARE:
                s.bare_refs += 1
            if t.get("experienced_by") != "self":
                continue
            hits = [tok for tok in toks if tok in surface]
            if not hits:
                continue
            if any(tok in low for tok in hits):
                s.self_attested += 1
            else:
                s.self_unattested += 1
    return s


async def run(cfg: Config) -> tuple[Side, Side, int]:
    if PROMPT_VERSION == BASELINE:
        raise SystemExit(
            f"PROMPT_VERSION is still {BASELINE!r}. Run with "
            f"PROMPT_VERSION=v6-noparent PARENT_CONTEXT_CHARS=0"
        )

    toks = _named_entity_tokens()
    schema = extraction_schema()

    with connect(cfg) as conn:
        picked = conn.execute(
            SELECT_SQL, {"toks": toks, "base": BASELINE, "n": PER_GROUP}
        ).fetchall()
    cols = ["doc_id", "author_id", "title", "body", "parent_title", "parent_body",
            "in_own", "in_parent", "grp"]
    docs = [dict(zip(cols, r)) for r in picked]
    print(f"selected {len(docs)} replies "
          f"({sum(1 for d in docs if d['grp']=='parent_only')} parent-only, "
          f"{sum(1 for d in docs if d['grp']=='own_text')} own-text)")
    print(f"ablation: PROMPT_VERSION={PROMPT_VERSION} PARENT_CONTEXT_CHARS={PARENT_CONTEXT_CHARS}")

    sem = asyncio.Semaphore(cfg.ollama_num_parallel)
    done = {"n": 0}

    async def one(client: OllamaClient, d: dict) -> tuple[str, dict | None]:
        prompt = render_document(
            doc_type="reply", author_id=d["author_id"], title=d["title"], body=d["body"],
            parent_title=d["parent_title"], parent_body=d["parent_body"],
        )
        async with sem:
            try:
                g = await client.generate(prompt, schema=schema, system=system_prompt())
                parsed = g.parsed
            except Exception:
                try:
                    g = await client.generate(
                        prompt, schema=schema, system=system_prompt(),
                        options={"temperature": 0.4, "repeat_penalty": 1.3})
                    parsed = g.parsed
                except Exception:
                    parsed = None
        done["n"] += 1
        if done["n"] % 50 == 0:
            print(f"  {done['n']}/{len(docs)}")
        return d["doc_id"], parsed

    t0 = time.perf_counter()
    async with OllamaClient(cfg) as client:
        results = await asyncio.gather(*(one(client, d) for d in docs))
    failed = sum(1 for _, p in results if p is None)
    print(f"extracted {len(results) - failed}/{len(results)} in {time.perf_counter()-t0:.0f}s "
          f"({failed} failed)")

    with connect(cfg) as conn:
        for doc_id, parsed in results:
            if parsed is None:
                continue
            conn.execute(
                """INSERT INTO extractions (doc_id, content_hash, prompt_version, model,
                       raw_json, parsed_ok, prompt_tokens, completion_tokens, latency_ms)
                   SELECT %s, d.content_hash, %s, %s, %s::jsonb, TRUE, 0, 0, 0
                     FROM documents d WHERE d.doc_id = %s
                   ON CONFLICT (doc_id, content_hash, prompt_version, model)
                   DO UPDATE SET raw_json = EXCLUDED.raw_json""",
                (doc_id, PROMPT_VERSION, cfg.extraction_model, json.dumps(parsed), doc_id),
            )

        ids = [d["doc_id"] for d in docs]
        fetch = """SELECT coalesce(d.title,'')||' '||d.body, e.raw_json
                     FROM extractions e JOIN documents d ON d.doc_id = e.doc_id
                    WHERE e.doc_id = ANY(%s) AND e.prompt_version = %s AND e.parsed_ok"""
        base_rows = conn.execute(fetch, (ids, BASELINE)).fetchall()
        abl_rows = conn.execute(fetch, (ids, PROMPT_VERSION)).fetchall()

    return _tally(BASELINE, base_rows, toks), _tally(PROMPT_VERSION, abl_rows, toks), len(docs)
