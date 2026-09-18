"""Stage 4: mentions -> user_profiles + treatment_summary.

Two passes:
  1. Deterministic SQL rollup. Produces every number in the data product.
  2. One LLM call per author (100 total, ~5 min) writing a `narrative`.
     Entirely separate from the rollup - drop the column and nothing else moves.

Attribution is load-bearing here. Only `experienced_by='self'` mentions count
towards a member's own regimen; without that filter every replier discussing the
original poster's Climara would be counted as a Climara patient.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from pipeline.core.config import Config
from pipeline.core.db import connect
from pipeline.rules.domain import load as load_domain
from pipeline.rules.lexicon import load as load_lexicon
from pipeline.core.ollama import OllamaClient

# Stance at the LATEST mention wins: "I'm on Climara" in November and "I came off
# it" in March must resolve to past, not current.
TREATMENTS_SQL = """
SELECT author_id, canonical_id, stance, sentiment, first_seen, last_seen, n, context
  FROM (
    SELECT m.author_id, m.canonical_id,
           first_value(m.stance)    OVER w AS stance,
           first_value(m.sentiment) OVER w AS sentiment,
           first_value(m.context)   OVER w AS context,
           min(m.created_at)        OVER (PARTITION BY m.author_id, m.canonical_id) AS first_seen,
           max(m.created_at)        OVER (PARTITION BY m.author_id, m.canonical_id) AS last_seen,
           count(*)                 OVER (PARTITION BY m.author_id, m.canonical_id) AS n,
           row_number()             OVER w AS rn
      FROM mentions m
     WHERE m.kind = 'treatment' AND m.experienced_by = 'self' AND m.supported
       AND m.canonical_id IS NOT NULL AND m.canonical_id <> 'non_entity'
    WINDOW w AS (PARTITION BY m.author_id, m.canonical_id ORDER BY m.created_at DESC)
  ) x
 WHERE rn = 1
"""

SYMPTOMS_SQL = """
SELECT author_id, canonical_id, status, n, context
  FROM (
    SELECT m.author_id, m.canonical_id,
           first_value(m.status)  OVER w AS status,
           first_value(m.context) OVER w AS context,
           count(*) OVER (PARTITION BY m.author_id, m.canonical_id) AS n,
           row_number() OVER w AS rn
      FROM mentions m
     WHERE m.kind = 'symptom' AND m.experienced_by = 'self' AND m.supported
       AND m.canonical_id IS NOT NULL AND m.canonical_id <> 'non_entity'
    WINDOW w AS (PARTITION BY m.author_id, m.canonical_id ORDER BY m.created_at DESC)
  ) x
 WHERE rn = 1
"""

# Accepted claims only, most frequent value per field, ties broken by recency.
DEMOGRAPHICS_SQL = """
SELECT author_id, field, value
  FROM (
    SELECT author_id, field, value,
           row_number() OVER (PARTITION BY author_id, field
                              ORDER BY count(*) DESC, max(created_at) DESC) AS rn
      FROM demographic_claims
     WHERE NOT rejected
     GROUP BY author_id, field, value
  ) x
 WHERE rn = 1
"""

STANCE_BUCKET = {
    "current": "current_treatments",
    "past": "past_treatments",
    "considering": "considering_treatments",
    "rejected": "rejected_treatments",
    "warned_against": "rejected_treatments",
    "recommended": "current_treatments",
}

PRESCRIBER_CLASS = {"telehealth": "telehealth", "provider": "specialist"}

# Umbrella entities double-count: a member who names a specific brand AND says
# "the patch" AND says "HRT" is describing one therapy three ways. The mapping
# lives in config/domain.yml so it travels with the domain, not the code.
UMBRELLA_SUPERSEDED_BY = load_domain().umbrella_superseded_by


def _mark_superseded(items: list[dict]) -> list[dict]:
    specific_classes = {
        i["class"] for i in items if i["id"] not in UMBRELLA_SUPERSEDED_BY
    }
    for i in items:
        covers = UMBRELLA_SUPERSEDED_BY.get(i["id"])
        i["is_umbrella"] = covers is not None
        i["superseded"] = bool(covers and (covers & specific_classes))
    return items


@dataclass
class ProfileResult:
    profiles: int = 0
    treatments_rolled: int = 0
    symptoms_rolled: int = 0
    narratives: int = 0
    narrative_failures: int = 0


def _labels() -> dict[str, dict]:
    lex = load_lexicon()
    return {
        e.id: {"label": e.label, "class": e.entity_class, "manufacturer": e.manufacturer}
        for e in lex.entries.values()
    }


def build(cfg: Config) -> ProfileResult:
    res = ProfileResult()
    meta = _labels()

    with connect(cfg) as conn:
        base = conn.execute(
            """SELECT author_id, count(*) AS n_docs,
                      count(*) FILTER (WHERE doc_type='post')  AS n_posts,
                      count(*) FILTER (WHERE doc_type='reply') AS n_replies,
                      min(created_at), max(created_at)
                 FROM documents GROUP BY author_id"""
        ).fetchall()

        treatments: dict[str, list] = {}
        for a, cid, stance, sent, first, last, n, ctx in conn.execute(TREATMENTS_SQL).fetchall():
            treatments.setdefault(a, []).append(
                {"id": cid, "label": meta.get(cid, {}).get("label", cid),
                 "class": meta.get(cid, {}).get("class"), "stance": stance,
                 "sentiment": sent, "mentions": n,
                 "first_seen": first.date().isoformat() if first else None,
                 "last_seen": last.date().isoformat() if last else None,
                 "context": ctx}
            )

        symptoms: dict[str, list] = {}
        for a, cid, status, n, ctx in conn.execute(SYMPTOMS_SQL).fetchall():
            symptoms.setdefault(a, []).append(
                {"id": cid, "label": meta.get(cid, {}).get("label", cid),
                 "status": status, "mentions": n, "context": ctx}
            )

        demo: dict[str, dict] = {}
        for a, field, value in conn.execute(DEMOGRAPHICS_SQL).fetchall():
            demo.setdefault(a, {})[field] = value

        switches = dict(
            conn.execute(
                "SELECT author_id, count(*) FROM switch_events"
                " WHERE experienced_by='self' GROUP BY author_id"
            ).fetchall()
        )

        conn.execute("TRUNCATE user_profiles")
        for author_id, n_docs, n_posts, n_replies, first, last in base:
            ts = _mark_superseded(
                sorted(treatments.get(author_id, []), key=lambda x: -x["mentions"])
            )
            sy = sorted(symptoms.get(author_id, []), key=lambda x: -x["mentions"])
            d = demo.get(author_id, {})

            buckets: dict[str, list] = {v: [] for v in set(STANCE_BUCKET.values())}
            for item in ts:
                buckets.setdefault(STANCE_BUCKET.get(item["stance"], "current_treatments"), []).append(item)

            classes = {t["class"] for t in ts}
            prescriber = next(
                (PRESCRIBER_CLASS[c] for c in classes if c in PRESCRIBER_CLASS), None
            )

            age = d.get("age")
            conn.execute(
                """INSERT INTO user_profiles (author_id, n_docs, n_posts, n_replies,
                       first_seen, last_seen, age, city, region, country, profession,
                       race_ethnicity, has_partner, caregiving, prescriber_type,
                       current_treatments, past_treatments, considering_treatments,
                       rejected_treatments, symptoms, n_treatment_mentions,
                       n_symptom_mentions, n_switches)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (author_id, n_docs, n_posts, n_replies, first, last,
                 int(age) if age and str(age).isdigit() else None,
                 d.get("city"), d.get("region"), d.get("country"), d.get("profession"),
                 d.get("race_ethnicity"),
                 {"True": True, "False": False}.get(str(d.get("has_partner"))),
                 d.get("caregiving"), prescriber,
                 json.dumps(buckets.get("current_treatments", [])),
                 json.dumps(buckets.get("past_treatments", [])),
                 json.dumps(buckets.get("considering_treatments", [])),
                 json.dumps(buckets.get("rejected_treatments", [])),
                 json.dumps(sy), len(ts), len(sy), switches.get(author_id, 0)),
            )
            res.profiles += 1
            res.treatments_rolled += len(ts)
            res.symptoms_rolled += len(sy)

        # ---- treatment_summary --------------------------------------------
        conn.execute("TRUNCATE treatment_summary")
        conn.execute(
            """
            INSERT INTO treatment_summary (canonical_id, label, entity_class, manufacturer,
                   n_users_self, n_mentions, n_mentions_self, stance_counts, sentiment_counts,
                   n_switch_from, n_switch_to)
            SELECT m.canonical_id, '', NULL, NULL,
                   count(DISTINCT m.author_id) FILTER (WHERE m.experienced_by='self'),
                   count(*), count(*) FILTER (WHERE m.experienced_by='self'),
                   jsonb_object_agg(COALESCE(s.stance,'none'), s.c) FILTER (WHERE s.stance IS NOT NULL),
                   '{}'::jsonb, 0, 0
              FROM mentions m
              LEFT JOIN LATERAL (
                   SELECT m2.stance, count(*) AS c FROM mentions m2
                    WHERE m2.supported AND m2.canonical_id = m.canonical_id AND m2.kind='treatment'
                    GROUP BY m2.stance
              ) s ON TRUE
             WHERE m.kind='treatment' AND m.canonical_id IS NOT NULL AND m.supported
             GROUP BY m.canonical_id
            """
        )
        for cid, info in meta.items():
            conn.execute(
                "UPDATE treatment_summary SET label=%s, entity_class=%s, manufacturer=%s"
                " WHERE canonical_id=%s",
                (info["label"], info["class"], info["manufacturer"], cid),
            )
        conn.execute(
            """UPDATE treatment_summary ts SET
                   n_switch_from = COALESCE((SELECT count(*) FROM switch_events
                                              WHERE from_canonical = ts.canonical_id), 0),
                   n_switch_to   = COALESCE((SELECT count(*) FROM switch_events
                                              WHERE to_canonical = ts.canonical_id), 0)"""
        )

    return res


# --------------------------------------------------------------------------
# Narrative pass: 100 calls, independent of everything above.
# --------------------------------------------------------------------------
NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {"narrative": {"type": "string"}},
    "required": ["narrative"],
}

NARRATIVE_SYSTEM = (
    f"You write a two-to-four sentence factual profile of a {load_domain().member_label} "
    "from structured data about them. Use ONLY the supplied facts. Do not invent "
    "treatments, symptoms, ages or locations. If a field is absent, say nothing about "
    "it. Write plainly, no marketing tone, no advice."
)


def _brief(row: dict) -> str:
    def fmt(items, key="stance"):
        kept = [i for i in items if not i.get("superseded")]
        return ", ".join(f"{i['label']} ({i.get(key)})" for i in kept[:8]) or "none recorded"

    parts = [
        f"Member {row['author_id']}: {row['n_docs']} posts/replies.",
        f"Currently using: {fmt(row['current_treatments'])}.",
        f"Previously used: {fmt(row['past_treatments'])}.",
        f"Considering: {fmt(row['considering_treatments'])}.",
        f"Symptoms: {fmt(row['symptoms'], 'status')}.",
    ]
    d = [f"{k}={row[k]}" for k in ("age", "city", "country", "profession") if row.get(k)]
    if d:
        parts.append("Stated about themselves: " + ", ".join(d) + ".")
    return " ".join(parts)


async def narrate(cfg: Config, limit: int | None = None, progress=None) -> ProfileResult:
    res = ProfileResult()
    with connect(cfg) as conn:
        cur = conn.execute(
            """SELECT author_id, n_docs, current_treatments, past_treatments,
                      considering_treatments, symptoms, age, city, country, profession
                 FROM user_profiles ORDER BY author_id""" + (" LIMIT %s" % limit if limit else "")
        )
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    sem = asyncio.Semaphore(cfg.ollama_num_parallel)

    async def one(row: dict) -> None:
        async with sem:
            try:
                g = await client.generate(_brief(row), schema=NARRATIVE_SCHEMA,
                                          system=NARRATIVE_SYSTEM)
                text = g.parsed["narrative"]
            except Exception:  # noqa: BLE001
                res.narrative_failures += 1
                return
        with connect(conn_cfg) as c:
            c.execute("UPDATE user_profiles SET narrative=%s WHERE author_id=%s",
                      (text, row["author_id"]))
        res.narratives += 1
        if progress:
            progress(res)

    conn_cfg = cfg
    async with OllamaClient(cfg) as client:
        await asyncio.gather(*(one(r) for r in rows))
    return res
