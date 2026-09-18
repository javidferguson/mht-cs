"""Evaluation metrics. Every number states what it can and cannot prove."""

from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.core.config import Config
from pipeline.core.db import connect
from pipeline.qa.gold import NEGATIVE_CONTROLS, named_entity_tokens
from pipeline.rules.lexicon import load as load_lexicon
from pipeline.core.prompts import PROMPT_VERSION
from pipeline.rules.negation import check as negation_check


@dataclass
class EntityScore:
    entity: str
    docs_containing: int = 0
    docs_found: int = 0
    emitted_without_text: int = 0

    @property
    def recall(self) -> float:
        return self.docs_found / self.docs_containing if self.docs_containing else float("nan")


@dataclass
class EvalReport:
    prompt_version: str = ""
    mentions_built_from: object = None
    mentions_match_pv: bool = True
    entities: dict[str, EntityScore] = field(default_factory=dict)
    neg_pass: int = 0
    neg_total: int = 0
    neg_failures: list[str] = field(default_factory=list)
    parent_only_total: int = 0
    parent_only_non_self: int = 0
    distinct_surfaces: int = 0
    distinct_canonical: int = 0
    docs_extracted: int = 0
    docs_total: int = 0
    parse_ok: int = 0
    parse_total: int = 0
    mapped_pct: float = 0.0
    context_pct: float = 0.0

    @property
    def recall_overall(self) -> float:
        c = sum(b.docs_containing for b in self.entities.values())
        f = sum(b.docs_found for b in self.entities.values())
        return f / c if c else float("nan")

    @property
    def hallucinated(self) -> int:
        return sum(b.emitted_without_text for b in self.entities.values())


def run(cfg: Config, prompt_version: str | None = None) -> EvalReport:
    """Score one extraction run.

    Every version ever run is retained in `extractions`, so without a filter this
    blends them - v1 through v7 at once, which is how a half-finished re-extract
    can silently drag the headline number. Scoped to a single prompt_version so
    two runs can be compared honestly.
    """
    pv = prompt_version or PROMPT_VERSION
    rep = EvalReport(prompt_version=pv)
    tokens = named_entity_tokens()

    with connect(cfg) as conn:
        rep.docs_total = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        rep.docs_extracted = conn.execute(
            "SELECT count(DISTINCT doc_id) FROM extractions WHERE parsed_ok AND prompt_version = %s", (pv,)
        ).fetchone()[0]
        rep.parse_ok, rep.parse_total = conn.execute(
            "SELECT count(*) FILTER (WHERE parsed_ok), count(*) FROM extractions WHERE prompt_version = %s", (pv,)
        ).fetchone()
        # NOTE: the `mentions` table holds exactly ONE run - whichever
        # prompt_version `normalize` last built from. These three coverage
        # numbers therefore describe that run, NOT the `pv` being scored, and
        # printing them under a v5 header while mentions holds v7 is misleading.
        # The caller is told which run they actually describe.
        row = conn.execute(
            "SELECT DISTINCT prompt_version FROM mentions WHERE prompt_version IS NOT NULL"
        ).fetchall()
        rep.mentions_built_from = row[0][0] if len(row) == 1 else None
        rep.mentions_match_pv = rep.mentions_built_from == pv
        rep.mapped_pct, rep.context_pct = conn.execute(
            """SELECT 100.0*avg((canonical_id IS NOT NULL)::int),
                      100.0*avg((context IS NOT NULL)::int) FROM mentions"""
        ).fetchone()
        rep.distinct_surfaces, rep.distinct_canonical = conn.execute(
            """SELECT count(DISTINCT lower(surface)), count(DISTINCT canonical_id)
                 FROM mentions WHERE canonical_id IS NOT NULL"""
        ).fetchone()

        # --- named-entity recall & hallucination ---------------------------------
        # Measured against extractions.raw_json, NOT the mentions table: mentions
        # is rebuilt by `normalize` and lags a running extract, which would show
        # up as phantom misses.
        for token, canonical in tokens.items():
            bs = rep.entities.setdefault(canonical, EntityScore(entity=canonical))
            row = conn.execute(
                """
                SELECT count(*), count(*) FILTER (WHERE hit)
                  FROM (
                    SELECT EXISTS (
                             SELECT 1 FROM jsonb_array_elements(e.raw_json->'treatments') t
                              WHERE t->>'surface' ILIKE %(like)s
                           ) AS hit
                      FROM extractions e
                      JOIN documents d ON d.doc_id = e.doc_id
                     WHERE e.parsed_ok AND e.prompt_version = %(pv)s
                       AND (coalesce(d.title,'') || ' ' || d.body) ILIKE %(like)s
                  ) x
                """,
                {"like": f"%{token}%", "pv": pv},
            ).fetchone()
            bs.docs_containing += row[0]
            bs.docs_found += row[1]

        # The "no supporting text" check must use EVERY alias that resolves to the
        # entity, not just the gold token. "Evorel" legitimately maps to estradot
        # and "Imvexxa" to imvexxy; checking only the canonical token scores those
        # correct extractions as hallucinations.
        lex = load_lexicon()
        for canonical in set(tokens.values()):
            toks = sorted(
                {t for t, c in tokens.items() if c == canonical}
                | ({canonical} | set(lex.entries[canonical].aliases)
                   if canonical in lex.entries else set())
            )
            like = " OR ".join(["txt ILIKE %s"] * len(toks))
            rep.entities.setdefault(canonical, EntityScore(entity=canonical))
            rep.entities[canonical].emitted_without_text = conn.execute(
                f"""SELECT count(*) FROM (
                        SELECT m.doc_id,
                               coalesce(d.title,'')||' '||d.body||' '||
                               coalesce(p.title,'')||' '||coalesce(p.body,'') AS txt
                          FROM mentions m
                          JOIN documents d ON d.doc_id = m.doc_id
                          LEFT JOIN documents p ON p.doc_id = d.parent_id
                         WHERE m.canonical_id = %s
                    ) x WHERE NOT ({like})""",
                (canonical, *[f"%{t}%" for t in toks]),
            ).fetchone()[0]

        # --- attribution tendency (proxy, not accuracy) --------------------
        for token, canonical in tokens.items():
            rows = conn.execute(
                """SELECT m.experienced_by
                     FROM mentions m
                     JOIN documents d ON d.doc_id = m.doc_id
                     JOIN documents p ON p.doc_id = d.parent_id
                    WHERE m.canonical_id = %s
                      AND (coalesce(d.title,'')||' '||d.body) NOT ILIKE %s
                      AND (coalesce(p.title,'')||' '||p.body) ILIKE %s""",
                (canonical, f"%{token}%", f"%{token}%"),
            ).fetchall()
            rep.parent_only_total += len(rows)
            rep.parent_only_non_self += sum(1 for r in rows if r[0] != "self")

    # --- negative controls -------------------------------------------------
    for text, fieldname, value, must_reject, note in NEGATIVE_CONTROLS:
        ok, _ = negation_check(fieldname, value, text)
        rejected = not ok
        rep.neg_total += 1
        if rejected == must_reject:
            rep.neg_pass += 1
        else:
            rep.neg_failures.append(f"{note}: {fieldname}={value!r} -> {'rejected' if rejected else 'accepted'}")

    return rep
