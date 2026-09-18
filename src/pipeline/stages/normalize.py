"""Stage 3: extractions -> mentions, switch_events, demographic_claims.

Fully deterministic - no model calls. The tables are truncated and rebuilt on
every run, so the lexicon and the negation guard can be tuned and re-run for free
until the OOV report is clean.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from pipeline.core.config import Config
from pipeline.core.db import connect
from pipeline.rules.context import locate
from pipeline.rules.lexicon import Lexicon, load as load_lexicon
from pipeline.rules.negation import check as negation_check
from pipeline.core.prompts import PROMPT_VERSION

UNSPECIFIED = "unspecified"


def _nullable(value: str | None) -> str | None:
    """Sentinel back to NULL.

    dose / duration / reason are REQUIRED strings in the extraction schema because
    as optional fields the model left them null on every single mention (610/610).
    The sentinel is a prompt-level device; the data product wants a real NULL.
    """
    if value is None:
        return None
    v = value.strip()
    return None if v.lower() in ("", UNSPECIFIED, "n/a", "none", "not stated") else v


@dataclass
class NormalizeResult:
    docs: int = 0
    treatments: int = 0
    symptoms: int = 0
    switches: int = 0
    switches_dropped: int = 0
    demo_accepted: int = 0
    demo_rejected: int = 0
    surface_located: int = 0
    unattested_name: int = 0
    attribution_fixed: int = 0
    kind_fixed: int = 0
    surface_missing: int = 0
    context_any: int = 0
    oov: Counter = field(default_factory=Counter)
    mapped: int = 0

    @property
    def mention_total(self) -> int:
        return self.treatments + self.symptoms

    @property
    def oov_total(self) -> int:
        return sum(self.oov.values())

    @property
    def mapped_pct(self) -> float:
        tot = self.mapped + self.oov_total
        return 100.0 * self.mapped / tot if tot else 0.0


FETCH = """
SELECT e.doc_id, e.raw_json, d.author_id, d.doc_type, d.created_at,
       coalesce(d.title, '') || ' ' || d.body AS text
  FROM extractions e
  JOIN documents d ON d.doc_id = e.doc_id
 WHERE e.parsed_ok AND e.prompt_version = %s AND e.model = %s
"""



def _context_for(text: str, surface: str, evidence: str | None, lex: Lexicon,
                 cid: str | None):
    """Find text that actually shows the term in use.

    Order matters. The surface is preferred; when it is a clinical label the
    author never wrote ("brain fog" for "I keep losing words"), the entity's own
    aliases are searched next, because the lexicon already knows every way this
    community phrases that entity. The model's evidence quote is the last resort
    and is matched strictly - a loose match lands on unrelated sentences.
    """
    loc = locate(text, surface, near=evidence)
    if loc:
        return loc.context, loc.start, loc.end, True, "surface"

    if cid and cid in lex.entries:
        entry = lex.entries[cid]
        for alias in sorted((entry.id, *entry.aliases), key=len, reverse=True):
            hit = locate(text, alias, near=evidence, strict=True)
            if hit:
                return hit.context, hit.start, hit.end, False, "alias"

    if evidence:
        ev = locate(text, evidence, strict=True)
        if ev:
            return ev.context, ev.start, ev.end, False, "evidence"

    return None, None, None, False, None



_SECOND_PERSON = re.compile(r"\b(your|you're|you have|you might|you could|you should|ur)\b", re.I)
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|my|me|mine)\b", re.I)


def _fix_attribution(experienced_by: str, context: str | None) -> tuple[str, bool]:
    """Downgrade 'self' when the surrounding text is advice to someone else.

    One member was recorded as a current Climara user on the strength of
    "worth asking your GP whether your Climara dose is optimised" - second-person
    advice, not personal use. Since "who is the Climara patient" is built from
    experienced_by='self', that error goes straight into the client answer.

    Deliberately conservative: it fires only when there is no first-person marker
    anywhere in the context, so "I'm on Climara, your mileage may vary" is left
    alone.
    """
    if experienced_by != "self" or not context:
        return experienced_by, False
    if _SECOND_PERSON.search(context) and not _FIRST_PERSON.search(context):
        return "other", True
    return experienced_by, False


def _fix_kind(lex: Lexicon, kind: str, cid: str | None) -> tuple[str, bool]:
    """The lexicon is authoritative on whether an entity is a treatment or a
    symptom. 47 mentions arrived with HRT and progestins listed as symptoms."""
    if not cid or cid not in lex.entries:
        return kind, False
    want = lex.entries[cid].kind
    if want != kind:
        return want, True
    return kind, False


def _name_appears_in_text(lex: Lexicon, cid: str | None, text: str) -> bool:
    """A named product must be provable from the text.

    Until v7 the prompt named Climara six times in its examples, which primed the
    model to reach for it under uncertainty - one mention arrived with the evidence
    "unspecified (no mention of Climara)". v7 replaced every brand in the prompt
    with DRUG-A placeholders, so the bias is fixed at source; this check stays as
    the backstop, because any concrete phrasing in a prompt is liable to be copied
    (the same leak reappeared in the tolerability field's examples before that
    field became an enum). Every NAMED entity is checked against the document's
    own text, and unverifiable ones are marked rather than counted. Generic
    entities (hrt_unspecified, symptoms) are exempt: authors describe those in
    their own words.
    """
    if not cid or cid not in lex.entries:
        return True
    entry = lex.entries[cid]
    if not entry.name_must_appear:
        return True
    hay = text.lower()
    return any(a.lower() in hay for a in (entry.id.replace("_", " "), *entry.aliases))


def _canon(lex: Lexicon, surface: str) -> tuple[str | None, str | None]:
    hit = lex.match(surface)
    return (hit.id, hit.entity_class) if hit else (None, None)


def run(cfg: Config) -> NormalizeResult:
    lex = load_lexicon()
    res = NormalizeResult()

    with connect(cfg) as conn:
        cur = conn.execute(FETCH, (PROMPT_VERSION, cfg.extraction_model))
        rows = cur.fetchall()

        # Deterministic rebuild: no incremental state to drift.
        conn.execute("TRUNCATE mentions, switch_events, demographic_claims RESTART IDENTITY")

        for doc_id, j, author_id, doc_type, created_at, text in rows:
            res.docs += 1

            for t in j.get("treatments", []):
                cid, cls = _canon(lex, t["surface"])
                if cid:
                    res.mapped += 1
                else:
                    res.oov[t["surface"].strip().lower()] += 1
                supported = _name_appears_in_text(lex, cid, text)
                if not supported:
                    res.unattested_name += 1
                ctx, cs, ce, found, csrc = _context_for(text, t["surface"], t.get("evidence"), lex, cid)
                exp_by, fixed = _fix_attribution(t["experienced_by"], ctx)
                if fixed:
                    res.attribution_fixed += 1
                if found:
                    res.surface_located += 1
                else:
                    res.surface_missing += 1
                if ctx:
                    res.context_any += 1
                # The lexicon is authoritative on kind in BOTH directions. Without
                # this, 161 mentions the model emitted as treatments but which
                # resolve to symptom entities ("histamine intolerance", "dry eyes")
                # stayed filed as treatments - and showed up as treatment rows in
                # the analysis.
                real_kind, kfix = _fix_kind(lex, "treatment", cid)
                if kfix:
                    res.kind_fixed += 1
                conn.execute(
                    """INSERT INTO mentions (doc_id, author_id, doc_type, created_at, kind,
                           surface, canonical_id, canonical_class, stance, experienced_by,
                           dose, duration, sentiment, evidence, lexicon_hash,
                           char_start, char_end, context, surface_found, context_source,
                           supported, prompt_version)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (doc_id, author_id, doc_type, created_at, real_kind, t["surface"], cid, cls,
                     t.get("stance"), exp_by, _nullable(t.get("dose")), _nullable(t.get("duration")),
                     t.get("sentiment"), t["evidence"], lex.digest, cs, ce, ctx, found, csrc, supported,
                     PROMPT_VERSION),
                )
                res.treatments += 1

            for s in j.get("symptoms", []):
                cid, cls = _canon(lex, s["surface"])
                if cid:
                    res.mapped += 1
                else:
                    res.oov[s["surface"].strip().lower()] += 1
                real_kind, kfix = _fix_kind(lex, "symptom", cid)
                if kfix:
                    res.kind_fixed += 1
                ctx, cs, ce, found, csrc = _context_for(text, s["surface"], s.get("evidence"), lex, cid)
                # Symptoms need the same attribution guard as treatments: a reply
                # describing the original poster's hot flashes is not the replier's
                # own symptom, and the profile's symptom list is built from 'self'.
                s_exp_by, s_fixed = _fix_attribution(s["experienced_by"], ctx)
                if s_fixed:
                    res.attribution_fixed += 1
                if found:
                    res.surface_located += 1
                else:
                    res.surface_missing += 1
                if ctx:
                    res.context_any += 1
                conn.execute(
                    """INSERT INTO mentions (doc_id, author_id, doc_type, created_at, kind,
                           surface, canonical_id, canonical_class, status, experienced_by,
                           evidence, lexicon_hash, char_start, char_end, context, surface_found,
                           context_source, prompt_version)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (doc_id, author_id, doc_type, created_at, real_kind, s["surface"], cid, cls,
                     s.get("status"), s_exp_by, s["evidence"], lex.digest,
                     cs, ce, ctx, found, csrc, PROMPT_VERSION),
                )
                res.symptoms += 1

            for sw in j.get("switches", []):
                frm = (sw.get("from_treatment") or "").strip()
                to = (sw.get("to_treatment") or "").strip()
                # An edge that names neither side carries no information.
                if frm.lower() in ("", UNSPECIFIED) and to.lower() in ("", UNSPECIFIED):
                    res.switches_dropped += 1
                    continue
                fid, _ = _canon(lex, frm) if frm.lower() != UNSPECIFIED else (None, None)
                tid, _ = _canon(lex, to) if to.lower() != UNSPECIFIED else (None, None)
                floc = locate(text, frm, near=sw.get("evidence")) if frm else None
                tloc = locate(text, to, near=sw.get("evidence")) if to else None
                conn.execute(
                    """INSERT INTO switch_events (doc_id, author_id, created_at, from_surface,
                           to_surface, from_canonical, to_canonical, reason, experienced_by,
                           evidence, lexicon_hash, from_context, to_context)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (doc_id, author_id, created_at, frm, to, fid, tid, _nullable(sw.get("reason")),
                     sw["experienced_by"], sw["evidence"], lex.digest,
                     floc.context if floc else None, tloc.context if tloc else None),
                )
                res.switches += 1

            for fld, val in (j.get("demographics") or {}).items():
                if val is None or val == "":
                    continue
                ok, note = negation_check(fld, val, text)
                conn.execute(
                    """INSERT INTO demographic_claims (doc_id, author_id, created_at, field,
                           value, rejected, reject_note)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                    (doc_id, author_id, created_at, fld, str(val), not ok, note),
                )
                if ok:
                    res.demo_accepted += 1
                else:
                    res.demo_rejected += 1

    return res
