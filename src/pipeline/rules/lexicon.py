"""Deterministic canonicalisation: surface form -> canonical entity.

This is the half of the hybrid design that makes aggregates stable. The LLM finds
mentions and reads stance; this file decides that "vagifem", "vaginal estrogen",
"local estrogen", "Estring" and "Ovestin" are one thing. Because it runs without
the model, the lexicon can be tuned and re-run for free.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

LEXICON_PATH = Path("/app/config/lexicon.yml")

_ARTICLES = re.compile(r"^(the|a|an|my|some)\s+", re.I)
_PUNCT = re.compile(r"[^\w\s/+-]")
_WS = re.compile(r"\s+")


def normalise(surface: str) -> str:
    """Fold a surface form for matching: lowercase, drop articles and punctuation."""
    s = surface.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    s = _ARTICLES.sub("", s).strip()
    return s


@dataclass(frozen=True)
class Entry:
    id: str
    label: str
    kind: str
    entity_class: str
    aliases: tuple[str, ...]
    manufacturer: str | None = None
    name_must_appear: bool = False


@dataclass
class Lexicon:
    entries: dict[str, Entry]
    exact: dict[str, str]
    phrases: tuple[tuple[str, str], ...]  # (normalised alias, id), longest first
    digest: str

    def match(self, surface: str) -> Entry | None:
        """Exact normalised match, then longest whole-phrase containment.

        Containment is deliberately conservative and whole-word only, so "gel"
        does not swallow "Divigel" - the longest-alias-first ordering means the
        specific brand always wins over the generic class.
        """
        n = normalise(surface)
        if not n:
            return None
        if n in self.exact:
            return self.entries[self.exact[n]]
        for alias, eid in self.phrases:
            if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", n):
                return self.entries[eid]
        return None


@lru_cache(maxsize=1)
def load(path: Path = LEXICON_PATH) -> Lexicon:
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or []
    entries: dict[str, Entry] = {}
    exact: dict[str, str] = {}
    phrases: list[tuple[str, str]] = []

    for item in data:
        e = Entry(
            id=item["id"],
            label=item["label"],
            kind=item.get("kind", "treatment"),
            entity_class=item.get("class", "other"),
            aliases=tuple(item.get("aliases", [])),
            manufacturer=item.get("manufacturer"),
            name_must_appear=bool(item.get("name_must_appear", False)),
        )
        if e.id in entries:
            raise ValueError(f"duplicate lexicon id: {e.id}")
        entries[e.id] = e
        for alias in (e.id, *e.aliases):
            n = normalise(alias)
            if not n:
                continue
            if n in exact and exact[n] != e.id:
                raise ValueError(f"alias {alias!r} claimed by both {exact[n]} and {e.id}")
            exact[n] = e.id
            phrases.append((n, e.id))

    # Longest alias first so specific brands beat generic classes.
    phrases.sort(key=lambda p: (-len(p[0]), p[0]))
    return Lexicon(
        entries=entries,
        exact=exact,
        phrases=tuple(phrases),
        digest=hashlib.sha256(raw.encode()).hexdigest()[:12],
    )
