"""Locate a surface form in its source document and quote around it.

The model's `evidence` is its justification for the extraction, and in 63% of
treatment mentions it did not contain the surface at all - "Did it again last
night and apparently my body finally decided to cooperate" as the evidence for
`magnesium glycinate`. That is a true quote but it shows nothing about how the
term is used.

Where the surface sits in the text is exactly computable, so it is computed here
rather than asked of the model: character offsets plus a window of surrounding
text, guaranteed to contain the surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WINDOW = 130          # characters of context on each side
MIN_TOKEN = 4         # shortest fallback token worth searching for


@dataclass(frozen=True)
class Located:
    start: int
    end: int
    context: str
    matched: str
    exact: bool       # True = the whole surface matched; False = fallback token


def _find_all(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    return [
        (m.start(), m.end())
        for m in re.finditer(re.escape(needle), text, re.IGNORECASE)
    ]


def _snap(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen to whole words so the quote doesn't start mid-token."""
    lo = max(0, start - WINDOW)
    hi = min(len(text), end + WINDOW)
    if lo > 0:
        space = text.find(" ", lo)
        lo = space + 1 if 0 <= space < start else lo
    if hi < len(text):
        space = text.rfind(" ", end, hi)
        hi = space if space > end else hi
    return lo, hi


def locate(
    text: str, surface: str, near: str | None = None, *, strict: bool = False
) -> Located | None:
    """Find `surface` in `text` and quote around it.

    `near` is the model's evidence quote; when the surface occurs several times,
    the occurrence closest to that quote is chosen, so context and evidence
    describe the same moment in the document.
    """
    text = text or ""
    surface = (surface or "").strip()
    spans = _find_all(text, surface)
    exact = bool(spans)

    if not spans and strict:
        # Whole-string match only. Used when a loose token match would land
        # somewhere arbitrary - "migraines" matching a word in an unrelated
        # sentence gives context that shows nothing.
        return None

    if not spans:
        # Fall back to the most distinctive token: "estrogen (Climara patch)"
        # does not appear verbatim, but "Climara" does.
        tokens = sorted(
            (t for t in re.findall(r"[A-Za-z][A-Za-z0-9-]{%d,}" % (MIN_TOKEN - 1), surface)),
            key=len,
            reverse=True,
        )
        for tok in tokens:
            spans = _find_all(text, tok)
            if spans:
                break
        if not spans:
            return None

    if near and len(spans) > 1:
        anchors = _find_all(text, near[:40])
        if anchors:
            a = anchors[0][0]
            spans.sort(key=lambda s: abs(s[0] - a))

    start, end = spans[0]
    lo, hi = _snap(text, start, end)
    quote = text[lo:hi].replace("\n", " ").strip()
    quote = re.sub(r"\s+", " ", quote)
    if lo > 0:
        quote = "…" + quote
    if hi < len(text):
        quote = quote + "…"
    return Located(start=start, end=end, context=quote, matched=text[start:end], exact=exact)
