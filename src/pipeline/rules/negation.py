"""Deterministic guard for demographic claims.

Four rounds of prompt engineering could not stop llama3.1:8b extracting
`profession: neurologist` from "I'm not a neurologist but...". Rather than keep
tuning the prompt, the known failure mode is corrected here, where it is cheap,
inspectable and testable. This is the hybrid design doing its job.
"""

from __future__ import annotations

import re

# Cues that negate a following self-description.
_NEG = re.compile(
    r"\b(?:not|never|n't|no longer|aren't|isn't|wasn't|ain't)\b[\s\w]{0,12}$",
    re.I,
)
LOOKBEHIND = 48

# Fields whose value must literally appear in the source text. Age is handled
# separately (digits), and free-form fields like caregiving are paraphrased too
# often to require a literal match.
_MUST_APPEAR = {"profession", "race_ethnicity", "city", "region", "country"}

# Surface collisions where a demographic word is really part of a product name
# (e.g. "black" in "black cohosh"). Domain-specific, so it lives in
# config/domain.yml rather than here.
def _decoys() -> dict[str, tuple[str, ...]]:
    from pipeline.rules.domain import load as load_domain

    return load_domain().decoys


def _occurrences(text: str, needle: str) -> list[int]:
    return [m.start() for m in re.finditer(re.escape(needle), text, re.I)]


def check(field: str, value: object, text: str) -> tuple[bool, str | None]:
    """Return (accept, reject_note)."""
    if value is None or value == "":
        return False, "empty"

    if field == "age":
        try:
            age = int(value)
        except (TypeError, ValueError):
            return False, "age not an integer"
        if not 25 <= age <= 85:
            return False, f"age {age} outside plausible range"
        if not re.search(rf"(?<!\d){age}(?!\d)", text):
            return False, f"age {age} does not appear in the text"
        return True, None

    if field in ("has_partner", "caregiving"):
        return True, None

    sval = str(value).strip()
    if field not in _MUST_APPEAR:
        return True, None

    # The value, or its head word, must actually be present in the document.
    spots = _occurrences(text, sval)
    if not spots:
        head = sval.split()[-1] if sval.split() else sval
        spots = _occurrences(text, head)
        if not spots:
            return False, f"{field} {sval!r} does not appear in the text (inferred)"

    # Drop occurrences that are really part of a treatment name.
    decoys = _decoys().get(sval.lower(), ())
    if decoys:
        kept = [
            s for s in spots
            if not any(
                d in text[max(0, s - 6) : s + len(sval) + 12].lower() for d in decoys
            )
        ]
        if not kept:
            return False, f"{field} {sval!r} only appears inside {decoys[0]!r}"
        spots = kept

    # Present - but is every occurrence negated?
    for start in spots:
        window = text[max(0, start - LOOKBEHIND) : start]
        if not _NEG.search(window):
            return True, None
    return False, f"{field} {sval!r} is negated in the text"
