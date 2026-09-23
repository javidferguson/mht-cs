"""Segment a member into the buckets the analysis cuts by.

Deterministic, no model calls - the same shape as the rest of `rules/`, and here
rather than in a notebook because three notebooks want them and the age banding
already existed as a copy-paste pair in two cells of `02_explore`.

Two of these exist because the raw column is unusable as-is:

- **Location comes from `city`, not `country`.** The extracted `country` field is
  populated for 36 of 100 members; `city` for 99. Cutting on `country` silently
  drops two thirds of the community.
- **`profession` is free text** - 46 distinct strings across 99 members, the
  largest bucket 6. Ungrouped it cannot be a chart axis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

AGE_BANDS = ("<45", "45-49", "50-54", "55-59", "60+")


def age_band(age: float | int | str | None) -> str | None:
    """Observed range is 38-64, so the edge bands are open-ended by design."""
    if age is None or age == "":
        return None
    try:
        a = int(float(age))
    except (TypeError, ValueError):
        return None
    if a < 45:
        return "<45"
    if a < 50:
        return "45-49"
    if a < 55:
        return "50-54"
    if a < 60:
        return "55-59"
    return "60+"


# City -> country. Members write where they live, not which country it is in, so
# the roll-up is a lookup. "UK & Ireland" is one bucket because Ireland alone is
# 5 members and no bar should rest on five people.
_COUNTRY_BY_PLACE = {
    # Canada
    "vancouver": "Canada", "calgary": "Canada", "toronto": "Canada",
    # Australia
    "melbourne": "Australia", "sydney": "Australia",
    # UK & Ireland
    "dublin": "UK & Ireland", "edinburgh": "UK & Ireland",
    "manchester": "UK & Ireland", "uk": "UK & Ireland", "scotland": "UK & Ireland",
    # United States - cities
    "portland": "United States", "atlanta": "United States", "seattle": "United States",
    "albuquerque": "United States", "detroit": "United States", "phoenix": "United States",
    "austin": "United States", "brooklyn": "United States", "chicago": "United States",
    "boston": "United States", "raleigh": "United States", "minneapolis": "United States",
    # United States - members who gave a state instead of a city
    "pa": "United States", "pennsylvania": "United States", "vermont": "United States",
    "texas": "United States", "new mexico": "United States",
}

# Stripped before lookup: members write "rural PA" and "suburban Detroit", which
# are the same place as "PA" and "Detroit" for a country roll-up.
_PLACE_MODIFIERS = ("rural ", "suburban ", "urban ", "greater ", "west ", "east ",
                    "north ", "south ", "outside ")


def country_of_city(city: str | None) -> str | None:
    """None when the place is unknown - never guessed, and never 'Other'."""
    if not city:
        return None
    v = " ".join(str(city).strip().lower().split())
    if not v or v == "nan":
        return None
    changed = True
    while changed:
        changed = False
        for m in _PLACE_MODIFIERS:
            if v.startswith(m):
                v, changed = v[len(m):].strip(), True
    return _COUNTRY_BY_PLACE.get(v)


# ORDER IS PRIORITY: the first group with a matching key wins. Healthcare leads
# so a "school nurse" is clinical rather than educational, and Education precedes
# Corporate so a "library director" is not filed as an executive - it was, first
# time. Picking the LONGEST matching key instead reads as more principled and is
# worse: `school` (6) would beat `nurse` (5) and misfile the nurse.
_PROFESSION_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Healthcare", (
        "physician", "neurologist", "nurse", "physio", "therapist", "psychotherapist",
        "social worker", "vet", "gp", "pt", "doctor", "midwife", "pharmacist",
    )),
    ("Education", ("teacher", "professor", "library director", "librar",
                   "lecturer", "tutor", "school")),
    ("Creative", (
        "florist", "artist", "designer", "writer", "photograph", "illustrat", "musician",
    )),
    ("Service & hospitality", (
        "barista", "cook", "chef", "yoga", "server", "bartender", "stylist", "retail",
    )),
    ("Corporate & professional", (
        "director", "consult", "accountant", "litigat", "attorney", "lawyer",
        "executive assistant", "ea", "logistic", "manager", "engineer", "analyst",
        "marketing", "hr", "real estate", "business owner", "boutique", "agent",
    )),
    ("Not working", ("retired", "stay at home", "stay-at-home", "homemaker")),
)

# Extracted as a profession but is not one - the member wrote "professionally" in
# a sentence about something else. Dropped rather than bucketed, so a junk value
# never inflates a real group.
_NOT_A_PROFESSION = {"professionally", "none", "n/a", "nan", "unspecified"}


def profession_group(profession: str | None) -> str | None:
    """None when unstated or unrecognised - counted as missing, not as 'Other'."""
    if not profession:
        return None
    v = " ".join(str(profession).strip().lower().split())
    if not v or v in _NOT_A_PROFESSION:
        return None
    for group, keys in _PROFESSION_GROUPS:
        for k in keys:
            # Whole-word match. Bare "ea" and "pt" are real profession strings here
            # and must not fire inside "teacher" or "physiotherapist", so only keys
            # longer than three characters are allowed to match as a substring.
            if v == k or v.startswith(k + " ") or v.endswith(" " + k) or f" {k} " in f" {v} ":
                return group
            if len(k) > 3 and k in v:
                return group
    return None


TOOK, CONSIDERED, NEVER = "took", "considered", "never"


def exposure(stance_ids: Mapping[str, Iterable[str]], entity: str) -> str:
    """How a member relates to one treatment, from their rolled-up profile lists.

    `took` means the member's own stance on it is current or past. That is the
    definition the cuts use, because the two alternatives do not work: "ever
    self-mentioned" is 81 of 100 members and cannot discriminate, while the
    mention-count cohort in `cohort.py` is 7 - real, but one or two members per
    age band, which is not a chart.
    """
    def has(stance: str) -> bool:
        return entity in set(stance_ids.get(stance) or ())

    if has("current") or has("past"):
        return TOOK
    if has("considering") or has("rejected"):
        return CONSIDERED
    return NEVER
