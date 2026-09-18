"""Gold sets for evaluation.

Hand-labelling 150 documents was not done, and pretending otherwise would make
every metric below meaningless. Instead each metric states exactly what it can
prove from the data:

  * NAMED-ENTITY RECALL is a real, deterministic gold standard. A distinctive
    name either appears verbatim in a document or it does not - no judgement
    involved - and the extractor either found it or missed it.
  * HALLUCINATION is its mirror: an entity emitted for a document where neither
    the document nor its parent contains its name.
  * ATTRIBUTION is a proxy, not a gold standard. When a name appears ONLY in
    the parent post, a reply mentioning it is usually discussing someone else's
    treatment, so `experienced_by` should lean away from 'self'. Stated as a
    tendency, never as accuracy.
  * NEGATIVE CONTROLS are hand-written and exact.

Anything a human would have to adjudicate (is this stance really 'considering'?)
is out of scope here and needs real annotation.
"""

from __future__ import annotations

# Derived from the lexicon rather than hand-listed: entries marked
# `name_must_appear: true` are NAMED entities - ones a member cannot refer to
# without writing the name - which is exactly the property a recall gold standard
# needs. Note "named" is about lexical distinctiveness, not trademarks: Climara is
# a brand, spironolactone is a generic molecule, Midi is a company, and all three
# behave identically here because none of them has a casual synonym. Multi-word and
# short aliases are excluded - a short token like "ala" or a phrase like "vaginal
# estrogen" is not unambiguous enough to score against.
#
# This is what makes the evaluation travel with the domain: swap lexicon.yml and
# the gold set follows.
MIN_TOKEN_LEN = 6
# Generic substance and form words: present in the lexicon as aliases, but far
# too common in free text to score named-entity recall against.
STOP_TOKENS = {"generic", "unspecified", "combined", "systemic", "topical",
               "progesterone", "oestrogen", "estrogen", "estradiol", "oestradiol",
               "sprays", "patches", "tablets", "capsules", "lozenge", "pessary",
               # Compound classes and lab markers: valid aliases for normalising a
               # mention, but far too broad to score a specific entity's recall on.
               # A post about phytoestrogens is not necessarily about red clover.
               "isoflavones", "phytoestrogens", "ferritin"}


def named_entity_tokens() -> dict[str, str]:
    """token -> canonical_id, for tokens distinctive enough to score recall on."""
    from pipeline.rules.lexicon import load as load_lexicon

    out: dict[str, str] = {}
    for entry in load_lexicon().entries.values():
        if not entry.name_must_appear:
            continue
        for alias in (entry.id.replace("_", " "), *entry.aliases):
            a = alias.strip().lower()
            if " " in a or len(a) < MIN_TOKEN_LEN or a in STOP_TOKENS:
                continue
            # A token claimed by two entities cannot be scored unambiguously.
            if a in out and out[a] != entry.id:
                out.pop(a)
                STOP_TOKENS.add(a)
                continue
            out[a] = entry.id
    return out


# (text, field, value, must_be_rejected, note)
NEGATIVE_CONTROLS: list[tuple[str, str, object, bool, str]] = [
    ("I'm not a neurologist but I do know the fluctuation theory",
     "profession", "neurologist", True, "negated profession"),
    ("I have never been to Vancouver", "city", "Vancouver", True, "negated location"),
    ("black cohosh did nothing for me", "race_ethnicity", "Black", True,
     "supplement name, not a demographic"),
    ("as a Black woman my doctor dismissed me", "race_ethnicity", "Black", False,
     "genuine self-identification"),
    ("I am 150 years old", "age", 150, True, "implausible age"),
    ("I am 51 and still getting them", "age", 51, False, "plausible stated age"),
    ("as an accountant I stare at spreadsheets all day", "profession", "accountant",
     False, "stated profession"),
    ("here in Toronto the wait lists are brutal", "city", "Toronto", False,
     "stated location"),
]
