"""Extraction schema - the single source of truth for what the LLM returns.

Pydantic models here are converted to JSON Schema and handed to Ollama's
`format` parameter, which constrains decoding. Output is therefore parseable by
construction rather than by regex salvage.

Design notes:
  * The model emits the VERBATIM surface form it saw. It does NOT canonicalise -
    that is Phase 3's deterministic lexicon job, so aggregates stay stable across
    runs and across model versions.
  * `experienced_by` exists because ~80% of the corpus is replies, where the
    author frequently discusses someone else's treatment. Without it, the OP's
    Climara gets credited to every replier.
  * No free-text rationale fields. Output length dominates runtime, and `evidence`
    (a short verbatim quote) already provides the receipt.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Stance(StrEnum):
    CURRENT = "current"
    PAST = "past"
    CONSIDERING = "considering"
    REJECTED = "rejected"
    RECOMMENDED = "recommended"
    WARNED_AGAINST = "warned_against"
    UNCLEAR = "unclear"


class ExperiencedBy(StrEnum):
    SELF = "self"
    OTHER = "other"
    GENERAL = "general"


class Sentiment(StrEnum):
    """Constrained rather than a float: a free -1..1 number invites the model to
    emit 0.0 as a default, which it did for ~60% of mentions in the v1 gate."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class SymptomStatus(StrEnum):
    ONGOING = "ongoing"
    RESOLVED = "resolved"
    IMPROVED = "improved"
    WORSENED = "worsened"
    UNCLEAR = "unclear"


class TreatmentMention(BaseModel):
    surface: str = Field(description="The treatment exactly as written in the text.")
    stance: Stance
    experienced_by: ExperiencedBy
    # Required strings with an explicit sentinel. As `str | None` these came back
    # null on 610 of 610 mentions - the model will not volunteer an optional field.
    # Normalize converts "unspecified" back to NULL.
    dose: str = Field(
        min_length=3,
        description='Dose or frequency as written ("0.05mg", "twice a week"). "unspecified" if not stated.',
    )
    duration: str = Field(
        min_length=3,
        description='How long they have been on it ("8 months"). "unspecified" if not stated.',
    )
    sentiment: Sentiment
    evidence: str = Field(
        min_length=15,
        description="Verbatim quote, 15-120 characters. Never empty.",
    )


class SymptomMention(BaseModel):
    surface: str = Field(description="The symptom exactly as written in the text.")
    status: SymptomStatus
    experienced_by: ExperiencedBy
    evidence: str = Field(
        min_length=15,
        description="Verbatim quote, 15-120 characters. Never empty.",
    )


class SwitchEvent(BaseModel):
    """Both sides are REQUIRED strings, not optional.

    As `str | None` these render as anyOf[string, null], and under constrained
    decoding the model reliably took the null branch - "switched from Climara to
    Divigel" came back as from=None, to=None. Forcing a string with an explicit
    "unspecified" sentinel fixed it. Phase 3 drops rows where both sides are
    unspecified.
    """

    from_treatment: str = Field(
        min_length=3,
        description='Treatment moved AWAY from, verbatim. Use "unspecified" if not named.',
    )
    to_treatment: str = Field(
        min_length=3,
        description='Treatment moved TO, verbatim. Use "unspecified" if not named.',
    )
    reason: str = Field(
        min_length=3,
        description='Why they switched, as stated. "unspecified" if no reason is given.',
    )
    experienced_by: ExperiencedBy
    evidence: str = Field(
        min_length=15,
        description="Verbatim quote, 15-120 characters. Never empty.",
    )


class Demographics(BaseModel):
    """All fields null unless the author states them about THEMSELVES.

    race_ethnicity is self-report only and must never be inferred from names,
    locations or cultural references - see the prompt.
    """

    age: int | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None
    profession: str | None = None
    race_ethnicity: str | None = None
    has_partner: bool | None = None
    caregiving: str | None = None


class DocExtraction(BaseModel):
    treatments: list[TreatmentMention]
    symptoms: list[SymptomMention]
    switches: list[SwitchEvent]
    demographics: Demographics


def extraction_schema() -> dict:
    """JSON Schema handed to Ollama's `format` parameter."""
    return DocExtraction.model_json_schema()
