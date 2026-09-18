"""The demographic guard. Both directions matter: a guard that rejects everything
passes a reject-only suite."""

import pytest

from pipeline.rules.negation import check


@pytest.mark.parametrize(
    "text,field,value,note",
    [
        ("I'm not a neurologist but I do know the fluctuation theory",
         "profession", "neurologist", "negated profession"),
        ("I have never been to Vancouver", "city", "Vancouver", "negated location"),
        ("I am 150 years old", "age", 150, "implausible age"),
        ("I am 12 and my mum has hot flashes", "age", 12, "below plausible range"),
        ("nothing about where I live", "city", "Toronto", "value absent from the text"),
    ],
)
def test_rejects(text, field, value, note):
    ok, why = check(field, value, text)
    assert ok is False, f"should reject: {note}"
    assert why, "a rejection must carry a reason"


@pytest.mark.parametrize(
    "text,field,value",
    [
        ("as an accountant I stare at spreadsheets all day", "profession", "accountant"),
        ("I AM a therapist, the irony is not lost on me", "profession", "therapist"),
        ("here in Toronto the wait lists are brutal", "city", "Toronto"),
        ("I am 51 and still getting them", "age", 51),
    ],
)
def test_accepts(text, field, value):
    ok, why = check(field, value, text)
    assert ok is True, f"should accept, got: {why}"


def test_decoy_blocks_a_supplement_masquerading_as_a_demographic(domain):
    """92 of 101 occurrences of 'black' in the real corpus are 'black cohosh'."""
    ok, why = check("race_ethnicity", "Black", "black cohosh did nothing for me")
    assert ok is False and "black cohosh" in why


def test_decoy_does_not_block_a_genuine_self_identification(domain):
    ok, _ = check("race_ethnicity", "Black", "as a Black woman my doctor dismissed me")
    assert ok is True


def test_decoy_and_real_mention_in_one_document_still_accepts(domain):
    ok, _ = check("race_ethnicity", "Black",
                  "I tried black cohosh. As a Black woman I felt unheard.")
    assert ok is True


def test_free_form_fields_are_not_required_to_appear_verbatim():
    ok, _ = check("caregiving", "teens and an aging parent", "my kids and my mum")
    assert ok is True


def test_empty_value_is_rejected():
    assert check("profession", None, "anything")[0] is False
    assert check("profession", "", "anything")[0] is False
