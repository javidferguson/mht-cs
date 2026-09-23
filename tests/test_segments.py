"""Segmentation rules: the buckets every presentation chart cuts by.

These are the functions that decide which bar a member lands in, so a silent
misclassification moves a person from one finding to another. The cases below are
the real values from the corpus, including the ones that were wrong first time.
"""

import pytest

from pipeline.rules.segments import (
    AGE_BANDS, CONSIDERED, NEVER, TOOK,
    age_band, country_of_city, exposure, profession_group,
)


# --- age -----------------------------------------------------------------

@pytest.mark.parametrize("age,expected", [
    (38, "<45"), (44, "<45"), (45, "45-49"), (49, "45-49"),
    (50, "50-54"), (54, "50-54"), (55, "55-59"), (59, "55-59"),
    (60, "60+"), (64, "60+"),
])
def test_age_band_boundaries(age, expected):
    """Every boundary, because an off-by-one moves real members between bars."""
    assert age_band(age) == expected


def test_age_band_accepts_the_shapes_the_export_actually_holds():
    assert age_band("52") == "50-54"      # CSV read without dtype coercion
    assert age_band(52.0) == "50-54"      # parquet float column


@pytest.mark.parametrize("bad", [None, "", "nan", "unknown", float("nan")])
def test_age_band_missing_is_none_not_a_bucket(bad):
    """`float('nan')` is truthy and once produced a literal 'nan' country bucket
    with 37 members in it. Missing must never become a category."""
    assert age_band(bad) is None


def test_age_bands_constant_matches_what_the_function_emits():
    assert set(AGE_BANDS) == {age_band(a) for a in (38, 46, 52, 57, 64)}


# --- location ------------------------------------------------------------

@pytest.mark.parametrize("city,country", [
    ("Vancouver", "Canada"), ("Calgary", "Canada"), ("Toronto", "Canada"),
    ("Melbourne", "Australia"), ("Sydney", "Australia"),
    ("Dublin", "UK & Ireland"), ("Edinburgh", "UK & Ireland"),
    ("Manchester", "UK & Ireland"),
    ("Portland", "United States"), ("Brooklyn", "United States"),
])
def test_country_of_city_maps_the_corpus_cities(city, country):
    assert country_of_city(city) == country


@pytest.mark.parametrize("written,expected", [
    ("dublin", "UK & Ireland"), ("toronto", "Canada"), ("sydney", "Australia"),
])
def test_city_lookup_is_case_insensitive(written, expected):
    """Members wrote both `Dublin` and `dublin`; they are one place."""
    assert country_of_city(written) == expected


@pytest.mark.parametrize("written,expected", [
    ("rural PA", "United States"),
    ("rural Pennsylvania", "United States"),
    ("suburban Detroit", "United States"),
    ("rural Vermont", "United States"),
    ("West Texas", "United States"),
])
def test_place_modifiers_are_stripped_before_lookup(written, expected):
    """`rural PA` and `PA` are the same country."""
    assert country_of_city(written) == expected


@pytest.mark.parametrize("bad", [None, "", "   ", "nan", "Atlantis"])
def test_unknown_place_is_none_never_guessed(bad):
    assert country_of_city(bad) is None


# --- profession ----------------------------------------------------------

@pytest.mark.parametrize("profession,group", [
    ("physician", "Healthcare"), ("GP", "Healthcare"), ("neurologist", "Healthcare"),
    ("ICU nurse", "Healthcare"), ("PT", "Healthcare"), ("physio", "Healthcare"),
    ("vet", "Healthcare"), ("psychotherapist", "Healthcare"),
    ("social worker", "Healthcare"),
    ("marketing director", "Corporate & professional"),
    ("HR director", "Corporate & professional"),
    ("litigation attorney", "Corporate & professional"),
    ("EA", "Corporate & professional"),
    ("software engineer", "Corporate & professional"),
    ("Logistics manager", "Corporate & professional"),
    ("real estate agent", "Corporate & professional"),
    ("florist", "Creative"), ("watercolor artist", "Creative"),
    ("graphic designer", "Creative"), ("freelance writer", "Creative"),
    ("barista", "Service & hospitality"), ("line cook", "Service & hospitality"),
    ("yoga instructor", "Service & hospitality"),
    ("teacher", "Education"), ("professor", "Education"),
    ("retired", "Not working"), ("stay-at-home parent", "Not working"),
])
def test_profession_group_assigns_the_corpus_values(profession, group):
    assert profession_group(profession) == group


def test_group_order_is_priority_and_education_precedes_corporate():
    """`librar` and `director` both match. Education is listed first, so the
    library director is not filed as a corporate executive. She was, first time."""
    assert profession_group("library director") == "Education"


def test_healthcare_precedes_education_so_a_school_nurse_is_clinical():
    """The case that rules out picking the longest matching key instead of the
    first: `school` is 6 characters and `nurse` is 5, so longest-match would file
    a school nurse under Education."""
    assert profession_group("school nurse") == "Healthcare"


@pytest.mark.parametrize("short_key_inside_a_word", ["teacher", "physician", "realtor"])
def test_two_letter_keys_do_not_fire_inside_longer_words(short_key_inside_a_word):
    """`ea` and `pt` are real profession strings in the corpus but must not match
    the `ea` in `teacher` or turn every word containing them into a bucket."""
    assert profession_group(short_key_inside_a_word) != "Not working"
    assert profession_group("teacher") == "Education"


def test_junk_extraction_is_dropped_not_bucketed():
    """One member's profession came back as the adverb `professionally`. Bucketing
    it would inflate a real group by a member who never stated a job."""
    assert profession_group("professionally") is None


@pytest.mark.parametrize("bad", [None, "", "   ", "nan"])
def test_missing_profession_is_none(bad):
    assert profession_group(bad) is None


# --- exposure ------------------------------------------------------------

def _stances(**kw):
    return {k: set(v) for k, v in kw.items()}


def test_current_or_past_counts_as_took():
    assert exposure(_stances(current=["climara"]), "climara") == TOOK
    assert exposure(_stances(past=["climara"]), "climara") == TOOK


def test_considering_or_rejecting_is_not_taking():
    """The distinction the whole cut rests on: someone weighing a patch has not
    used it, and folding them in is what turns a 51-member group into 66."""
    assert exposure(_stances(considering=["climara"]), "climara") == CONSIDERED
    assert exposure(_stances(rejected=["climara"]), "climara") == CONSIDERED


def test_took_wins_over_considered_when_both_are_present():
    """A member can reject it now and have used it before; they still took it."""
    both = _stances(past=["climara"], rejected=["climara"])
    assert exposure(both, "climara") == TOOK


def test_other_treatments_do_not_leak_into_the_cut():
    assert exposure(_stances(current=["estradot", "prometrium"]), "climara") == NEVER


def test_empty_profile_is_never():
    assert exposure({}, "climara") == NEVER
    assert exposure(_stances(current=[], past=[]), "climara") == NEVER
