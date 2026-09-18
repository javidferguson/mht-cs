"""The canonicalisation layer. Every aggregate in the product keys on its output."""

import pytest

from pipeline.rules import lexicon


def test_exact_alias_match(lex):
    assert lex.match("Climara").id == "climara"
    assert lex.match("climara patch").id == "climara"


def test_case_and_punctuation_are_folded(lex):
    assert lex.match("  CLIMARA!  ").id == "climara"


def test_leading_articles_are_stripped(lex):
    assert lex.match("the climara").id == "climara"
    assert lex.match("my hrt").id == "hrt_unspecified"


def test_containment_finds_the_entity_inside_a_longer_surface(lex):
    assert lex.match("estrogen (Climara patch) 0.05mg").id == "climara"


def test_longest_alias_wins_so_a_brand_never_collapses_into_its_class(lex):
    """Divigel must not be swallowed by the generic 'gel'. This is the whole
    reason phrases are sorted longest-first."""
    assert lex.match("Divigel").id == "divigel"
    assert lex.match("divigel pump").id == "divigel"
    assert lex.match("estradiol gel").id == "gel_generic"


def test_unknown_surface_returns_none_rather_than_guessing(lex):
    assert lex.match("something nobody charted") is None
    assert lex.match("") is None


def test_kind_defaults_to_treatment_and_is_read_from_the_entry(lex):
    assert lex.entries["climara"].kind == "treatment"
    assert lex.entries["vasomotor"].kind == "symptom"


def test_name_must_appear_flag_is_loaded(lex):
    assert lex.entries["climara"].name_must_appear is True
    assert lex.entries["hrt_unspecified"].name_must_appear is False


def test_duplicate_alias_across_entities_raises(tmp_path):
    """A token owned by two entities makes canonicalisation ambiguous. The loader
    must refuse rather than silently pick one - this fired for real on 'the pill'
    and on 'caffeine'."""
    bad = tmp_path / "dupe.yml"
    bad.write_text(
        "- id: a\n  label: A\n  aliases: [shared]\n"
        "- id: b\n  label: B\n  aliases: [shared]\n"
    )
    lexicon.load.cache_clear()
    with pytest.raises(ValueError, match="claimed by both"):
        lexicon.load(bad)
    lexicon.load.cache_clear()


def test_duplicate_id_raises(tmp_path):
    bad = tmp_path / "dupeid.yml"
    bad.write_text("- id: a\n  label: A\n- id: a\n  label: A again\n")
    lexicon.load.cache_clear()
    with pytest.raises(ValueError, match="duplicate"):
        lexicon.load(bad)
    lexicon.load.cache_clear()


@pytest.mark.parametrize(
    "raw,expected",
    [("Climara", "climara"), ("the  Gel ", "gel"), ("HRT!", "hrt"), ("", "")],
)
def test_normalise(raw, expected):
    assert lexicon.normalise(raw) == expected
