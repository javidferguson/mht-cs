"""Deterministic context location. The model's own `evidence` omitted the surface
in 63% of treatment mentions, so this is what actually shows a term in use."""

from pipeline.rules.context import locate

TEXT = ("ok so i have been on the estradiol patch for about four months now and "
        "honestly the difference is a lot but the adhesion is terrible in summer heat")


def test_locates_an_exact_surface():
    r = locate(TEXT, "estradiol patch")
    assert r and r.exact is True
    assert "estradiol patch" in r.context
    assert TEXT[r.start:r.end].lower() == "estradiol patch"


def test_falls_back_to_the_most_distinctive_token():
    """'the patch' is not present verbatim; 'patch' is."""
    r = locate(TEXT, "the patch")
    assert r and r.exact is False
    assert r.matched.lower() == "patch"


def test_strict_mode_refuses_the_loose_fallback():
    """A loose match lands on unrelated sentences - 'migraines' matched a word in a
    passage about a podcast. strict=True is what stops that."""
    assert locate(TEXT, "the patch", strict=True) is None


def test_absent_surface_returns_none():
    assert locate(TEXT, "nothing like this appears") is None


def test_empty_inputs_are_safe():
    assert locate("", "climara") is None
    assert locate(TEXT, "") is None


def test_context_does_not_start_mid_word():
    long = "x" * 400 + " the adhesion is terrible " + "y" * 400
    r = locate(long, "adhesion")
    assert r
    body = r.context.strip("…").strip()
    assert not body.startswith("x" * 5)


def test_near_anchor_picks_the_closer_occurrence():
    t = "climara early on. " + "filler " * 60 + "climara again at the end."
    first = locate(t, "climara")
    later = locate(t, "climara", near="again at the end")
    assert first and later and later.start > first.start
