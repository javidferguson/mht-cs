"""The normalize-stage guards.

Both of these shipped broken in one direction while working in the other, twice.
Every test here exists because something got through.
"""

import pytest

from pipeline.stages.normalize import _fix_attribution, _fix_kind


# ---------------------------------------------------------------- kind guard
def test_kind_guard_corrects_symptom_entity_filed_as_treatment(lex):
    """Shipped broken: _fix_kind ran only on the symptom branch, so 161 treatment
    mentions resolving to symptom entities kept the wrong kind and showed up as
    treatment rows in the analysis."""
    kind, fixed = _fix_kind(lex, "treatment", "vasomotor")
    assert (kind, fixed) == ("symptom", True)


def test_kind_guard_corrects_treatment_entity_filed_as_symptom(lex):
    """The mirror. 292 mentions went this way."""
    kind, fixed = _fix_kind(lex, "symptom", "climara")
    assert (kind, fixed) == ("treatment", True)


def test_kind_guard_leaves_correct_kinds_alone(lex):
    assert _fix_kind(lex, "treatment", "climara") == ("treatment", False)
    assert _fix_kind(lex, "symptom", "vasomotor") == ("symptom", False)


def test_kind_guard_passes_through_unmapped_mentions(lex):
    """No canonical id means the lexicon has no opinion."""
    assert _fix_kind(lex, "treatment", None) == ("treatment", False)
    assert _fix_kind(lex, "symptom", "not_in_the_lexicon") == ("symptom", False)


# --------------------------------------------------------- attribution guard
def test_downgrades_self_when_the_context_is_advice_to_someone_else():
    """A member was recorded as a current Climara user on the strength of
    'worth asking your GP whether your Climara dose is optimised'."""
    ctx = ("It may be worth asking your GP specifically about whether your Climara "
           "dose is optimised for this")
    assert _fix_attribution("self", ctx) == ("other", True)


def test_keeps_self_when_a_first_person_marker_is_present():
    """Deliberately conservative - 'I'm on Climara, your mileage may vary' is still
    the author's own use."""
    ctx = "I'm on Climara myself and it helped, though your mileage may vary"
    assert _fix_attribution("self", ctx) == ("self", False)


def test_never_upgrades_a_non_self_attribution():
    for exp in ("other", "general"):
        assert _fix_attribution(exp, "your GP might suggest it") == (exp, False)


def test_no_context_means_no_change():
    assert _fix_attribution("self", None) == ("self", False)
    assert _fix_attribution("self", "") == ("self", False)


@pytest.mark.parametrize("ctx", [
    "you should ask about it",
    "your doctor may suggest it",
    "ur gyno is giving you real talk",
])
def test_second_person_only_contexts_are_downgraded(ctx):
    assert _fix_attribution("self", ctx)[0] == "other"
