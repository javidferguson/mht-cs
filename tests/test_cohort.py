"""Cohort derivation: no threshold is chosen by hand.

Two independent methods run over the per-member counts, and their AGREEMENT is the
evidence that a distinct cohort exists at all.
"""

from pipeline.rules.cohort import derive


def test_bimodal_distribution_yields_agreement_and_the_heavy_users():
    """Shaped like the real Climara data: a long tail of people who mentioned it
    once or twice, and a handful who talk about it constantly."""
    counts = {f"tail_{i}": n for i, n in enumerate([1, 1, 2, 2, 3, 3, 4, 5, 6, 7])}
    counts.update({f"user_{i}": n for i, n in enumerate([48, 50, 55, 62, 65])})

    c = derive(counts)
    assert c.agree is True
    assert c.members == {f"user_{i}" for i in range(5)}
    assert 7 < c.gap_cut < 48


def test_flat_distribution_reports_no_distinct_cohort():
    """A commodity like magnesium: everyone mentions it a similar amount, so the
    two methods disagree and the caller is told rather than handed a number."""
    counts = {f"m{i}": n for i, n in enumerate(range(5, 26))}
    assert derive(counts).agree is False


def test_too_few_members_does_not_invent_a_threshold():
    c = derive({"a": 10, "b": 2})
    assert c.agree is False and c.gap_cut is None and c.otsu_cut is None


def test_empty_input_is_safe():
    c = derive({})
    assert len(c) == 0 and c.agree is False


def test_result_len_is_the_member_count():
    counts = {f"t{i}": 1 for i in range(9)}
    counts.update({"big1": 90, "big2": 95})
    assert len(derive(counts)) == 2
