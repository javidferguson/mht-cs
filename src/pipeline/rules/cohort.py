"""Derive a treatment's user cohort from per-member mention counts.

"Anyone who ever mentioned it" is not a cohort - it returns 84 of 100 members for
Climara, most of whom were answering someone else's post. The real users show up
as a separate mode in the distribution.

Two independent one-dimensional thresholding methods are run, and **their
agreement is the evidence**: where a distinct cohort exists they select the same
people; where none does, they diverge and the caller is told so rather than handed
a number. No threshold is chosen by hand anywhere.

Lived in both notebooks as a copy-paste pair before it lived here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cohort:
    members: frozenset[str]
    gap_cut: float | None
    otsu_cut: float | None
    agree: bool

    def __len__(self) -> int:
        return len(self.members)


def _largest_gap(desc: list[float]) -> float:
    """Midpoint of the biggest drop between consecutive sorted counts."""
    gaps = [desc[i] - desc[i + 1] for i in range(len(desc) - 1)]
    i = max(range(len(gaps)), key=gaps.__getitem__)
    return (desc[i] + desc[i + 1]) / 2


def _otsu(desc: list[float]) -> float:
    """Cut maximising between-class variance (Otsu's method, 1-D)."""
    best, cut = -1.0, desc[-1]
    for t in sorted(set(desc)):
        hi = [v for v in desc if v >= t]
        lo = [v for v in desc if v < t]
        if not hi or not lo:
            continue
        w = len(hi) * len(lo) * (sum(hi) / len(hi) - sum(lo) / len(lo)) ** 2
        if w > best:
            best, cut = w, t
    return cut


def derive(counts: dict[str, int], min_members: int = 4) -> Cohort:
    """counts: member id -> number of self-attributed mentions of one treatment."""
    if len(counts) < min_members:
        return Cohort(frozenset(counts), None, None, False)

    desc = sorted((float(v) for v in counts.values()), reverse=True)
    gap, otsu = _largest_gap(desc), _otsu(desc)
    by_gap = frozenset(m for m, n in counts.items() if n >= gap)
    by_otsu = frozenset(m for m, n in counts.items() if n >= otsu)
    agree = bool(by_gap) and by_gap == by_otsu
    return Cohort(by_gap, gap, otsu, agree)
