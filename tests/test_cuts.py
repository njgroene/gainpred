"""Tests for the P4 cut table and carry detection."""

import random

import pytest

from gainpred.cuts import (
    ADDITION_CUTS,
    EASY_DIGITS,
    carry_crosses,
    cut_positions,
    piece_problems,
    piece_starts,
    split,
)
from gainpred.data import draw_hard
from gainpred.formats import Addition, solve

ALL_CUTS = [(size, i + 1, cut) for size, cuts in ADDITION_CUTS.items() for i, cut in enumerate(cuts)]


# ----------------------------------------------------------------------------
# The two rules that govern the table (P4)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("size, number, cut", ALL_CUTS)
def test_pieces_are_easy_sizes_and_cover_the_number(size, number, cut):
    """Rule 1: every piece is 3-7 digits, and the pieces add up to the size."""
    assert sum(cut) == size
    assert all(width in EASY_DIGITS for width in cut)


@pytest.mark.parametrize("size", ADDITION_CUTS)
def test_no_two_cuts_share_a_position(size):
    """Rule 2: otherwise a carry at the shared position would make two cuts
    wrong in the same way, and their agreement would be no evidence (P5)."""
    cuts = ADDITION_CUTS[size]
    for i in range(len(cuts)):
        for j in range(i + 1, len(cuts)):
            assert not cut_positions(cuts[i]) & cut_positions(cuts[j]), (size, cuts[i], cuts[j])


# ----------------------------------------------------------------------------
# Splitting
# ----------------------------------------------------------------------------


def test_split_and_positions():
    assert split(61332071, (4, 4)) == [6133, 2071]
    assert split(123456789012345, (6, 3, 6)) == [123456, 789, 12345]
    assert cut_positions((4, 5, 4)) == {4, 9}
    assert piece_starts((4, 5, 4)) == {0, 4, 9}


def test_split_rejects_wrong_length():
    with pytest.raises(ValueError):
        split(1234567, (4, 4))


# ----------------------------------------------------------------------------
# Carries
# ----------------------------------------------------------------------------


def side_by_side(problem: Addition, cut) -> int:
    """The built answer from *true* piece sums (P2): lower sums zero-padded to
    their piece's width, the leading sum written as is."""
    sums = [p.a + p.b for p in piece_problems(problem, cut)]
    text = str(sums[0]) + "".join(str(s).zfill(w) for s, w in zip(sums[1:], cut[1:]))
    return int(text)


def test_docstring_example():
    problem = Addition(61332071, 88945512)
    assert not carry_crosses(problem, (4, 4))
    assert side_by_side(problem, (4, 4)) == solve(problem) == 150277583
    assert carry_crosses(Addition(61332071, 88948512), (4, 4))  # 2071 + 8512 overflows


def test_carry_crosses_exactly_when_side_by_side_is_wrong():
    """The defining property: with perfect piece answers, the built answer is
    right if and only if no carry crosses a cut. Checked for every cut of every
    size on random hard problems."""
    rng = random.Random(0)
    seen = {True: 0, False: 0}
    for size, cuts in ADDITION_CUTS.items():
        for _ in range(2000):
            problem = draw_hard(rng, size)
            for cut in cuts:
                crosses = carry_crosses(problem, cut)
                seen[crosses] += 1
                assert (side_by_side(problem, cut) == solve(problem)) == (not crosses)
    assert min(seen.values()) > 1000  # both cases were actually exercised


def test_leading_piece_may_overflow():
    """An overflowing *leading* piece is not a carry across a cut: its extra
    digit just becomes the answer's first digit."""
    problem = Addition(99991111, 99991111)  # 9999 + 9999 = 19998 in the leading piece
    assert not carry_crosses(problem, (4, 4))
    assert side_by_side(problem, (4, 4)) == solve(problem)


def test_five_cuts_at_13_to_15_and_cut_1_unchanged():
    """Decision point 1: five cuts at 13-15 digits, three elsewhere; cut 1 is the old one."""
    from gainpred.cuts import THREE_CUTS

    assert {size: len(cuts) for size, cuts in ADDITION_CUTS.items()} == {
        8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 5, 14: 5, 15: 5}
    assert all(ADDITION_CUTS[s][:3] == THREE_CUTS[s] for s in THREE_CUTS)
