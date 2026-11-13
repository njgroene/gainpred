"""The cuts of protocol P4: how a hard addition problem is split into easy pieces.

The idea in one example
=======================
The model starts out good at 3-7 digit additions (the "easy" sizes) and bad at
anything longer. To answer an 8-digit problem without being able to do 8-digit
additions, cut both numbers at the same place into easy pieces, add the pieces
separately, and write the piece sums side by side (P2, "built addition answer").
Cut 1 for 8 digits is 4|4::

        6133 2071          pieces of a:  6133 | 2071
      + 8894 5512          pieces of b:  8894 | 5512
                           piece sums:  15027 | 7583
      built answer:  150277583   (true answer 150277583: correct)

This only works if no carry crosses the cut. When the lower piece sum
overflows its width, e.g. 2071 + 8512 = 10583 (5 digits in a 4-digit piece),
the carry belongs to the piece above and the built answer is wrong. "Their
rule" (Shin et al., P5) keeps a built answer only when no carry crosses a cut.

Why three cuts
==============
Each size has three different cuts. If two cuts give the identical answer, that
is evidence it is right ("two cuts agree", P5). For this to be real evidence,
no two cuts of a size may cut at the same position: a carry at a shared
position would make both wrong *in the same way*, so they would agree on a
wrong answer. The tests check this rule for every row.

Representation
==============
A cut is a tuple of piece widths, read from the most significant (leftmost)
digit: ``(4, 5, 4)`` means "the first 4 digits, the next 5, the last 4".
This module only describes cuts and detects carries; *combining* model answers
into a built answer belongs to the teachers (P5) and comes later.
"""

from __future__ import annotations

from gainpred.formats import Addition

Cut = tuple[int, ...]

EASY_DIGITS = range(3, 8)
"""Sizes the model learns directly (P1: "both numbers 3-7 digits"). Every piece
of every cut must be one of these, so that each piece looks like an easy problem."""

ADDITION_CUTS: dict[int, tuple[Cut, Cut, Cut]] = {
    #   size:  (cut 1,      cut 2,      cut 3)        -- P4, verbatim
    8: ((4, 4), (3, 5), (5, 3)),
    9: ((3, 6), (4, 5), (5, 4)),
    10: ((5, 5), (4, 6), (6, 4)),
    11: ((4, 7), (5, 6), (6, 5)),
    12: ((6, 6), (5, 7), (7, 5)),
    13: ((6, 7), (7, 6), (4, 5, 4)),
    14: ((7, 7), (4, 5, 5), (6, 4, 4)),
    15: ((5, 5, 5), (4, 7, 4), (6, 3, 6)),
}
"""P4. Cut 1 (index 0) supplies the answer for "their rule" and "no selection"."""

HARD_DIGITS = range(min(ADDITION_CUTS), max(ADDITION_CUTS) + 1)
"""Sizes the model must learn by self-improvement: 8-15 digits (P3)."""


def cut_positions(cut: Cut) -> frozenset[int]:
    """Where a cut splits the number, counted in digits from the left.

    ``(4, 5, 4)`` -> ``{4, 9}``: after the 4th digit and after the 9th.
    """
    positions, total = set(), 0
    for width in cut[:-1]:
        total += width
        positions.add(total)
    return frozenset(positions)


def piece_starts(cut: Cut) -> frozenset[int]:
    """0-based index (from the left) of each piece's first digit.

    ``(4, 5, 4)`` -> ``{0, 4, 9}``. P2 forbids a 0 at any of these positions,
    under any of the three cuts, so no piece looks unlike an easy problem.
    """
    return frozenset({0} | cut_positions(cut))


def split(number: int, cut: Cut) -> list[int]:
    """Cut a number into pieces: ``split(61332071, (4, 4))`` -> ``[6133, 2071]``."""
    digits = str(number)
    if len(digits) != sum(cut):
        raise ValueError(f"{number} has {len(digits)} digits; cut {cut} needs {sum(cut)}")
    pieces, start = [], 0
    for width in cut:
        pieces.append(int(digits[start : start + width]))
        start += width
    return pieces


def piece_problems(problem: Addition, cut: Cut) -> list[Addition]:
    """The easy problems a cut produces, most significant first."""
    return [Addition(a, b) for a, b in zip(split(problem.a, cut), split(problem.b, cut))]


def carry_crosses(problem: Addition, cut: Cut) -> bool:
    """Does a carry cross any position where this cut splits the numbers?

    True iff some piece other than the leading one has a sum that does not fit
    in its width (P5, their rule: "every lower piece sum fits in its piece's
    width"). The leading piece may overflow freely: its extra digit simply
    becomes the first digit of the answer (P2).

    Why checking each piece's *own* sum is enough: a carry can only arrive at
    a piece if some piece below it overflowed, and that piece would already
    have been caught. So there are no "hidden" chained carries to look for.

    Equivalently: ``carry_crosses`` is False exactly when writing the true
    piece sums side by side gives the true answer (checked in the tests).
    """
    pieces = piece_problems(problem, cut)
    return any(p.a + p.b >= 10**width for p, width in zip(pieces[1:], cut[1:]))
