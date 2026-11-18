"""Check 3 on addition (amendment 22): the teacher-with-pieces wrapper and the check itself."""
import random

from gainpred.check3 import GREEDY, check_3_addition, teacher_with_pieces, unreadable_pieces
from gainpred.data import draw_hard
from gainpred.formats import Unreadable
from gainpred.teachers import TEACHERS, Decision, PerfectCalculator, solve


class Sloppy:
    """A deterministic adder that is wrong (or unreadable) on some problems, like a model."""

    def __call__(self, problems, temperature=0.0):
        out = []
        for p in problems:
            h = hash((p.a, p.b)) % 7
            out.append(Unreadable("x") if h == 0 else solve(p) + (h == 1) * 10 ** (len(str(p.a)) - 1))
        return out


class Sampler:
    """A seeded 'sampling' adder for sample and vote: right about half the time, else off by 1-3."""

    def __init__(self):
        self.rng = random.Random(0)

    def __call__(self, problems, temperature=0.0):
        return [solve(p) if self.rng.random() < 0.5 else solve(p) + self.rng.randint(1, 3) for p in problems]


def problems(digits=10, n=300, seed=1):
    rng = random.Random(seed)
    return [draw_hard(rng, digits) for _ in range(n)]


def test_decisions_are_the_teachers_own():
    for digits in (8, 11, 15):
        ps = problems(digits)
        for solver in (PerfectCalculator(), Sloppy()):
            for name in GREEDY:
                ours = [d for d, _ in teacher_with_pieces(name, ps, solver)]
                # repr: an Unreadable is never == anything, not even an identical copy
                assert repr(ours) == repr(TEACHERS[name](ps, solver)), (name, digits)


def test_a_joined_answer_never_matches_a_piece_sum():
    for digits in range(8, 16):
        ps = problems(digits)
        for solver in (PerfectCalculator(), Sloppy()):
            for name in GREEDY:
                pairs = teacher_with_pieces(name, ps, solver)
                assert check_3_addition(pairs) == 0.0, (name, digits)


def test_a_voted_answer_always_matches_a_sample():
    pairs = teacher_with_pieces("sample_and_vote", problems(12), Sampler())
    assert any(d.kept is None for d, _ in pairs) and any(d.kept is not None for d, _ in pairs)
    assert check_3_addition(pairs) == 1.0
    assert all(len(pieces) == 5 for _, pieces in pairs)


def test_set_aside_problems_are_skipped_and_unreadables_never_match():
    p = problems(n=3)
    pairs = [(Decision(p[0], 5, ()), (5, 7)), (Decision(p[1], None, ()), (Unreadable("5"),)),
             (Decision(p[2], 9, ()), (Unreadable("9"),))]
    assert check_3_addition(pairs) == 0.5  # 2 kept: one matches; the set-aside one is not counted
    assert check_3_addition([(Decision(p[0], None, ()), (1,))]) is None
    assert unreadable_pieces(pairs) == (2, 4)


def test_agreement_records_every_cut_it_built():
    ps = problems(12, n=5)
    for d, pieces in teacher_with_pieces("agreement", ps, PerfectCalculator()):
        from gainpred.teachers import cuts_of
        assert len(pieces) == sum(len(c) for c in cuts_of(d.problem))
    for d, pieces in teacher_with_pieces("no_carry", ps, PerfectCalculator()):
        from gainpred.teachers import cuts_of
        assert len(pieces) == len(cuts_of(d.problem)[0])
