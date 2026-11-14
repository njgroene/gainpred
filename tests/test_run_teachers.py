"""Tests for the longest-run teachers (P5), the check-2 diagnostic (P4), check 3 (P8) and the paper test (P9)."""

import itertools
import random

import pytest
import torch

from gainpred.checks import check_3, hands_back, run_checks_longest_run
from gainpred.formats import Run, RunQuestion, Unreadable, longest_run, solve_run
from gainpred.run_data import HARD_LENGTHS, RANDOM_JOIN, RUN_ACROSS, HardRun, draw_hard
from gainpred.run_paper_test import boundary_share, paper_test
from gainpred.run_teachers import (
    RUN_TEACHER_NAMES,
    RUN_TEACHERS,
    PerfectRunSolver,
    RunModelSolver,
    agreement_from_built,
    build_diagnostic,
    diagnostic_positions,
    edges_differ,
    join_at_the_cut,
    no_selection,
    pick_the_longer,
)

PERFECT = PerfectRunSolver()
# The module docstring's example: "493077" + "777476", a run of five 7s across the join.
ACROSS = HardRun("493077777476", 6, RUN_ACROSS, 2, 3)
# Shin et al.'s Figure 9: "493036" + "887476", edges 6 and 8 differ; label 8|2.
SHIN = HardRun("493036887476", 6, RANDOM_JOIN)


class Scripted:
    """A solver with planted mistakes: ``wrong`` maps a question to the answer to give."""

    def __init__(self, wrong):
        self.wrong, self.calls = wrong, 0

    def __call__(self, questions, temperature=0.0):
        self.calls += 1
        return [self.wrong.get(q, solve_run(q)) for q in questions]


# --- the teachers on hand-made examples ------------------------------------------


def test_shin_figure_9_example_is_kept_by_every_teacher():
    for name, teacher in RUN_TEACHERS.items():
        (d,) = teacher([SHIN], PERFECT)
        assert d.kept == Run("8", 2), name


def test_pick_the_longer_misses_the_run_across_the_join():
    (d,) = pick_the_longer([ACROSS], PERFECT)
    assert d.candidates == (Run("7", 2), Run("7", 3)) and d.kept == Run("7", 3) != longest_run(ACROSS.string)


def test_pick_the_longer_takes_the_left_on_ties():
    p = HardRun("115678229876", 6, RANDOM_JOIN)
    assert pick_the_longer([p], PERFECT)[0].kept == Run("1", 2)


def test_edges_differ_sets_aside_equal_edges_and_reads_the_string():
    assert edges_differ([ACROSS], PERFECT)[0].kept is None
    # The rule reads the string, so a wrong piece answer is still kept when the edges differ.
    solver = Scripted({RunQuestion("runlen", "887476"): Run("7", 5)})
    (d,) = edges_differ([SHIN], solver)
    assert d.kept == Run("7", 5) and d.source == 0


def test_join_at_the_cut_builds_the_run_across_the_join():
    (d,) = join_at_the_cut([ACROSS], PERFECT)
    assert d.candidates == (Run("7", 2), Run("7", 5), Run("7", 3)) and d.kept == Run("7", 5)
    assert d.pieces == (Run("7", 2), Run("7", 2), Run("7", 3), Run("7", 3))


def test_join_at_the_cut_is_always_right_with_right_pieces():
    """P5's candidates + leftmost-on-ties is a valid rule. Checked on every 10th string
    of length 12 over three symbols (joined at 6; few symbols means many long and tied
    runs) and on random hard strings of every length."""
    problems = [HardRun("".join(s), 6, RANDOM_JOIN) for s in itertools.islice(itertools.product("012", repeat=12), 0, None, 10)]
    rng = random.Random(0)
    problems += [draw_hard(rng, rng.choice(list(HARD_LENGTHS))) for _ in range(5000)]
    for d in join_at_the_cut(problems, PERFECT):
        assert d.kept == longest_run(d.problem.string), d.problem


def test_pick_the_longer_is_right_whenever_the_edges_differ():
    rng = random.Random(1)
    for _ in range(3000):
        p = draw_hard(rng, rng.choice(list(HARD_LENGTHS)))
        right = pick_the_longer([p], PERFECT)[0].kept == longest_run(p.string)
        if p.string[p.join - 1] != p.string[p.join]:
            assert right


def test_unreadable_pieces_are_set_aside():
    solver = Scripted({RunQuestion("runlen", "493077"): Unreadable("7|")})
    for teacher in (pick_the_longer, join_at_the_cut, no_selection):
        assert teacher([ACROSS], solver)[0].kept is None
    solver = Scripted({RunQuestion("leftrun", "777476"): Unreadable("")})
    assert join_at_the_cut([ACROSS], solver)[0].kept is None
    assert pick_the_longer([ACROSS], solver)[0].kept == Run("7", 3)  # it never asked


def test_no_selection_is_pick_the_longer():
    rng = random.Random(2)
    problems = [draw_hard(rng, 15) for _ in range(300)]
    assert no_selection(problems, PERFECT) == pick_the_longer(problems, PERFECT)


def test_teachers_call_the_solver_once_per_batch():
    solver = Scripted({})
    join_at_the_cut([ACROSS, SHIN], solver)
    edges_differ([ACROSS, SHIN], solver)
    assert solver.calls == 2


def test_teacher_names():
    assert set(RUN_TEACHER_NAMES) == set(RUN_TEACHERS)
    assert {"edges_differ", "pick_the_longer", "join_at_the_cut"} <= set(RUN_TEACHERS)


# --- check 3 ------------------------------------------------------------------------


def test_check_3():
    assert check_3([(Run("1", 2), (Run("1", 2), Run("3", 1))), (None, (Run("1", 1),))]) == 1.0
    assert check_3([(Run("7", 5), (Run("7", 2), Run("7", 3))), (Run("7", 3), (Run("7", 2), Run("7", 3)))]) == 0.5
    assert check_3([(None, ())]) is None
    u = Unreadable("x")
    assert check_3([(u, (u,))]) == 1.0  # same object; a different Unreadable never matches
    assert check_3([(Unreadable("x"), (Unreadable("x"),))]) == 0.0
    assert hands_back(1.0) and not hands_back(0.99) and not hands_back(None)


# --- the check-2 diagnostic (P4) ------------------------------------------------------


def test_diagnostic_cuts_are_the_nearest_6_to_10_splits():
    """Decision point 1: the nearest alternative splits keeping both pieces 6-10 long."""
    assert diagnostic_positions(ACROSS) == (6,)  # length 12: 6|6 is the only split
    assert diagnostic_positions(ACROSS, offset=3) == (6, 3, 9)  # the original reading
    long = HardRun("1234567890123456", 8, "random_join")
    assert diagnostic_positions(long) == (8, 7, 9)
    edge = HardRun("1234567890123456", 6, "random_join")  # 16: splits 6..10
    assert diagnostic_positions(edge) == (6, 7, 8)


def test_diagnostic_agreement_with_a_perfect_solver_is_complete():
    rng = random.Random(3)
    problems = [draw_hard(rng, 18) for _ in range(300)]
    built = build_diagnostic(problems, PERFECT)
    assert all(d.kept == longest_run(d.problem.string) and d.source == 0 for d in agreement_from_built(problems, built))


def test_diagnostic_with_pick_the_longer_disagrees_on_runs_across_the_join():
    built = build_diagnostic([ACROSS], PERFECT, rule="pick_the_longer", offset=3)  # the original reading
    # cut at 6: 7|3; cut at 3 ("493" | "077777476"): 7|5; cut at 9 ("493077777" | "476"): 7|5.
    assert [b.answer for b in built[0]] == [Run("7", 3), Run("7", 5), Run("7", 5)]
    assert agreement_from_built([ACROSS], built)[0].kept == Run("7", 5)


def test_longest_run_checks_with_a_perfect_solver():
    rng = random.Random(4)
    problems = [draw_hard(rng, 14) for _ in range(400)]
    report = run_checks_longest_run(pick_the_longer(problems, PERFECT), PERFECT)
    wrong = sum(d.kept != longest_run(d.problem.string) for d in pick_the_longer(problems, PERFECT)) / 400
    assert report.check_1 == pytest.approx(wrong)  # the perfect direct answer differs exactly where the label is wrong
    assert report.check_2_kept == 1.0 and report.check_3 == 1.0
    assert report.check_2b == pytest.approx(1 - wrong)
    join = run_checks_longest_run(join_at_the_cut(problems, PERFECT), PERFECT)
    assert join.check_1 == 0.0 and join.check_2b == 1.0 and join.check_3 < 1.0


# --- the paper test (P9) -----------------------------------------------------------


@pytest.fixture(scope="module")
def table():
    return paper_test(n=2000)


def test_paper_test_matches_the_protocol(table):
    for length in HARD_LENGTHS:
        assert table["join_at_the_cut"]["all"][length]["right"] == 1.0
        assert table["edges_differ"]["all"][length]["right"] == 1.0
        assert table["edges_differ"][RUN_ACROSS][length]["kept"] == 0.0
        assert table["pick_the_longer"]["all"][length]["kept"] == 1.0
        assert table["pick_the_longer"]["all"][length]["check_3"] == 1.0
        assert table["edges_differ"]["all"][length]["check_3"] == 1.0
        assert table["join_at_the_cut"]["all"][length]["check_3"] < 1.0


def test_paper_test_matches_shin_et_al_table_7(table):
    """Unfiltered rule errors at length <= 20: 5.9%; filtered kept: 90.0% (random joins)."""
    n = len(HARD_LENGTHS)
    errors = 1 - sum(table["pick_the_longer"][RANDOM_JOIN][s]["right"] for s in HARD_LENGTHS) / n
    kept = sum(table["edges_differ"][RANDOM_JOIN][s]["kept"] for s in HARD_LENGTHS) / n
    assert errors == pytest.approx(0.059, abs=0.01)
    assert kept == pytest.approx(0.90, abs=0.01)


def test_boundary_share_matches_shin_et_al_section_c7():
    assert boundary_share(per_length=300) == pytest.approx(0.662, abs=0.01)


# --- with a real (tiny, untrained) model ----------------------------------------------


def test_run_model_solver_returns_answers_in_order():
    from gainpred.config import ModelConfig
    from gainpred.model import Transformer
    from gainpred.tokenizer import PAD_ID, VOCAB_SIZE

    torch.manual_seed(0)
    model = Transformer(ModelConfig(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, n_layers=1, d_model=32, n_heads=2, d_ff=64))
    questions = [RunQuestion("runlen", "493036"), RunQuestion("leftrun", "4930368874761"), RunQuestion("rightrun", "1234567")]
    cpu = torch.device("cpu")

    def text(answers):
        return [a.raw if isinstance(a, Unreadable) else a for a in answers]

    forward = RunModelSolver(model, cpu)(questions)
    backward = RunModelSolver(model, cpu)(questions[::-1])
    assert len(forward) == 3 and text(forward) == text(backward[::-1])


def test_check_2_is_undefined_without_a_second_cut():
    """Length 12 has only the 6|6 split: checks 2 and 2b are None there, not 0."""
    rng = random.Random(5)
    problems = [draw_hard(rng, 12) for _ in range(50)]
    report = run_checks_longest_run(join_at_the_cut(problems, PERFECT), PERFECT)
    assert report.check_2_kept is None and report.check_2b is None
    problems = [draw_hard(rng, 16) for _ in range(50)]
    assert run_checks_longest_run(join_at_the_cut(problems, PERFECT), PERFECT).check_2_kept == 1.0
