"""Tests for teachers (P2 combining, P5 rules), checks (P8) and the paper test (P9)."""

import random

import pytest
import torch

from gainpred.checks import check_1_and_teachable, run_checks
from gainpred.cuts import ADDITION_CUTS, carry_crosses
from gainpred.data import draw_hard
from gainpred.formats import Addition, Unreadable, solve
from gainpred.paper_test import paper_test
from gainpred.teachers import (
    Decision,
    ModelSolver,
    PerfectCalculator,
    combine,
    fits,
    majority,
    no_selection,
    sample_and_vote,
    no_carry,
    agreement,
)

PERFECT = PerfectCalculator()
NO_CARRY = Addition(61332071, 88945512)  # cut 4|4: 2071 + 5512 = 7583 fits
CARRY = Addition(61332071, 88948512)  # 2071 + 8512 = 10583 overflows


class Scripted:
    """A solver with planted mistakes: ``wrong`` maps a problem to the answer to give."""

    def __init__(self, wrong: dict[Addition, object]):
        self.wrong = wrong
        self.calls = 0

    def __call__(self, problems, temperature=0.0):
        self.calls += 1
        return [self.wrong.get(p, solve(p)) for p in problems]


# --- combining (P2) ----------------------------------------------------------


def test_combine_pads_lower_sums_and_lets_the_leading_sum_grow():
    assert combine([15027, 7583], (4, 4)) == 150277583
    assert combine([15027, 83], (4, 4)) == 150270083  # padded to width 4
    assert combine([1502, 7, 583], (3, 2, 3)) == 150207583


def test_combine_of_an_overflowing_lower_sum_is_one_digit_too_long():
    built = combine([15027, 10583], (4, 4))
    assert built == 1502710583 and built != solve(CARRY)


def test_combine_with_an_unreadable_piece_is_unreadable():
    assert isinstance(combine([15027, Unreadable("7x")], (4, 4)), Unreadable)
    assert isinstance(combine([0, 7583], (4, 4)), Unreadable)  # "07583" is not canonical


def test_fits():
    assert fits([15027, 9999], (4, 4))
    assert not fits([15027, 10000], (4, 4))
    assert not fits([15027, Unreadable("")], (4, 4))


def test_built_answer_is_right_exactly_when_no_carry_crosses():
    rng = random.Random(0)
    for digits, cuts in ADDITION_CUTS.items():
        for _ in range(200):
            p = draw_hard(rng, digits)
            assert (no_carry([p], PERFECT)[0].kept == solve(p)) == (not carry_crosses(p, cuts[0]))


# --- teachers (P5) -------------------------------------------------------------


def test_no_carry_keeps_only_when_lower_sums_fit():
    kept, set_aside = no_carry([NO_CARRY, CARRY], PERFECT)
    assert kept.kept == solve(NO_CARRY) and set_aside.kept is None


def test_no_carry_keeps_a_wrong_piece_that_still_fits():
    # The model gets the lower piece wrong but within width: the no-carry rule cannot tell.
    solver = Scripted({Addition(2071, 5512): 7584})
    (d,) = no_carry([NO_CARRY], solver)
    assert d.kept == 150277584 != solve(NO_CARRY)


def test_no_selection_keeps_even_wrong_answers_but_not_unreadable_ones():
    assert no_selection([CARRY], PERFECT)[0].kept == 1502710583
    solver = Scripted({Addition(6133, 8894): Unreadable("<pad>")})
    assert no_selection([CARRY], solver)[0].kept is None


def test_agreement_needs_two_identical_answers():
    # 8 digits: cuts 4|4, 3|5, 5|3; kept iff at least two cuts have no carry.
    cuts = ADDITION_CUTS[8]
    p = NO_CARRY
    ok = [not carry_crosses(p, c) for c in cuts]
    d = agreement([p], PERFECT)[0]
    assert len(d.candidates) == 3
    assert (d.kept == solve(p)) == (sum(ok) >= 2)


def test_agreement_rejects_one_wrong_piece_when_others_disagree():
    # Break cut 1's lower piece: cuts 2 and 3 still agree on the right answer.
    solver = Scripted({Addition(2071, 5512): 7584})
    d = agreement([NO_CARRY], solver)[0]
    assert d.kept == solve(NO_CARRY)
    assert d.candidates[0] == 150277584


def test_two_unreadable_answers_never_agree():
    assert majority([Unreadable("x"), Unreadable("x"), 5], needed=2) is None
    assert majority([5, 7, 5], needed=2) == 5


def test_sample_and_vote_needs_three_of_five():
    p = NO_CARRY
    answers = iter([1, 1, 2, 3, 4,   # problem 0 never reaches 3
                    9, 9, 9, 1, 2])  # problem 1 does

    class Sampler:
        def __call__(self, problems, temperature=0.0):
            assert temperature == 1.0 and len(problems) == 10
            # Sampled as [p0, p1] x 5: sample k of problem i is at i + 2k.
            seq = list(answers)
            by_problem = [seq[:5], seq[5:]]
            return [by_problem[i % 2][i // 2] for i in range(10)]

    d0, d1 = sample_and_vote([p, CARRY], Sampler())
    assert d0.kept is None and d1.kept == 9


def test_teachers_call_the_solver_once_per_batch():
    solver = Scripted({})
    agreement([NO_CARRY, CARRY], solver)
    assert solver.calls == 1


# --- checks (P8) ---------------------------------------------------------------


def test_check_1_and_teachable_share():
    decisions = [Decision(NO_CARRY, 5, ()), Decision(NO_CARRY, 6, ()), Decision(NO_CARRY, None, ())]
    direct = [5, 7, 8]
    check_1, teachable = check_1_and_teachable(decisions, direct)
    assert check_1 == pytest.approx(1 / 2)  # of 2 kept, 1 differs from the direct answer
    assert teachable == pytest.approx(1 / 3)  # of 3 problems, 1 offers something new


def test_a_perfect_model_has_nothing_to_learn():
    report = run_checks(no_carry([NO_CARRY, CARRY], PERFECT), PERFECT)
    assert report.check_1 == 0.0 and report.teachable == 0.0 and report.kept == 1


# --- with a real (tiny, untrained) model ----------------------------------------


def test_model_solver_returns_answers_in_order_and_samples_reproducibly():
    from gainpred.config import ModelConfig
    from gainpred.model import Transformer
    from gainpred.tokenizer import PAD_ID, VOCAB_SIZE

    torch.manual_seed(0)
    model = Transformer(ModelConfig(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, n_layers=1, d_model=32, n_heads=2, d_ff=64))
    problems = [Addition(123, 456), NO_CARRY, Addition(1234, 5678)]
    cpu = torch.device("cpu")
    def text(answers):  # Unreadable compares by identity, so compare raw text
        return [a.raw if isinstance(a, Unreadable) else a for a in answers]

    # Answers come back in the order asked, although problems are sorted by
    # length inside. (Comparing to one-at-a-time would not work: an untrained
    # model never stops, and the length cap depends on the batch.)
    forward = ModelSolver(model, cpu)(problems)
    backward = ModelSolver(model, cpu)(problems[::-1])
    assert len(forward) == 3 and text(forward) == text(backward[::-1])
    # Sampling with the same seed gives the same answers.
    a = ModelSolver(model, cpu, seed=1)(problems, temperature=1.0)
    b = ModelSolver(model, cpu, seed=1)(problems, temperature=1.0)
    assert text(a) == text(b)


# --- the paper test (P9 acceptance: "cutting and combining") ---------------------


def test_paper_test_matches_the_protocol():
    table = paper_test(n=3000)
    for digits in ADDITION_CUTS:
        # Perfect pieces: the filtering teachers never keep a wrong answer.
        for name in ("no_carry", "agreement", "sample_and_vote"):
            assert table[name][digits]["right"] == 1.0
        # No selection keeps everything, right exactly when the no-carry rule would keep it.
        assert table["no_selection"][digits]["kept"] == 1.0
        assert table["no_selection"][digits]["right"] == table["no_carry"][digits]["kept"]
    # P3: about three draws per kept answer up to 12 digits, about fourteen at 15.
    for digits in range(8, 13):
        assert 1 / table["agreement"][digits]["kept"] == pytest.approx(3, abs=0.4)
    assert 1 / table["agreement"][15]["kept"] == pytest.approx(14, abs=2)


def test_rule_errors_match_shin_et_al_table_7():
    """Shin et al., Table 7: unfiltered addition labels up to 9 digits have 60.8% rule errors."""
    table = paper_test(n=3000)
    rule_errors = 1 - (table["no_selection"][8]["right"] + table["no_selection"][9]["right"]) / 2
    assert rule_errors == pytest.approx(0.608, abs=0.025)


def test_old_teacher_names_in_early_logs_still_resolve():
    from gainpred.teachers import TEACHER_NAMES, TEACHERS, teacher_key

    assert teacher_key("their_rule") == "no_carry" and teacher_key("two_cuts_agree") == "agreement"
    assert teacher_key("no_selection") == "no_selection"
    assert set(TEACHER_NAMES) == set(TEACHERS)
