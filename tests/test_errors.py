"""Tests for gainpred.errors.error_profile."""

from gainpred.errors import error_profile
from gainpred.formats import Addition, Unreadable


def test_error_profile_classifies_each_kind_of_mistake():
    problems = [Addition(80781974, 46466177)] * 4  # true sum 127248151
    answers = [
        127248151,  # right
        127248121,  # right length, tens digit wrong
        12634811,  # one digit short, first two digits right
        Unreadable("1272<pad>"),
    ]
    p = error_profile(problems, answers)
    assert (p["problems"], p["right"], p["unreadable"]) == (4, 1, 1)
    assert p["length_minus_true"] == {"-1": 1, "0": 1}
    assert p["leading_digits_right"] == {"2": 1, "7": 1}
    assert p["wrong_place"] == {"1": 1}
    assert p["most_common_difference"] == {"-30": 1}
