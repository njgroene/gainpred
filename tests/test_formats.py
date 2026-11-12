"""Tests for the addition text format (protocol P2).

The central property is P9's "Format" acceptance test: a problem converts to
characters and back unchanged. We check it on many random problems of every
size used in the study, rather than a handful of hand-picked ones, because
format bugs tend to hide in edge cases (a carry that adds a digit, a 1-digit
operand, a zero).
"""

import random

import pytest

from gainpred.formats import (
    Addition,
    Unreadable,
    example_text,
    is_correct,
    max_answer_chars,
    parse_answer,
    parse_example,
    parse_question,
    prompt_text,
    question_text,
    solve,
)


def random_problems(n: int, seed: int) -> list[Addition]:
    """n problems with both operands of the same length, 1 to 15 digits.

    Lengths 3-15 cover every size in the study (P3, P7); 1-2 add edge cases.
    """
    rng = random.Random(seed)
    problems = []
    for _ in range(n):
        digits = rng.randint(1, 15)
        low, high = 10 ** (digits - 1), 10**digits - 1
        problems.append(Addition(rng.randint(low, high), rng.randint(low, high)))
    return problems


# ----------------------------------------------------------------------------
# The exact layout
# ----------------------------------------------------------------------------


def test_example_from_the_protocol():
    """P2's own example, character for character."""
    problem = Addition(6133, 8894)
    assert question_text(problem) == "Q: 6133 + 8894 = ?"
    assert prompt_text(problem) == "Q: 6133 + 8894 = ?\nA: "
    assert solve(problem) == 15027
    assert example_text(problem, solve(problem)) == "Q: 6133 + 8894 = ?\nA: 15027"


def test_digits_are_written_most_significant_first():
    """P2: "digits are written forward". 1000 + 1 is 1001, not 1001 reversed."""
    assert example_text(Addition(1000, 25), 1025).endswith("A: 1025")


# ----------------------------------------------------------------------------
# Round trips (P9 "Format")
# ----------------------------------------------------------------------------


def test_question_round_trip():
    for problem in random_problems(5000, seed=0):
        assert parse_question(question_text(problem)) == problem


def test_example_round_trip_with_true_and_wrong_answers():
    """Works for any answer, not just the right one: teachers can be wrong (P5)."""
    rng = random.Random(1)
    for problem in random_problems(5000, seed=1):
        for total in (solve(problem), rng.randint(0, 10**16)):
            assert parse_example(example_text(problem, total)) == (problem, total)


# ----------------------------------------------------------------------------
# Reading answers strictly
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("15027", 15027),
        ("0", 0),
        ("1000000000000000", 10**15),
    ],
)
def test_readable_answers(text, expected):
    assert parse_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",  # model wrote nothing
        "015027",  # leading zero: not the canonical form
        "00",
        "15 027",  # stray space
        " 15027",
        "15027 ",
        "+15027",
        "-5",
        "1502a",
        "15027\n",
        "A: 15027",  # repeated the prefix
        "１５",  # full-width digits: str.isdigit() would accept these, our parser must not
    ],
)
def test_unreadable_answers(text):
    answer = parse_answer(text)
    assert isinstance(answer, Unreadable)
    assert answer.raw == text  # the raw text is kept for logs (P2: never dropped)


def test_unreadable_answers_are_always_distinct():
    """P2: an unreadable answer is "its own distinct answer". Two identical
    garbled strings must not count as two cuts agreeing (P5)."""
    first, second = parse_answer("12x"), parse_answer("12x")
    assert first != second
    assert first == first  # but an object still equals itself
    assert len({first, second}) == 2


def test_scoring():
    problem = Addition(6133, 8894)
    assert is_correct(problem, 15027)
    assert not is_correct(problem, 15028)
    assert not is_correct(problem, parse_answer("015027"))  # unreadable = wrong


# ----------------------------------------------------------------------------
# Questions are written by our code, so malformed ones must fail loudly
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["Q: 6133 + 8894 = ", "Q: 6133+8894 = ?", "Q: 06133 + 8894 = ?", "6133 + 8894", "Q: 1 + -2 = ?"],
)
def test_malformed_questions_raise(text):
    with pytest.raises(ValueError):
        parse_question(text)


def test_negative_operands_rejected():
    with pytest.raises(ValueError):
        Addition(-1, 2)


def test_max_answer_chars_covers_every_true_answer():
    """Generation is capped at this length, so it must fit the longest sum."""
    for problem in random_problems(5000, seed=2) + [Addition(9, 9), Addition(99999, 99999)]:
        assert len(str(solve(problem))) <= max_answer_chars(problem)
