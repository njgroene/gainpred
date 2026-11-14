"""Tests for the longest-run text format (P2) and the tokenizer's generic hooks.

P9's "Format" acceptance test applies here too: a question converts to
characters and back unchanged, checked on many random strings.
"""

import random

import pytest

from gainpred.formats import (
    RUN_QUESTIONS,
    Addition,
    Run,
    RunQuestion,
    Unreadable,
    example_text,
    is_run_correct,
    longest_run,
    max_run_answer_chars,
    parse_run_answer,
    parse_run_example,
    parse_run_question,
    prompt_text,
    question_text,
    runs,
    solve,
    solve_run,
)
from gainpred.tokenizer import EOS_ID, decode, encode, make_prompts, make_training_batch, read_generated


def random_questions(n: int, seed: int) -> list[RunQuestion]:
    rng = random.Random(seed)
    return [RunQuestion(rng.choice(RUN_QUESTIONS), "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 25))))
            for _ in range(n)]


# --- the protocol's own examples (P2) ------------------------------------------


def test_examples_from_the_protocol():
    q = RunQuestion("runlen", "493036")
    assert question_text(q) == "Q: runlen(493036) = ?"
    assert prompt_text(q) == "Q: runlen(493036) = ?\nA: "
    assert solve_run(q) == Run("4", 1)  # all runs length 1: the leftmost wins
    assert example_text(q, solve_run(q)) == "Q: runlen(493036) = ?\nA: 4|1"
    assert solve_run(RunQuestion("runlen", "887476")) == Run("8", 2)


def test_edge_run_questions():
    assert question_text(RunQuestion("leftrun", "493036")) == "Q: leftrun(493036) = ?"
    assert question_text(RunQuestion("rightrun", "493036")) == "Q: rightrun(493036) = ?"
    assert solve_run(RunQuestion("leftrun", "887476")) == Run("8", 2)
    assert solve_run(RunQuestion("rightrun", "887476")) == Run("6", 1)
    assert solve_run(RunQuestion("rightrun", "1222")) == Run("2", 3)


def test_shin_table_1_examples():
    """Table 1 of Shin et al.: 3111 -> 1|3, 022227 -> 2|4, 01111 -> 1|4, 11200 -> 1|2, 0111111200 -> 1|6."""
    for string, (symbol, length) in {"3111": ("1", 3), "022227": ("2", 4), "3111022227": ("2", 4),
                                     "01111": ("1", 4), "11200": ("1", 2), "0111111200": ("1", 6)}.items():
        assert longest_run(string) == Run(symbol, length)


def test_leftmost_run_wins_ties():
    assert longest_run("1122") == Run("1", 2)
    assert longest_run("12233") == Run("2", 2)


def test_runs_split_and_recombine():
    for q in random_questions(500, 1):
        rs = runs(q.string)
        assert "".join(r.symbol * r.length for r in rs) == q.string
        assert all(a.symbol != b.symbol for a, b in zip(rs, rs[1:]))  # maximal


def test_strings_may_start_with_zero():
    assert question_text(RunQuestion("runlen", "022227")) == "Q: runlen(022227) = ?"


def test_bad_questions_and_runs_are_rejected():
    for kind, string in [("runlength", "123"), ("runlen", ""), ("runlen", "12a"), ("runlen", "１２")]:
        with pytest.raises(ValueError):
            RunQuestion(kind, string)
    with pytest.raises(ValueError):
        Run("x", 1)
    with pytest.raises(ValueError):
        Run("1", 0)


# --- round trips (P9 "Format") ---------------------------------------------------


def test_question_and_answer_round_trip():
    for q in random_questions(2000, 2):
        assert parse_run_question(question_text(q)) == q
        assert parse_run_example(example_text(q, solve_run(q))) == (q, solve_run(q))


@pytest.mark.parametrize("text, expected", [("4|1", Run("4", 1)), ("0|12", Run("0", 12)), ("9|99", Run("9", 99))])
def test_parse_answer_readable(text, expected):
    assert parse_run_answer(text) == expected


@pytest.mark.parametrize("text", ["", "4", "4|", "|1", "4|01", "4|0", "44|1", "4 | 1", "4|1|", "a|1", "4|-1", "4|1\n"])
def test_parse_answer_unreadable(text):
    answer = parse_run_answer(text)
    assert isinstance(answer, Unreadable) and answer.raw == text


def test_scoring_needs_symbol_and_length():
    q = RunQuestion("runlen", "887476")
    assert is_run_correct(q, Run("8", 2))
    assert not is_run_correct(q, Run("7", 2))
    assert not is_run_correct(q, Run("8", 3))
    assert not is_run_correct(q, Unreadable("8|2"))
    assert max_run_answer_chars(q) == 3 and max_run_answer_chars(RunQuestion("runlen", "1" * 20)) == 4


# --- the tokenizer works for both tasks (generic hooks) --------------------------


def test_every_character_is_in_the_vocabulary():
    for q in random_questions(200, 3):
        assert decode(encode(example_text(q, solve_run(q)))) == example_text(q, solve_run(q))


def test_training_batch_and_prompts_for_longest_run():
    examples = [(RunQuestion("runlen", "493036"), Run("4", 1)), (RunQuestion("leftrun", "8874760000"), Run("8", 2))]
    tokens, mask = make_training_batch(examples)
    for row, m, (q, a), prompt in zip(tokens.tolist(), mask.tolist(), examples, make_prompts([q for q, _ in examples])):
        assert row[: len(prompt)] == prompt  # test-time prompts match how training rows begin
        graded = [t for t, keep in zip(row, m) if keep]
        assert decode(graded[:-1]) == f"{a.symbol}|{a.length}" and graded[-1] == EOS_ID
        assert not any(m[: len(prompt)])  # the question is read, not graded


def test_mixed_batches_still_work_for_addition():
    tokens, _ = make_training_batch([(Addition(6133, 8894), 15027)])
    assert decode(tokens[0].tolist()[:-1]) == "Q: 6133 + 8894 = ?\nA: 15027"
    assert solve(Addition(1, 2)) == 3


def test_read_generated_with_the_run_parser():
    assert read_generated(encode("8|2") + [EOS_ID], parse=parse_run_answer) == Run("8", 2)
    assert isinstance(read_generated(encode("8|2"), parse=parse_run_answer), Unreadable)  # did not stop
    assert isinstance(read_generated(encode("8|02") + [EOS_ID], parse=parse_run_answer), Unreadable)
    # The default parser is still addition's.
    assert isinstance(read_generated(encode("8|2") + [EOS_ID]), Unreadable)
    assert read_generated(encode("15027") + [EOS_ID]) == 15027
