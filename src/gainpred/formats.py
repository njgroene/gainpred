"""How problems and answers are written as text (protocol P2): addition and longest run.

The model only ever sees characters. This module is the single place that
decides what those characters are, so that training data, prompts, test sets
and the reading of model output can never drift apart.

One training example, in full (P2)::

    Q: 6133 + 8894 = ?
    A: 15027

As one string that is ``"Q: 6133 + 8894 = ?\\nA: 15027"``, and it splits into
three parts that matter for training and generation:

    "Q: 6133 + 8894 = ?"     question       -- read by the model, not graded
    "\\nA: "                  answer prefix  -- read by the model, not graded
    "15027"                  answer         -- graded (P1: loss on answer only)

The question plus the answer prefix is the **prompt**: at test time we give the
model exactly that and let it write the answer. Ending the prompt with "A: "
(rather than leaving the model to write it) means every generated character
is part of the answer, which keeps parsing simple.

Digits are written forward, most significant first, as in Shin et al. (P2).

P10 assumption: the separator between question and answer is a newline. P2
only says the answer comes "then"; the paper's code may differ.

Longest run (experiment 2) is the second half of this module; its own
explanation sits just above that code. The addition functions below are not
changed by it: ``question_text`` and ``answer_text`` simply hand a longest-run
object on to its own function, so the tokenizer can build batches of either.

What is *not* here
    * the "built addition answer" rule (piece sums side by side) -- it needs
      the cuts of P4, so it lives with the teachers;
    * "no piece starts with 0" -- a rule for drawing problems, so it lives
      with the generators;
    * how a hard longest-run string is joined from two pieces -- that is
      ``gainpred.run_data`` and ``gainpred.run_teachers``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# Layout constants. Changing any of these changes every sequence the model
# sees, so they are defined once here and imported everywhere else.
# ----------------------------------------------------------------------------
SEPARATOR = "\n"
"""Between question and answer (P10 assumption: newline)."""

ANSWER_PREFIX = SEPARATOR + "A: "
"""Ends every prompt. Read by the model but never graded."""


# ----------------------------------------------------------------------------
# Problems and answers as Python objects
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Addition:
    """An addition problem ``a + b``.

    ``frozen=True`` makes it immutable and hashable, so problems can be put in
    sets -- handy for checking that no test problem leaks into training (P7).
    """

    a: int
    b: int

    def __post_init__(self) -> None:
        if self.a < 0 or self.b < 0:
            raise ValueError("operands must be non-negative")

    @property
    def n_digits(self) -> int:
        """The problem's "size" for the study: the longer operand's length.

        P1 draws both numbers with the same length, so in practice the two
        lengths are equal.
        """
        return max(len(str(self.a)), len(str(self.b)))


@dataclass(frozen=True, eq=False)
class Unreadable:
    """Model output that is not a well-formed answer (P2, "unreadable answers").

    It is never dropped: it counts as **wrong** when scoring, and as its own
    distinct answer when comparing answers (e.g. two cuts in P5).

    ``eq=False`` means two Unreadable objects are equal only if they are the
    *same object*, so ``Unreadable("x") != Unreadable("x")``. This implements
    "distinct": two garbled outputs never count as agreeing, even when the
    garbage is identical, because agreeing on nonsense is no evidence of a right
    answer. (Our reading of P2; flip to ``eq=True`` to compare raw text instead.)

    ``raw`` keeps the exact text for logging and debugging.
    """

    raw: str


Answer = int | Unreadable
"""What reading a model's output yields: a number, or Unreadable."""


def solve(problem: Addition) -> int:
    """The true answer, by plain arithmetic (P9: "every answer matches plain arithmetic")."""
    return problem.a + problem.b


# ----------------------------------------------------------------------------
# Objects -> text
# ----------------------------------------------------------------------------


def question_text(problem: Addition | RunQuestion) -> str:
    """``Addition(6133, 8894)`` -> ``"Q: 6133 + 8894 = ?"``

    A longest-run question is passed on to ``run_question_text``. This one
    ``isinstance`` test is the whole "generic hook": it lets the tokenizer's
    ``make_prompts`` and ``make_training_batch`` work for both tasks unchanged.
    """
    if isinstance(problem, RunQuestion):
        return run_question_text(problem)
    return f"Q: {problem.a} + {problem.b} = ?"


def prompt_text(problem: Addition | RunQuestion) -> str:
    """What the model is given at test time: question + answer prefix."""
    return question_text(problem) + ANSWER_PREFIX


def answer_text(total: int | Run) -> str:
    """The graded characters: ``15027`` -> ``"15027"``; ``Run("4", 1)`` -> ``"4|1"``."""
    if isinstance(total, Run):
        return run_answer_text(total)
    if total < 0:
        raise ValueError("an addition answer cannot be negative")
    return str(total)


def example_text(problem: Addition, total: int) -> str:
    """A full training example: prompt followed by answer.

    ``total`` is passed in rather than computed, because in this study the
    answer we train on is often *not* the true one -- it is whatever a teacher
    built (P5), right or wrong.
    """
    return prompt_text(problem) + answer_text(total)


# ----------------------------------------------------------------------------
# Text -> objects
# ----------------------------------------------------------------------------

# A canonical non-negative integer: "0", or digits not starting with 0.
_NUMBER = r"(0|[1-9][0-9]*)"
_QUESTION = re.compile(rf"Q: {_NUMBER} \+ {_NUMBER} = \?")
_ANSWER = re.compile(_NUMBER)


def parse_question(text: str) -> Addition:
    """``"Q: 6133 + 8894 = ?"`` -> ``Addition(6133, 8894)``.

    Raises ValueError on anything else. Questions are written by our own code,
    never by the model, so a malformed one is a bug and should fail loudly.
    """
    match = _QUESTION.fullmatch(text)
    if match is None:
        raise ValueError(f"not an addition question: {text!r}")
    return Addition(int(match[1]), int(match[2]))


def parse_answer(text: str) -> Answer:
    """Read what the model wrote after "A: ". Never raises.

    Strict: only the canonical form of a number is readable. ``"015027"``,
    ``"15 027"``, ``"+15027"``, ``""`` and anything containing other characters
    are all Unreadable.

    Why strict? Answers are compared as *strings* (P5: "the identical string"),
    and scoring compares against the canonical answer. A lenient parser that
    turned "015027" into 15027 would make two different strings look like the
    same answer -- inventing agreement the model did not produce.
    """
    if _ANSWER.fullmatch(text) is None:
        return Unreadable(text)
    return int(text)


def parse_example(text: str) -> tuple[Addition, Answer]:
    """Split a full example back into (problem, answer). Inverse of example_text."""
    question, sep, answer = text.partition(ANSWER_PREFIX)
    if not sep:
        raise ValueError(f"no answer prefix in {text!r}")
    return parse_question(question), parse_answer(answer)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def is_correct(problem: Addition, answer: Answer) -> bool:
    """True iff the answer is readable and equals the true sum.

    An Unreadable answer is never equal to an int, so it scores as wrong (P2).
    """
    return answer == solve(problem)


def max_answer_chars(problem: Addition) -> int:
    """Longest a correct answer can be: one digit more than the longer operand.

    Used to cap generation. A model that keeps writing past this is wrong
    anyway, so there is no need to let it run longer.
    """
    return problem.n_digits + 1


# ============================================================================
# Longest run (experiment 2)
# ============================================================================
#
# The task (Shin et al., Section A.1.3 "Max run length")
# ------------------------------------------------------
# The input is a string of digits. A *run* is a maximal block of equal
# symbols: "887476" splits into the runs 88 | 7 | 4 | 7 | 6. The answer is the
# symbol and the length of the longest run, written ``symbol|length``::
#
#     Q: runlen(887476) = ?
#     A: 8|2                      the run "88": symbol 8, length 2
#
# Ties go to the leftmost run (P2). In "493036" every run has length 1, so the
# answer is the first symbol::
#
#     Q: runlen(493036) = ?
#     A: 4|1
#
# The edge-run questions (P2) ask for the run touching one end of the string,
# in the same ``symbol|length`` form::
#
#     Q: leftrun(887476) = ?      A: 8|2        (the run at the left edge)
#     Q: rightrun(887476) = ?     A: 6|1        (the run at the right edge)
#
# Why edge runs at all? Shin et al. only ask ``runlen``. Their way of building
# a hard answer ("pick the longer", P5) keeps the longer of the two pieces'
# longest runs, which is wrong whenever a run continues across the join. To
# build a *right* answer one also needs the run that sticks out of each piece
# at the join -- the left piece's right-edge run and the right piece's
# left-edge run. That is "join at the cut" (P5), and it is the reason the
# model is also taught ``leftrun`` and ``rightrun`` (P10, departures).
#
# Strings may start with 0 (Shin et al.'s own example "022227" does): unlike
# addition, a string is not a number, and "no piece starts with 0" is an
# addition rule only.
# ============================================================================

RUN_QUESTIONS = ("runlen", "leftrun", "rightrun")
"""The three longest-run question types (P2). The names are what the model
reads, so they double as the code names."""


@dataclass(frozen=True)
class RunQuestion:
    """One longest-run question: ``RunQuestion("runlen", "493036")``.

    ``kind`` is one of ``RUN_QUESTIONS``. ``string`` is kept as text (not an
    int) because leading zeros matter: "022227" and "22227" are different
    strings with different runs.
    """

    kind: str
    string: str

    def __post_init__(self) -> None:
        if self.kind not in RUN_QUESTIONS:
            raise ValueError(f"unknown question type {self.kind!r}; expected one of {RUN_QUESTIONS}")
        if not self.string or not all(c in "0123456789" for c in self.string):
            raise ValueError(f"a longest-run string is one or more digits 0-9, not {self.string!r}")

    @property
    def length(self) -> int:
        """The question's size for the study: the number of symbols (P1: 6-10 easy, P3: 12-20 hard)."""
        return len(self.string)


@dataclass(frozen=True)
class Run:
    """A run, or an answer to a longest-run question: symbol and length.

    ``Run("4", 1)`` is written ``"4|1"``. Frozen and hashable, like Addition,
    so answers can be counted and compared with ``==`` (P5 compares answers as
    identical strings; two Runs are equal exactly when their texts are).
    """

    symbol: str
    length: int

    def __post_init__(self) -> None:
        if len(self.symbol) != 1 or self.symbol not in "0123456789":
            raise ValueError(f"a run's symbol is one digit, not {self.symbol!r}")
        if self.length < 1:
            raise ValueError("a run has length at least 1")


RunAnswer = Run | Unreadable
"""What reading a model's longest-run output yields: a Run, or Unreadable."""


def runs(string: str) -> list[Run]:
    """Split a string into its runs, left to right.

    ``"887476"`` -> ``[Run("8", 2), Run("7", 1), Run("4", 1), Run("7", 1), Run("6", 1)]``
    """
    out: list[Run] = []
    for char in string:
        if out and out[-1].symbol == char:
            out[-1] = Run(char, out[-1].length + 1)
        else:
            out.append(Run(char, 1))
    return out


def longest_run(string: str) -> Run:
    """The longest run; the leftmost one on ties (P2).

    ``max`` returns the *first* maximal element it meets, and ``runs`` lists
    runs left to right, so this is leftmost-on-ties by construction.
    """
    return max(runs(string), key=lambda run: run.length)


def solve_run(question: RunQuestion) -> Run:
    """The true answer to any longest-run question (P9: answers match a plain solver)."""
    all_runs = runs(question.string)
    if question.kind == "leftrun":
        return all_runs[0]
    if question.kind == "rightrun":
        return all_runs[-1]
    return longest_run(question.string)


# ----------------------------------------------------------------------------
# Longest run: objects <-> text
# ----------------------------------------------------------------------------


def run_question_text(question: RunQuestion) -> str:
    """``RunQuestion("runlen", "493036")`` -> ``"Q: runlen(493036) = ?"``"""
    return f"Q: {question.kind}({question.string}) = ?"


def run_answer_text(answer: Run) -> str:
    """``Run("4", 1)`` -> ``"4|1"``: symbol, a bar, then the length (P2)."""
    return f"{answer.symbol}|{answer.length}"


_RUN_QUESTION = re.compile(r"Q: (runlen|leftrun|rightrun)\(([0-9]+)\) = \?")
# One digit, a bar, then a canonical positive length (no leading zero, not 0).
_RUN_ANSWER = re.compile(r"([0-9])\|([1-9][0-9]*)")


def parse_run_question(text: str) -> RunQuestion:
    """``"Q: runlen(493036) = ?"`` -> ``RunQuestion("runlen", "493036")``.

    Raises ValueError on anything else: questions are written by our code,
    never by the model, so a malformed one is a bug.
    """
    match = _RUN_QUESTION.fullmatch(text)
    if match is None:
        raise ValueError(f"not a longest-run question: {text!r}")
    return RunQuestion(match[1], match[2])


def parse_run_answer(text: str) -> RunAnswer:
    """Read what the model wrote after "A: " to a longest-run question. Never raises.

    Strict, for the same reason as ``parse_answer``: answers are compared as
    strings (P5), so only the canonical form is readable. ``"4|01"``,
    ``"4|0"``, ``"44|1"``, ``"4 | 1"``, ``"|1"``, ``"4|"`` and ``""`` are all
    Unreadable. Readability is only about form: ``"4|99"`` for a 6-symbol
    string is readable, and simply wrong.
    """
    match = _RUN_ANSWER.fullmatch(text)
    if match is None:
        return Unreadable(text)
    return Run(match[1], int(match[2]))


def parse_run_example(text: str) -> tuple[RunQuestion, RunAnswer]:
    """Split a full longest-run example back into (question, answer)."""
    question, sep, answer = text.partition(ANSWER_PREFIX)
    if not sep:
        raise ValueError(f"no answer prefix in {text!r}")
    return parse_run_question(question), parse_run_answer(answer)


# ----------------------------------------------------------------------------
# Longest run: scoring
# ----------------------------------------------------------------------------


def is_run_correct(question: RunQuestion, answer: RunAnswer) -> bool:
    """True iff the answer is readable and equals the true run (symbol *and* length).

    An answer with the right length but the wrong symbol is wrong: P2 asks for
    both, and the symbol is what tells which of two equally long runs was meant.
    """
    return answer == solve_run(question)


def max_run_answer_chars(question: RunQuestion) -> int:
    """Longest a correct answer can be: one symbol, the bar, and the string's
    length written out (a run cannot be longer than the string)."""
    return 2 + len(str(question.length))
