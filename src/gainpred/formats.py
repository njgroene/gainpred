"""How problems and answers are written as text (protocol P2). Addition only, so far.

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

What is *not* here yet
    * the "built addition answer" rule (piece sums side by side) -- it needs
      the cuts of P4, so it lives with the teachers;
    * "no piece starts with 0" -- a rule for drawing problems, so it lives
      with the generators;
    * longest-run questions.
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


def question_text(problem: Addition) -> str:
    """``Addition(6133, 8894)`` -> ``"Q: 6133 + 8894 = ?"``"""
    return f"Q: {problem.a} + {problem.b} = ?"


def prompt_text(problem: Addition) -> str:
    """What the model is given at test time: question + answer prefix."""
    return question_text(problem) + ANSWER_PREFIX


def answer_text(total: int) -> str:
    """The graded characters: ``15027`` -> ``"15027"``."""
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
