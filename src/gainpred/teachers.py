"""Teachers: turn a model's answers to easy pieces into answers to hard problems (P2, P5).

The self-improvement loop in one paragraph
==========================================
The model can add 3-7 digit numbers but not longer ones. For each hard
(8-15 digit) problem we cut both numbers into easy pieces (``gainpred.cuts``),
ask the *model* to add each pair of pieces, and write its piece sums side by
side: the **built answer**. A **teacher** decides which built answers to keep.
Kept answers become training data for the next round, so the model learns from
its own (filtered) work. Nobody ever looks at the right answer: the whole point
is to see which filters let a model improve without one.

Combining piece sums (P2, "built addition answer")
==================================================
Write the piece sums side by side, most significant first. The leading sum is
written as it is (it may be one digit longer than its piece: that digit becomes
the first digit of the answer). Every lower sum is padded with zeros on the
left to its piece's width::

    a = 6133 0071, b = 8894 0012   (cut 4|4; P2 forbids pieces starting with 0,
                                    this is only to show the padding)
    piece sums: 15027 | 83  ->  "15027" + "0083"  ->  150270083

If a lower sum is *wider* than its piece (a carry crosses the cut, or the
model got the piece wrong), it is written as it is, so the built answer comes
out one digit too long -- and wrong. Teachers that check "every lower sum fits
its width" catch exactly this. If any piece answer is unreadable, or the built
string is not a canonical number, the built answer is unreadable (P2).

The four teachers (P5)
======================
``their_rule``
    Build from cut 1; keep it if every lower piece sum fits its width. With a
    perfect calculator this keeps exactly the problems where no carry crosses
    cut 1, and every kept answer is right.
``two_cuts_agree``
    Build from all three cuts; if at least two give the identical answer, keep
    it. Two *different* cuts can only agree on a wrong answer if the model
    makes matching mistakes, which is why P4 forbids shared cut positions.
``no_selection``
    Keep the cut-1 answer always (unless it is unreadable: an unreadable
    answer cannot be trained on, so it is set aside and counted).
``sample_and_vote``
    No cutting at all: ask the model the *hard* problem directly five times at
    temperature 1.0; keep an answer given at least three times. The control:
    does cutting add anything over the model's own direct attempts?

Solvers
=======
A teacher does not care what adds the pieces, only that something answers a
list of problems. ``PerfectCalculator`` is plain arithmetic, used for the
paper test (P9: "with a perfect calculator in place of the model, reproduces
the paper-test table"); ``ModelSolver`` asks a trained model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from gainpred.cuts import ADDITION_CUTS, Cut, piece_problems
from gainpred.formats import Addition, Answer, Unreadable, max_answer_chars, parse_answer, solve
from gainpred.model import Transformer
from gainpred.tokenizer import EOS_ID, make_prompts, read_generated

VOTE_SAMPLES = 5
VOTE_NEEDED = 3
VOTE_TEMPERATURE = 1.0


# ----------------------------------------------------------------------------
# Solvers: something that answers addition problems
# ----------------------------------------------------------------------------


class Solver(Protocol):
    def __call__(self, problems: list[Addition], temperature: float = 0.0) -> list[Answer]: ...


class PerfectCalculator:
    """Always right, at any length and any temperature."""

    def __call__(self, problems: list[Addition], temperature: float = 0.0) -> list[Answer]:
        return [solve(p) for p in problems]


class ModelSolver:
    """Ask a trained model. Greedy at temperature 0; seeded sampling above 0.

    Problems are answered in batches, sorted by length so a batch wastes little
    padding, and returned in the original order.
    """

    def __init__(self, model: Transformer, device: torch.device, batch_size: int = 1000, seed: int = 0):
        self.model = model.eval()
        self.device = device
        self.batch_size = batch_size
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.calls = 0  # problems answered so far
        self.generated_tokens = 0  # characters written, including end tokens (P9: log tokens)

    def __call__(self, problems: list[Addition], temperature: float = 0.0) -> list[Answer]:
        from gainpred.train import autocast  # avoid an import cycle

        order = sorted(range(len(problems)), key=lambda i: problems[i].n_digits)
        answers: list[Answer | None] = [None] * len(problems)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            chunk = [problems[i] for i in idx]
            limit = max(max_answer_chars(p) for p in chunk) + 1
            with autocast(self.device):
                outputs = self.model.generate(make_prompts(chunk), eos_id=EOS_ID, max_new_tokens=limit,
                                              temperature=temperature, generator=self.generator)
            for i, ids in zip(idx, outputs):
                answers[i] = read_generated(ids)
                self.generated_tokens += len(ids)
        self.calls += len(problems)
        return answers  # type: ignore[return-value]


# ----------------------------------------------------------------------------
# Building answers
# ----------------------------------------------------------------------------


def combine(piece_answers: list[Answer], cut: Cut) -> Answer:
    """P2: write the piece sums side by side, padding lower sums to their width."""
    if any(isinstance(a, Unreadable) for a in piece_answers):
        raw = "|".join(a.raw if isinstance(a, Unreadable) else str(a) for a in piece_answers)
        return Unreadable(f"built from unreadable piece: {raw}")
    leading, *lower = piece_answers
    text = str(leading) + "".join(str(a).zfill(width) for a, width in zip(lower, cut[1:]))
    return parse_answer(text)


def fits(piece_answers: list[Answer], cut: Cut) -> bool:
    """Their rule's test: every lower piece sum is readable and fits in its width."""
    return all(not isinstance(a, Unreadable) and a < 10**w for a, w in zip(piece_answers[1:], cut[1:]))


@dataclass(frozen=True)
class Built:
    """One cut's built answer for one problem, with the piece answers behind it."""

    cut: Cut
    pieces: tuple[Answer, ...]
    answer: Answer

    @property
    def fits(self) -> bool:
        return fits(list(self.pieces), self.cut)


def build_all(problems: list[Addition], cuts_per_problem: list[tuple[Cut, ...]], solver: Solver) -> list[list[Built]]:
    """Build every listed cut of every problem with ONE call to the solver.

    One call means one pass of batched generation over all pieces, instead of
    a model call per problem: thousands of times faster on a GPU.
    """
    jobs = [(i, cut, piece_problems(p, cut)) for i, (p, cuts) in enumerate(zip(problems, cuts_per_problem))
            for cut in cuts]
    flat = [piece for _, _, pieces in jobs for piece in pieces]
    answers = iter(solver(flat))
    built: list[list[Built]] = [[] for _ in problems]
    for i, cut, pieces in jobs:
        piece_answers = tuple(next(answers) for _ in pieces)
        built[i].append(Built(cut, piece_answers, combine(list(piece_answers), cut)))
    return built


# ----------------------------------------------------------------------------
# Teachers
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """What a teacher did with one hard problem.

    ``kept`` is the training answer, or None if the problem was set aside.
    ``candidates`` are what the teacher looked at: built answers per cut, or
    the five sampled direct answers for sample-and-vote.
    """

    problem: Addition
    kept: int | None
    candidates: tuple[Answer, ...]


def readable(answer: Answer) -> int | None:
    """An answer that can be trained on, or None (unreadables are set aside)."""
    return None if isinstance(answer, Unreadable) else answer


def majority(answers: list[Answer], needed: int) -> int | None:
    """The answer given at least ``needed`` times, if any (unreadables never count)."""
    counts = Counter(a for a in answers if not isinstance(a, Unreadable))
    if not counts:
        return None
    answer, n = counts.most_common(1)[0]
    return answer if n >= needed else None


def cuts_of(problem: Addition) -> tuple[Cut, ...]:
    return ADDITION_CUTS[problem.n_digits]


def their_rule(problems: list[Addition], solver: Solver) -> list[Decision]:
    built = build_all(problems, [cuts_of(p)[:1] for p in problems], solver)
    return [Decision(p, readable(b.answer) if b.fits else None, (b.answer,)) for p, (b,) in zip(problems, built)]


def two_cuts_agree(problems: list[Addition], solver: Solver) -> list[Decision]:
    built = build_all(problems, [cuts_of(p) for p in problems], solver)
    decisions = []
    for p, bs in zip(problems, built):
        answers = [b.answer for b in bs]
        decisions.append(Decision(p, majority(answers, needed=2), tuple(answers)))
    return decisions


def no_selection(problems: list[Addition], solver: Solver) -> list[Decision]:
    built = build_all(problems, [cuts_of(p)[:1] for p in problems], solver)
    return [Decision(p, readable(b.answer), (b.answer,)) for p, (b,) in zip(problems, built)]


def sample_and_vote(problems: list[Addition], solver: Solver) -> list[Decision]:
    # Ask each problem five times in one call: [p1..pn, p1..pn, ...].
    samples = solver(problems * VOTE_SAMPLES, temperature=VOTE_TEMPERATURE)
    n = len(problems)
    decisions = []
    for i, p in enumerate(problems):
        answers = [samples[i + k * n] for k in range(VOTE_SAMPLES)]
        decisions.append(Decision(p, majority(answers, VOTE_NEEDED), tuple(answers)))
    return decisions


Teacher = Callable[[list[Addition], Solver], list[Decision]]

TEACHERS: dict[str, Teacher] = {
    "their_rule": their_rule,
    "two_cuts_agree": two_cuts_agree,
    "no_selection": no_selection,
    "sample_and_vote": sample_and_vote,
}
