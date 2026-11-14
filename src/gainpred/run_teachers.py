"""Teachers for longest run (experiment 2): building answers to hard strings from piece answers (P4, P5).

The idea in one example
=======================
The model answers ``runlen`` well on strings of 6-10 symbols and badly on
longer ones. A hard string (12-20 symbols) is two easy pieces joined
(``gainpred.run_data``). Ask the model about each piece, then combine::

    hard string   "4930 77|7774 76"      join after symbol 6
    left piece    "493077"   runlen -> 7|2    rightrun -> 7|2
    right piece   "777476"   runlen -> 7|3    leftrun  -> 7|3
    true answer   7|5      (the run 77|777 crosses the join)

Shin et al.'s rule, **pick the longer**, keeps the piece answer with the
longer run (the left one on ties, because every run of the left piece comes
first): here 7|3. Wrong: the two edge runs merge into one run of 5 that
neither piece can see. Pick the longer can only ever hand back one of the
piece answers, so it cannot produce a run longer than both (Shin et al.,
Section 3.4, "limited supervision coverage"; our check 3 measures this).

The teachers (P5)
=================
``pick_the_longer``
    Keep the left piece's longest run if it is at least as long as the right
    piece's, otherwise the right piece's. Always keeps an answer.
``edges_differ`` -- the edges-differ rule (Shin et al.'s filter, Table 1)
    The pick-the-longer answer, kept only if the last symbol of the left piece
    differs from the first symbol of the right piece. Then no run can cross
    the join, and pick the longer is right whenever the piece answers are.
    The test reads the *string*, not the model: like Shin et al.'s
    G(x, y) = 1{x_|x| != y_1}, it needs no answer at all.
``join_at_the_cut``
    Ask each piece two questions: its longest run, and its edge run at the
    join (``rightrun`` of the left piece, ``leftrun`` of the right). The
    candidates are, left to right in the string: the left piece's longest
    run; if the two edge symbols are equal, a run of that symbol whose
    length is the two edge runs added (the run across the join); the right
    piece's longest run. Keep the longest, leftmost on ties. With right piece
    answers this is always right (proved in the tests by brute force).
``no_selection``
    By analogy with addition ("keep the cut-1 answer always"): the answer of
    the plain combining rule, pick the longer, always kept. That *is*
    ``pick_the_longer`` -- P5's "pick the longer" never sets anything aside --
    so the two names give identical decisions. Kept as an alias so code that
    loops over "no selection" works for both tasks; see the report's open
    questions.

As for addition, an unreadable piece answer that the rule needs makes the
problem set aside: an unreadable built answer cannot be trained on. So even
pick the longer and no selection can set a problem aside, but only for this
reason.

Cuts (P4)
=========
The training answer is built from one cut, at the join. Cuts three positions
either side (join - 3 and join + 3) are built only for the check-2
diagnostic (the agreement rule over the three cuts) and check 2b. Note that
their pieces are mostly *not* 6-10 symbols long: with a join at 6, the cut
at 3 gives pieces of 3 and 9 or more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from gainpred.formats import (
    Run,
    RunAnswer,
    RunQuestion,
    Unreadable,
    max_run_answer_chars,
    parse_run_answer,
    solve_run,
)
from gainpred.run_data import HardRun
from gainpred.teachers import majority

DIAGNOSTIC_OFFSET = 3
"""The original reading of P4 (join +- 3), replaced at decision point 1; kept for comparison."""

EDGE_SYMBOLS_FROM = "string"
"""Where join at the cut reads "the left piece's right-edge symbol" and "the
right piece's left-edge symbol" (P5): ``"string"`` (the symbols as written,
the literal reading) or ``"answers"`` (the symbols of the model's own edge-run
answers). Identical with a perfect solver."""


# ----------------------------------------------------------------------------
# Solvers: something that answers longest-run questions
# ----------------------------------------------------------------------------


class RunSolver(Protocol):
    def __call__(self, questions: list[RunQuestion], temperature: float = 0.0) -> list[RunAnswer]: ...


class PerfectRunSolver:
    """Always right, at any length and any temperature."""

    def __call__(self, questions: list[RunQuestion], temperature: float = 0.0) -> list[RunAnswer]:
        return [solve_run(q) for q in questions]


class RunModelSolver:
    """Ask a trained model longest-run questions (the twin of ``teachers.ModelSolver``).

    Questions are answered in batches sorted by length and returned in the
    original order; greedy at temperature 0, seeded sampling above 0.
    """

    def __init__(self, model, device: torch.device, batch_size: int = 1000, seed: int = 0):
        self.model = model.eval()
        self.device = device
        self.batch_size = batch_size
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.calls = 0
        self.generated_tokens = 0

    def __call__(self, questions: list[RunQuestion], temperature: float = 0.0) -> list[RunAnswer]:
        from gainpred.tokenizer import EOS_ID, make_prompts, read_generated
        from gainpred.train import autocast  # avoid an import cycle

        order = sorted(range(len(questions)), key=lambda i: questions[i].length)
        answers: list[RunAnswer | None] = [None] * len(questions)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            chunk = [questions[i] for i in idx]
            limit = max(max_run_answer_chars(q) for q in chunk) + 1
            with autocast(self.device):
                outputs = self.model.generate(make_prompts(chunk), eos_id=EOS_ID, max_new_tokens=limit,
                                              temperature=temperature, generator=self.generator)
            for i, ids in zip(idx, outputs):
                answers[i] = read_generated(ids, parse=parse_run_answer)  # type: ignore[assignment]
                self.generated_tokens += len(ids)
        self.calls += len(questions)
        return answers  # type: ignore[return-value]


# ----------------------------------------------------------------------------
# Piece answers at a cut
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class PieceAnswers:
    """What the solver said about the two pieces of one cut.

    ``position`` is where the cut splits the string (the join, or join +- 3).
    The edge answers are None when the rule did not ask for them.
    """

    position: int
    left_longest: RunAnswer
    right_longest: RunAnswer
    left_edge: RunAnswer | None = None  # rightrun of the left piece
    right_edge: RunAnswer | None = None  # leftrun of the right piece

    def all(self) -> tuple[RunAnswer, ...]:
        """Every answer actually asked, in the order of the string."""
        return tuple(a for a in (self.left_longest, self.left_edge, self.right_edge, self.right_longest)
                     if a is not None)


def ask_pieces(
    problems: list[HardRun], positions: list[tuple[int, ...]], solver: RunSolver, edges: bool
) -> list[list[PieceAnswers]]:
    """Ask about the pieces of every listed cut of every problem, in ONE solver call.

    ``edges`` also asks each piece its edge run at the cut (join at the cut
    needs them; pick the longer does not).
    """
    questions: list[RunQuestion] = []
    for p, cuts in zip(problems, positions, strict=True):
        for pos in cuts:
            left, right = p.string[:pos], p.string[pos:]
            questions += [RunQuestion("runlen", left), RunQuestion("runlen", right)]
            if edges:
                questions += [RunQuestion("rightrun", left), RunQuestion("leftrun", right)]
    answers = iter(solver(questions))
    out: list[list[PieceAnswers]] = []
    for cuts in positions:
        row = []
        for pos in cuts:
            left_longest, right_longest = next(answers), next(answers)
            left_edge, right_edge = (next(answers), next(answers)) if edges else (None, None)
            row.append(PieceAnswers(pos, left_longest, right_longest, left_edge, right_edge))
        out.append(row)
    return out


# ----------------------------------------------------------------------------
# Combining rules
# ----------------------------------------------------------------------------


def _unreadable(why: str, answers: tuple[RunAnswer | None, ...]) -> Unreadable:
    raw = "|".join(a.raw if isinstance(a, Unreadable) else "-" if a is None else f"{a.symbol}:{a.length}"
                   for a in answers)
    return Unreadable(f"{why}: {raw}")


def pick_the_longer_rule(pieces: PieceAnswers) -> tuple[RunAnswer, tuple[RunAnswer, ...]]:
    """P5 pick the longer: (built answer, candidates looked at).

    The left piece wins ties: every run of the left piece starts before every
    run of the right, so a tie is won by the left one (Shin et al., A.1.3).
    """
    left, right = pieces.left_longest, pieces.right_longest
    candidates = (left, right)
    if isinstance(left, Unreadable) or isinstance(right, Unreadable):
        return _unreadable("built from unreadable piece", candidates), candidates
    return (left if left.length >= right.length else right), candidates


def join_at_the_cut_rule(
    problem: HardRun, pieces: PieceAnswers, edge_symbols_from: str = EDGE_SYMBOLS_FROM
) -> tuple[RunAnswer, tuple[RunAnswer, ...]]:
    """P5 join at the cut: (built answer, candidates looked at).

    Candidates in string order: left longest; the run across the cut (only if
    the edge symbols are equal; its length is the two edge runs added);
    right longest. ``max`` keeps the first of the longest, which is the
    leftmost: the run across the cut starts inside the left piece, after or
    at the start of the left piece's longest run, and before every run of
    the right piece.
    """
    left_edge, right_edge = pieces.left_edge, pieces.right_edge
    asked = (pieces.left_longest, left_edge, right_edge, pieces.right_longest)
    if any(a is None or isinstance(a, Unreadable) for a in asked):
        return _unreadable("built from unreadable piece", asked), tuple(a for a in asked if a is not None)
    assert isinstance(left_edge, Run) and isinstance(right_edge, Run)
    if edge_symbols_from == "string":
        left_symbol, right_symbol = problem.string[pieces.position - 1], problem.string[pieces.position]
    elif edge_symbols_from == "answers":
        left_symbol, right_symbol = left_edge.symbol, right_edge.symbol
    else:
        raise ValueError(f"edge_symbols_from must be 'string' or 'answers', not {edge_symbols_from!r}")
    candidates: list[Run] = [pieces.left_longest]  # type: ignore[list-item]
    if left_symbol == right_symbol:
        candidates.append(Run(left_symbol, left_edge.length + right_edge.length))
    candidates.append(pieces.right_longest)  # type: ignore[arg-type]
    return max(candidates, key=lambda r: r.length), tuple(candidates)


COMBINE_RULES = ("pick_the_longer", "join_at_the_cut")


def build(problem: HardRun, pieces: PieceAnswers, rule: str) -> RunAnswer:
    """The built answer of one cut under one of ``COMBINE_RULES``."""
    if rule == "pick_the_longer":
        return pick_the_longer_rule(pieces)[0]
    if rule == "join_at_the_cut":
        return join_at_the_cut_rule(problem, pieces)[0]
    raise ValueError(f"unknown combining rule {rule!r}")


# ----------------------------------------------------------------------------
# Teachers
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunDecision:
    """What a teacher did with one hard longest-run problem (twin of ``teachers.Decision``).

    ``kept`` is the training answer, or None if set aside. ``candidates`` are
    the answers the rule chose between (P5's wording). ``source`` is the
    index of the cut the kept answer came from: 0, the join (P4), or None if
    nothing was kept. ``pieces`` are every piece answer the teacher asked
    for, for check 3 ("does the teacher hand back a piece answer?").
    """

    problem: HardRun
    kept: Run | None
    candidates: tuple[RunAnswer, ...]
    source: int | None = None
    pieces: tuple[RunAnswer, ...] = ()


def edges_differ_in_string(problem: HardRun, position: int | None = None) -> bool:
    """True iff the symbols either side of the cut differ (the edges-differ rule's test)."""
    pos = problem.join if position is None else position
    return problem.string[pos - 1] != problem.string[pos]


def _join_only(problems: list[HardRun]) -> list[tuple[int, ...]]:
    return [(p.join,) for p in problems]


def pick_the_longer(problems: list[HardRun], solver: RunSolver) -> list[RunDecision]:
    """P5 pick the longer, at the join. Sets aside only unreadable answers."""
    decisions = []
    for p, (pieces,) in zip(problems, ask_pieces(problems, _join_only(problems), solver, edges=False)):
        built, candidates = pick_the_longer_rule(pieces)
        kept = None if isinstance(built, Unreadable) else built
        decisions.append(RunDecision(p, kept, candidates, None if kept is None else 0, pieces.all()))
    return decisions


def edges_differ(problems: list[HardRun], solver: RunSolver) -> list[RunDecision]:
    """P5 the edges-differ rule: pick the longer, kept only if the edges at the join differ."""
    decisions = []
    for d in pick_the_longer(problems, solver):
        keep = d.kept is not None and edges_differ_in_string(d.problem)
        decisions.append(RunDecision(d.problem, d.kept if keep else None, d.candidates, 0 if keep else None,
                                     d.pieces))
    return decisions


def join_at_the_cut(problems: list[HardRun], solver: RunSolver) -> list[RunDecision]:
    """P5 join at the cut, at the join. Sets aside only unreadable answers."""
    decisions = []
    for p, (pieces,) in zip(problems, ask_pieces(problems, _join_only(problems), solver, edges=True)):
        built, candidates = join_at_the_cut_rule(p, pieces)
        kept = None if isinstance(built, Unreadable) else built
        decisions.append(RunDecision(p, kept, candidates, None if kept is None else 0, pieces.all()))
    return decisions


def no_selection(problems: list[HardRun], solver: RunSolver) -> list[RunDecision]:
    """No selection, by analogy with addition: pick the longer's answer, always kept.

    Identical to ``pick_the_longer`` (see the module docstring); an alias.
    """
    return pick_the_longer(problems, solver)


RunTeacher = Callable[[list[HardRun], RunSolver], list[RunDecision]]

RUN_TEACHERS: dict[str, RunTeacher] = {
    "edges_differ": edges_differ,
    "pick_the_longer": pick_the_longer,
    "join_at_the_cut": join_at_the_cut,
    "no_selection": no_selection,
}

RUN_TEACHER_NAMES = {
    "edges_differ": "the edges-differ rule",
    "pick_the_longer": "pick the longer",
    "join_at_the_cut": "join at the cut",
    "no_selection": "no selection",
}
"""How each teacher is written in prose and tables (P5)."""


# ----------------------------------------------------------------------------
# The check-2 diagnostic (P4): three cuts, join and join +- 3
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunBuilt:
    """One cut's built answer for one problem (twin of ``teachers.Built``).

    Has ``.answer`` like ``Built``, so ``checks.confirmed_share`` (check 2b)
    works on it unchanged.
    """

    position: int
    pieces: PieceAnswers
    answer: RunAnswer


def diagnostic_positions(problem: HardRun, offset: int | None = None) -> tuple[int, ...]:
    """P4 diagnostic cuts. Index 0 is the join, the cut that supplies training answers.

    Decided at decision point 1 (15 November 2026): the two diagnostic cuts are
    the nearest alternative splits that keep both pieces 6-10 symbols long, the
    lengths the model was trained on (ties: the left one first). The original
    reading, the join +- 3, mostly made pieces of 3-13 symbols; pass ``offset=3``
    to get it back. Short strings have fewer alternatives: length 12 has none,
    13 has one, so check 2 there rests on fewer cuts.
    """
    if offset is not None:
        return problem.join, problem.join - offset, problem.join + offset
    n = len(problem.string)
    splits = [p for p in range(max(6, n - 10), min(10, n - 6) + 1) if p != problem.join]
    nearest = sorted(splits, key=lambda p: (abs(p - problem.join), p))[:2]
    return (problem.join, *nearest)


def build_diagnostic(problems: list[HardRun], solver: RunSolver, rule: str = "join_at_the_cut",
                     offset: int | None = None) -> list[list[RunBuilt]]:
    """Built answers at the three cuts of P4, with ONE solver call.

    ``rule`` is how each cut's piece answers are combined. P4 does not say;
    the default is join at the cut, the combining rule that is right whenever
    its piece answers are, so disagreement between cuts points at wrong piece
    answers rather than at the rule (open question in the report).
    """
    positions = [diagnostic_positions(p, offset) for p in problems]
    asked = ask_pieces(problems, positions, solver, edges=(rule == "join_at_the_cut"))
    return [[RunBuilt(pc.position, pc, build(p, pc, rule)) for pc in row] for p, row in zip(problems, asked)]


def agreement_from_built(problems: list[HardRun], built: list[list[RunBuilt]]) -> list[RunDecision]:
    """The agreement rule (P5) over the three cuts: keep a string at least two cuts give."""
    decisions = []
    for p, bs in zip(problems, built):
        answers = [b.answer for b in bs]
        kept = majority(answers, needed=2)  # unreadables never count
        decisions.append(RunDecision(p, kept, tuple(answers), None if kept is None else answers.index(kept)))
    return decisions
