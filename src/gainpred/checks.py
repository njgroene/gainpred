"""The checks of P8: label-free diagnostics of a teacher, computed each round.

None of these (except check 0, which lives in ``gainpred.train.evaluate``)
uses right answers: they are things we could compute in a real setting where
nobody knows the answers to hard problems. The study asks whether they predict
which teachers make the model improve.

Check 1 -- does the teacher say anything new?
    Of the problems a teacher kept, the share where the model's own greedy
    direct answer *differs* from the kept answer. If ~0%, the teacher only
    repeats what the model already says, and training on it cannot change
    the model.
Check 2 -- do two cuts agree?
    Run the two-cuts-agree rule (P5) on the same problems and report the share
    kept and set aside. Agreement between independent routes to an answer is
    evidence the pieces were right.
Teachable share
    Of *all* hard problems, the share with a kept answer that differs from the
    direct answer: how much new material the teacher offers per problem drawn.
Check 3 -- does the teacher just hand back a piece answer?
    Relevant to longest run (where "pick the longer" returns one of the piece
    answers verbatim); added with that task.
"""

from __future__ import annotations

from dataclasses import dataclass

from gainpred.formats import Answer
from gainpred.teachers import Decision, Solver, two_cuts_agree


@dataclass(frozen=True)
class CheckReport:
    problems: int
    kept: int
    check_1: float | None  # share of kept answers that differ from the direct answer
    teachable: float  # share of all problems with a kept answer differing from the direct one
    check_2_kept: float  # two-cuts-agree: share kept
    check_2_set_aside: float


def check_1_and_teachable(decisions: list[Decision], direct: list[Answer]) -> tuple[float | None, float]:
    kept = [(d, a) for d, a in zip(decisions, direct, strict=True) if d.kept is not None]
    differs = sum(d.kept != a for d, a in kept)
    return (differs / len(kept) if kept else None), differs / len(decisions)


def run_checks(decisions: list[Decision], solver: Solver) -> CheckReport:
    """All label-free checks for one teacher's decisions on one sample of hard problems."""
    problems = [d.problem for d in decisions]
    direct = solver(problems)  # greedy
    check_1, teachable = check_1_and_teachable(decisions, direct)
    agree = two_cuts_agree(problems, solver)
    kept_2 = sum(d.kept is not None for d in agree) / len(agree)
    return CheckReport(
        problems=len(decisions),
        kept=sum(d.kept is not None for d in decisions),
        check_1=check_1,
        teachable=teachable,
        check_2_kept=kept_2,
        check_2_set_aside=1 - kept_2,
    )
