"""The checks of P8: label-free diagnostics of a teacher, computed each round.

None of these (except check 0, which lives in ``gainpred.train.evaluate``)
uses right answers: they are things we could compute in a real setting where
nobody knows the answers to hard problems. The study asks whether they predict
which teachers make the model improve.

Check 1 -- does the teacher say anything new?
    Of the problems a teacher kept, the share where the model's own greedy
    direct answer *differs* from the kept answer. If ~0%, the teacher only
    repeats what the model already says, and training on it cannot change
    the model. The predictor of experiment 3 (P9b): "does nothing" when low.
Check 2 -- does the agreement rule keep an answer?
    Run the agreement rule (P5) on the same problems and report the share
    kept and set aside. Agreement between independent routes to an answer is
    evidence the pieces were right.
Check 2b -- confirmed share
    Of the teacher's kept answers, the share identical to the answer built
    from at least one *other* cut than the one that produced it (for sample
    and vote, which builds nothing: from any cut). 100% for the agreement rule
    by construction. Low 2b means the teacher keeps answers that no second
    route supports -- "does harm" in P9b.
Teachable share
    Of *all* hard problems, the share with a kept answer that differs from the
    direct answer. A cost measure: with a fixed quota it says how many
    candidates must be drawn, not how much the student learns.
Check 3 -- does the teacher just hand back a piece answer?
    Relevant to longest run (where "pick the longer" returns one of the piece
    answers verbatim); added with that task.

All checks are computed each round on 2,000 fresh problems at *every* length
from 8 to the current maximum (P8), with the model as it is before the
round's training: they are a verdict given in advance.
"""

from __future__ import annotations

from dataclasses import dataclass

from gainpred.formats import Answer
from gainpred.teachers import Built, Decision, Solver, agreement_from, build_all, cuts_of


@dataclass(frozen=True)
class CheckReport:
    problems: int
    kept: int
    check_1: float | None  # share of kept answers that differ from the direct answer
    teachable: float  # share of all problems with a kept answer differing from the direct one
    check_2_kept: float  # agreement rule: share kept
    check_2_set_aside: float
    check_2b: float | None = None  # share of kept answers confirmed by another cut


def check_1_and_teachable(decisions: list[Decision], direct: list[Answer]) -> tuple[float | None, float]:
    kept = [(d, a) for d, a in zip(decisions, direct, strict=True) if d.kept is not None]
    differs = sum(d.kept != a for d, a in kept)
    return (differs / len(kept) if kept else None), differs / len(decisions)


def confirmed_share(decisions: list[Decision], built: list[list[Built]]) -> float | None:
    """Check 2b: share of kept answers equal to the built answer of some other cut.

    ``built`` holds every cut's built answer per problem (from ``build_all``).
    For a decision with ``source`` None (a direct answer), any cut counts.
    """
    kept = [(d, bs) for d, bs in zip(decisions, built, strict=True) if d.kept is not None]
    if not kept:
        return None
    confirmed = sum(any(b.answer == d.kept for j, b in enumerate(bs) if j != d.source) for d, bs in kept)
    return confirmed / len(kept)


def run_checks(decisions: list[Decision], solver: Solver) -> CheckReport:
    """All label-free checks for one teacher's decisions on one sample of hard problems."""
    problems = [d.problem for d in decisions]
    direct = solver(problems)  # greedy
    check_1, teachable = check_1_and_teachable(decisions, direct)
    built = build_all(problems, [cuts_of(p) for p in problems], solver)  # all three cuts, once
    agree = agreement_from(problems, built)
    kept_2 = sum(d.kept is not None for d in agree) / len(agree)
    return CheckReport(
        problems=len(decisions),
        kept=sum(d.kept is not None for d in decisions),
        check_1=check_1,
        teachable=teachable,
        check_2_kept=kept_2,
        check_2_set_aside=1 - kept_2,
        check_2b=confirmed_share(decisions, built),
    )
