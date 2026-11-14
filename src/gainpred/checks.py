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
    Over 10,000 sets of piece answers, the share of the teacher's outputs
    identical to one of the piece answers it was given. "Hands back" if 100%.
    Relevant to longest run: "pick the longer" (and the edges-differ rule,
    which filters it) can only return one of the two piece answers verbatim,
    so the model can never be taught a run longer than both pieces' runs
    (Shin et al., Section 3.4). Join at the cut can return a new, longer run.

All checks are computed each round on 2,000 fresh problems at *every* length
from 8 to the current maximum (P8), with the model as it is before the
round's training: they are a verdict given in advance.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from gainpred.formats import Answer
from gainpred.teachers import Built, Decision, Solver, agreement_from, build_all, cuts_of

CHECK_3_SETS = 10_000
"""P8: check 3 is computed over this many sets of piece answers per teacher."""


@dataclass(frozen=True)
class CheckReport:
    problems: int
    kept: int
    check_1: float | None  # share of kept answers that differ from the direct answer
    teachable: float  # share of all problems with a kept answer differing from the direct one
    check_2_kept: float  # agreement rule: share kept
    check_2_set_aside: float
    check_2b: float | None = None  # share of kept answers confirmed by another cut


@dataclass(frozen=True)
class RunCheckReport(CheckReport):
    """The longest-run report: the same checks plus check 3. A separate class so
    that addition's logged reports keep exactly their old fields."""

    check_3: float | None = None  # share of kept answers identical to a piece answer


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


# ----------------------------------------------------------------------------
# Check 3 and the longest-run checks
# ----------------------------------------------------------------------------


def check_3(outputs: Iterable[tuple[object | None, Sequence[object]]]) -> float | None:
    """Check 3 (P8): share of teacher outputs identical to one of their piece answers.

    ``outputs`` are (kept answer or None, piece answers) pairs, one per set of
    piece answers; problems set aside produce no output and are skipped.
    Identity is ``==``, so an Unreadable piece answer never matches. None if
    the teacher kept nothing.
    """
    kept = [(k, pieces) for k, pieces in outputs if k is not None]
    if not kept:
        return None
    return sum(any(k == piece for piece in pieces) for k, pieces in kept) / len(kept)


def hands_back(share: float | None) -> bool:
    """P8: a teacher "hands back" if every one of its outputs is a piece answer."""
    return share == 1.0


def run_checks_longest_run(decisions: list, solver, check_2_rule: str = "join_at_the_cut") -> RunCheckReport:
    """All label-free checks for one longest-run teacher on one sample of hard strings.

    Check 1 and the teachable share compare with the model's greedy direct
    ``runlen`` answer. Checks 2 and 2b use the three cuts of P4 (the join and
    three positions either side), combined with ``check_2_rule`` -- a
    diagnostic only; the training answer always comes from the join. Check 3
    uses the piece answers each decision records.
    """
    from gainpred.run_teachers import agreement_from_built, build_diagnostic  # longest run only

    problems = [d.problem for d in decisions]
    direct = solver([p.question for p in problems])
    check_1, teachable = check_1_and_teachable(decisions, direct)
    built = build_diagnostic(problems, solver, rule=check_2_rule)
    agree = agreement_from_built(problems, built)
    kept_2 = sum(d.kept is not None for d in agree) / len(agree)
    return RunCheckReport(
        problems=len(decisions),
        kept=sum(d.kept is not None for d in decisions),
        check_1=check_1,
        teachable=teachable,
        check_2_kept=kept_2,
        check_2_set_aside=1 - kept_2,
        check_2b=confirmed_share(decisions, built),
        check_3=check_3((d.kept, d.pieces) for d in decisions),
    )
