"""Experiment 3 (P9b): the verdict given before training against what training then did.

Usage::

    uv run python -m gainpred.p9b                       # cells and summary -> results/p9b.md, results/p9b_cells.csv
    uv run python -m gainpred.p9b --level-2b 0.5 --level-1 0.1   # also score verdicts at these levels

The idea
========
Before each round's training, the label-free checks (P8) are computed on fresh
problems at every length. From them a **verdict** is given in advance: the
round will *pay off*, *do nothing* or *do harm* at that length. After
training, the test accuracy gives the **outcome**. Experiment 3 asks how often
the verdict matches the outcome. A "cell" is one run, round and length.

Verdict (P9b), decided in this order
====================================
1. *Does harm* if check 0 fails (easy accuracy under 98% at some length,
   measured before the round) or check 2b is below ``level_2b``.
2. *Does nothing* if check 1 is below ``level_1``.
3. *Pays off* otherwise.

The two levels are *to decide* on the learning runs, then frozen (P9b). Until
then this module reports the raw check values next to each outcome, so the
levels can be chosen from data, and scores verdicts only for levels given on
the command line.

Outcome (P9b, as amended 13 November 2026)
===========================================
Accuracy at a length is the mean of its two carry groups (equal-sized, so it
is the accuracy over the length). "Before" is the evaluation after the
previous round (round 0 = the starting model); "after" is this round's.

* At a length the model **had already been trained on**: *did nothing* if it
  moved by at most 3 points; *did harm* if it fell by more than 3 points, or
  easy accuracy fell by more than 3 points.
* At a **new** length (accuracy starts near 0% and cannot fall), compared with
  the best teacher from the same starting point in the same round: *paid off*
  if up at least 20 points and within 10 points of the best; *taught wrong
  answers* (counted with did harm) if up but 30 or more points below the best;
  *did nothing* if up less than 3 points.
* Anything else (e.g. up 3-20 points, or 10-30 points below the best, or up
  more than 3 at an already-trained length) is *unclassified* and reported.

Which runs count
================
Self-improvement runs of the adopted setup: names ending in ``-authors`` (and
run f's, which start from one of them). Earlier runs used the superseded
starting models (runbook A4) and are left out.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from gainpred.report import RESULTS_DIR, RoundsRun, load_rounds
from gainpred.teachers import teacher_key

MAX_DIGITS_BY_ROUND = (9, 11, 13, 15)
EASY_PASS = 0.98  # check 0 pass mark (P8)


@dataclass
class Cell:
    run: str
    teacher: str
    seed: int
    round: int
    digits: int
    new_length: bool
    check_0: bool | None  # passed, before this round
    check_1: float | None
    check_2b: float | None
    check_2_kept: float | None
    teachable: float | None
    before: float | None
    after: float | None
    easy_change: float | None
    best_after: float | None = None
    outcome: str = ""
    verdict: str = ""


def length_accuracy(acc: dict[str, float], digits: int) -> float | None:
    a, b = acc.get(f"{digits}/no-carry"), acc.get(f"{digits}/carry")
    return None if a is None or b is None else (a + b) / 2


def included(run: RoundsRun) -> bool:
    # Runs from an authors'-setup model: the starting models themselves are named
    # start-seed<k>-authors[-s<steps>] (the weak ones carry a step count).
    return run.name.endswith("-authors") or "-authors" in str(run.config.get("start_run", ""))


def cells_of(run: RoundsRun) -> list[Cell]:
    study = {e["round"]: e["accuracy"] for e in run.of("eval") if e["set"] == "study"}
    easy = {e["round"]: e["accuracy"] for e in run.of("eval") if e["set"] == "easy"}
    checks = {(e["round"], e["digits"]): e for e in run.of("checks")}
    cfg = run.config["rounds"]
    out = []
    for rnd in sorted(r for r in study if r - 1 in study):
        previous_max = MAX_DIGITS_BY_ROUND[rnd - 2] if rnd >= 2 else 7
        easy_before = easy.get(rnd - 1, {})
        check_0 = (all(v >= EASY_PASS for k, v in easy_before.items() if k != "all") if easy_before else None)
        for digits in range(8, MAX_DIGITS_BY_ROUND[rnd - 1] + 1):
            c = checks.get((rnd, digits), {})
            out.append(Cell(
                run=run.name, teacher=teacher_key(cfg["teacher"]), seed=cfg["seed"], round=rnd, digits=digits,
                new_length=digits > previous_max, check_0=check_0, check_1=c.get("check_1"),
                check_2b=c.get("check_2b"), check_2_kept=c.get("check_2_kept"), teachable=c.get("teachable"),
                before=length_accuracy(study[rnd - 1], digits), after=length_accuracy(study[rnd], digits),
                easy_change=(easy[rnd]["all"] - easy_before["all"]) if rnd in easy and easy_before else None))
    return out


def score_outcomes(cells: list[Cell]) -> None:
    """Fill in best_after (per seed, round, length) and the outcome of every cell."""
    best: dict[tuple[int, int, int], float] = {}
    for c in cells:
        if c.new_length and c.after is not None:
            key = (c.seed, c.round, c.digits)
            best[key] = max(best.get(key, 0.0), c.after)
    for c in cells:
        if c.before is None or c.after is None:
            c.outcome = "missing"
            continue
        change = c.after - c.before
        easy_harm = c.easy_change is not None and c.easy_change < -0.03
        if not c.new_length:
            if change < -0.03 or easy_harm:
                c.outcome = "did harm"
            elif abs(change) <= 0.03:
                c.outcome = "did nothing"
            else:
                c.outcome = "unclassified"
            continue
        c.best_after = best.get((c.seed, c.round, c.digits))
        gap = (c.best_after or 0.0) - c.after
        if easy_harm:
            c.outcome = "did harm"
        elif change < 0.03:
            c.outcome = "did nothing"
        elif change >= 0.20 and gap <= 0.10:
            c.outcome = "paid off"
        elif gap >= 0.30:
            c.outcome = "taught wrong answers"
        else:
            c.outcome = "unclassified"


def give_verdicts(cells: list[Cell], level_2b: float, level_1: float) -> None:
    for c in cells:
        if c.check_0 is False or (c.check_2b is not None and c.check_2b < level_2b):
            c.verdict = "does harm"
        elif c.check_1 is not None and c.check_1 < level_1:
            c.verdict = "does nothing"
        else:
            c.verdict = "pays off"


MATCH = {"does harm": {"did harm", "taught wrong answers"}, "does nothing": {"did nothing"}, "pays off": {"paid off"}}


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def to_markdown(cells: list[Cell], levels: tuple[float, float] | None) -> str:
    lines = ["# Experiment 3 (P9b): checks before training against outcomes", "",
             "Generated by `uv run python -m gainpred.p9b`; do not edit by hand. Accuracy = mean of the two carry",
             "groups at that length (%). Checks are computed before the round's training (P8).", ""]
    if levels:
        scored = [c for c in cells if c.outcome not in ("unclassified", "missing")]
        hits = sum(c.outcome in MATCH[c.verdict] for c in scored)
        lines += [f"**Verdicts at levels 2b < {levels[0]}, check 1 < {levels[1]}:** "
                  f"{hits} of {len(scored)} classified cells match ({100 * hits / max(len(scored), 1):.0f}%).", ""]
    counts: dict[str, int] = {}
    for c in cells:
        counts[c.outcome] = counts.get(c.outcome, 0) + 1
    lines += ["Outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), ""]
    header = ["run", "round", "digits", "new", "check 0", "check 1", "2b", "check 2", "teachable",
              "before", "after", "best", "outcome"] + (["verdict"] if levels else [])
    lines += ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for c in cells:
        row = [c.run, str(c.round), str(c.digits), "yes" if c.new_length else "",
               {True: "pass", False: "FAIL", None: "–"}[c.check_0], pct(c.check_1), pct(c.check_2b),
               pct(c.check_2_kept), pct(c.teachable), pct(c.before), pct(c.after), pct(c.best_after), c.outcome]
        if levels:
            row.append(c.verdict)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level-2b", type=float)
    parser.add_argument("--level-1", type=float)
    args = parser.parse_args()
    cells = [c for run in load_rounds() if included(run) for c in cells_of(run)]
    score_outcomes(cells)
    levels = (args.level_2b, args.level_1) if args.level_2b is not None and args.level_1 is not None else None
    if levels:
        give_verdicts(cells, *levels)
    (RESULTS_DIR / "p9b.md").write_text(to_markdown(cells, levels))
    with (RESULTS_DIR / "p9b_cells.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(cells[0]).keys()) if cells else ["run"])
        writer.writeheader()
        for c in cells:
            writer.writerow(asdict(c))
    print(f"{len(cells)} cells -> results/p9b.md, results/p9b_cells.csv")


if __name__ == "__main__":
    main()
