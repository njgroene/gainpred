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

Verdict (P9b, as decided at decision point 1), in this order
============================================================
1. *Does harm* if check 0 fails (easy accuracy under 98% at some length,
   measured before the round).
2. *Does nothing* if the teacher kept under 5% of the check sample (it has
   almost nothing to teach), or check 1 is below ``LEVEL_1`` (0.50).
3. *Does harm* if check 2b is below ``level_2b``.
4. *Pays off* otherwise.

Check 1 comes before 2b because "nothing new to teach" outranks "what it
teaches is unconfirmed": sample and vote, and no selection at lengths it
has already taught wrong, have low 2b yet change nothing. The 2b level is
still *to decide*; until then verdicts are scored only for a level given on
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
  *did nothing* if up less than 10 points (decision point 1: a gain under 10
  points is 'did nothing' regardless of the gap).
* A new length whose accuracy before the round is above 50% is scored like a
  length already trained on (decision point 1).
* At 13-15 digits accuracy comes from the five-cut test set (``study5``)
  where a run has it; the original set's 13-15 groups are secondary.

Case "new length started midway" (decision point 1, replacing "run 0 at 8
digits against 9"): every new-length cell whose accuracy before the round
is between 20% and 80% is tagged ``mid_start``, whatever seed or round.
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
LEVEL_1 = 0.50  # check-1 level, decided at decision point 1
MIN_KEPT_SHARE = 0.05  # under this share of the check sample kept: 'does nothing'
FIVE_CUT_DIGITS = range(13, 16)


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
    kept_share: float | None  # share of the check sample the teacher kept
    test_set: str  # 'study5' (five-cut set, 13-15 digits) or 'study'
    before: float | None
    after: float | None
    easy_change: float | None
    best_after: float | None = None
    mid_start: bool = False  # new length starting between 20% and 80%
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
    study5 = {e["round"]: e["accuracy"] for e in run.of("eval") if e["set"] == "study5"}
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
            five = digits in FIVE_CUT_DIGITS and rnd in study5 and rnd - 1 in study5
            src = study5 if five else study
            before = length_accuracy(src[rnd - 1], digits)
            new = digits > previous_max
            out.append(Cell(
                run=run.name, teacher=teacher_key(cfg["teacher"]), seed=cfg["seed"], round=rnd, digits=digits,
                new_length=new, check_0=check_0, check_1=c.get("check_1"),
                check_2b=c.get("check_2b"), check_2_kept=c.get("check_2_kept"), teachable=c.get("teachable"),
                kept_share=(c["kept"] / c["problems"]) if c.get("problems") else None,
                test_set="study5" if five else "study",
                mid_start=new and before is not None and 0.20 <= before <= 0.80,
                before=before, after=length_accuracy(src[rnd], digits),
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
        if not c.new_length or c.before > 0.50:  # learned, or new but started above 50%
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
        elif change < 0.10:
            c.outcome = "did nothing"
        elif change >= 0.20 and gap <= 0.10:
            c.outcome = "paid off"
        elif gap >= 0.30:
            c.outcome = "taught wrong answers"
        else:
            c.outcome = "unclassified"


def give_verdicts(cells: list[Cell], level_2b: float, level_1: float = LEVEL_1) -> None:
    for c in cells:
        if c.check_0 is False:
            c.verdict = "does harm"
        elif (c.kept_share is not None and c.kept_share < MIN_KEPT_SHARE) or c.check_1 is None \
                or c.check_1 < level_1:
            c.verdict = "does nothing"
        elif c.check_2b is not None and c.check_2b < level_2b:
            c.verdict = "does harm"
        else:
            c.verdict = "pays off"


MATCH = {"does harm": {"did harm", "taught wrong answers"}, "does nothing": {"did nothing"}, "pays off": {"paid off"}}


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


CONFIRMING_SEEDS = (2, 3, 4)


def support_lines(scored: list[Cell]) -> list[str]:
    """P9b's support test, on the confirming runs only, plus a per-round tally.

    The protocol's support rule: in the confirming runs (starting points 2-4) the
    verdict matches the outcome in at least 9 of 10 cells, with at least five cells
    in each of the did-nothing and did-harm groups ("did harm" here includes
    "taught wrong answers", which the protocol counts with did harm). The learning
    runs (starting points 0-1) are where the 2b level was set, so they cannot test it.

    The per-round table counts, for each run and round, how many of its classified
    cells match. It is a tally, not the protocol's "scored per round over all lengths
    together", whose round-level outcome is not yet defined (decision point 2).
    """
    conf = [c for c in scored if c.seed in CONFIRMING_SEEDS]
    hits = [c for c in conf if c.outcome in MATCH[c.verdict]]
    nothing = sum(c.outcome == "did nothing" for c in conf)
    harm = sum(c.outcome in ("did harm", "taught wrong answers") for c in conf)
    share = len(hits) / max(len(conf), 1)
    met = share >= 0.9 and nothing >= 5 and harm >= 5
    lines = ["**Support test (confirming runs, starting points 2-4):** "
             f"{len(hits)} of {len(conf)} cells match ({100 * share:.1f}%); did nothing {nothing} cells, "
             f"did harm or taught wrong answers {harm} cells -> {'met' if met else 'NOT met'} "
             "(needs 90% and five cells in each).", ""]
    misses = [c for c in conf if c.outcome not in MATCH[c.verdict]]
    if misses:
        lines += ["Misses in the confirming runs: " + "; ".join(
            f"{c.run} r{c.round} {c.digits} digits ({c.outcome}, verdict {c.verdict}"
            f"{', check 0 FAIL' if c.check_0 is False else ''})" for c in misses) + ".", ""]
    rounds: dict[tuple[str, int], list[Cell]] = {}
    for c in scored:
        rounds.setdefault((c.run, c.round), []).append(c)
    lines += ["Per run and round (classified cells matching / classified cells):", "",
              "| run | round | match |", "| --- | --- | --- |"]
    for (run, rnd), cs in sorted(rounds.items()):
        lines.append(f"| {run} | {rnd} | {sum(c.outcome in MATCH[c.verdict] for c in cs)}/{len(cs)} |")
    return lines + [""]


def to_markdown(cells: list[Cell], levels: tuple[float, float] | None) -> str:
    lines = ["# Experiment 3 (P9b): checks before training against outcomes", "",
             "Generated by `uv run python -m gainpred.p9b`; do not edit by hand. Accuracy = mean of the two carry",
             "groups at that length (%). Checks are computed before the round's training (P8).", ""]
    if levels:
        scored = [c for c in cells if c.outcome not in ("unclassified", "missing")]
        hits = sum(c.outcome in MATCH[c.verdict] for c in scored)
        lines += [f"**Verdicts at levels 2b < {levels[0]}, check 1 < {levels[1]}:** "
                  f"{hits} of {len(scored)} classified cells match ({100 * hits / max(len(scored), 1):.0f}%).", ""]
        groups = {"paid off": ["paid off"], "did nothing": ["did nothing"],
                  "did harm or taught wrong answers": ["did harm", "taught wrong answers"]}
        lines += ["| Outcome group | Cells | Verdict matches |", "| --- | --- | --- |"]
        for name, members in groups.items():
            g = [c for c in scored if c.outcome in members]
            m = sum(c.outcome in MATCH[c.verdict] for c in g)
            lines.append(f"| {name} | {len(g)} | {m} ({100 * m / max(len(g), 1):.0f}%) |")
        mid = [c for c in scored if c.mid_start]
        lines += ["", f"Case 'new length started midway' (20-80% before): {len(mid)} cells, "
                  f"{sum(c.outcome in MATCH[c.verdict] for c in mid)} matched.", ""]
        lines += support_lines(scored)
    counts: dict[str, int] = {}
    for c in cells:
        counts[c.outcome] = counts.get(c.outcome, 0) + 1
    lines += ["Outcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), ""]
    header = ["run", "round", "digits", "new", "test set", "check 0", "kept", "check 1", "2b", "check 2", "teachable",
              "before", "after", "best", "outcome"] + (["verdict"] if levels else [])
    lines += ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for c in cells:
        row = [c.run, str(c.round), str(c.digits), ("mid" if c.mid_start else "yes") if c.new_length else "",
               c.test_set, {True: "pass", False: "FAIL", None: "–"}[c.check_0], pct(c.kept_share),
               pct(c.check_1), pct(c.check_2b),
               pct(c.check_2_kept), pct(c.teachable), pct(c.before), pct(c.after), pct(c.best_after), c.outcome]
        if levels:
            row.append(c.verdict)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level-2b", type=float, help="the 2b level (still to decide)")
    parser.add_argument("--level-1", type=float, default=LEVEL_1)
    args = parser.parse_args()
    runs = [run for run in load_rounds() if included(run)]
    names = {run.name for run in runs}
    # Rounds 3-4 of a three-cut run are superseded by its five-cut rerun (decision point 1),
    # which continues from the same round-2 checkpoint: keep rounds 1-2 of the original.
    cells = [c for run in runs for c in cells_of(run)
             if not (c.round >= 3 and run.name.replace("-authors", "-5cut") in names - {run.name})]
    score_outcomes(cells)
    levels = (args.level_2b, args.level_1) if args.level_2b is not None else None
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
