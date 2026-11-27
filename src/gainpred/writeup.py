"""Results material for the write-up (phase C): tables and figures, all generated.

Usage::

    uv run python -m gainpred.report          # first: the per-run logs in results/runs
    uv run python -m gainpred.p9b --level-2b 0.38
    uv run python -m gainpred.writeup         # then: results/writeup/*.md and figures/

Nothing in ``results/writeup/`` is typed by hand: every number is read from the
run logs under ``results/runs/``, from ``results/p9b_cells.csv`` or from
``results/withheld_split.json``. Rebuild after any of them changes.

What is in it (owner's list at decision point 2, 15 November 2026)
==================================================================
``exp1.md``
    Experiment 1 (addition, five teachers). Final accuracy per length, with the
    confirming runs (starting points 2-4) as the evidence and the learning runs
    (starting points 0-1) in a separate table, never pooled. The pass mark of
    experiment 1, computed formally: agreement rule minus no-carry rule after
    round 4, per length, mean over the three confirming seeds and worst single
    seed. Next to it, the carries-per-place table: how often the training answers
    each teacher kept carry out of each place.
``exp2.md``
    Experiment 2 (longest run). One round from the 30,000-update models: the
    three teachers, then the withheld version and its control arm side by side,
    then the withheld split by left part. The control arm is an observation.
``exp3.md``
    Experiment 3 (P9b): the support test, baselines, per outcome group and per
    case, sensitivity without check 0, unclassified cells and the round-level
    score (secondary, defined after the confirming runs). Built from
    ``results/p9b_cells.csv`` by the same functions as ``results/p9b.md``.
``runs.md``
    Every run the project has started, including failed, superseded and parked
    ones, with the commit it was launched from.
``figures/``
    PNG figures, one script (this module), no typed numbers.

Reading the accuracy numbers
============================
Addition accuracy at a length is the mean of its two test groups (problems
whose carry crosses cut 1, and problems without one), as in P9b; the carry group
alone is shown where the owner asked for it. At 13-15 digits the five-cut study
set (``study5``) is primary (decision point 1); the original set is shown only
where stated. For starting points 0-1, rounds 3-4 come from the five-cut reruns
(``-5cut``), which continue from the same round-2 checkpoint (decision point 1).
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import subprocess
from pathlib import Path

from gainpred.cuts import ADDITION_CUTS, HARD_DIGITS, cut_positions
from gainpred.report import RESULTS_DIR

OUT = RESULTS_DIR / "writeup"
PAPER = Path(__file__).resolve().parents[2] / "paper" / "generated"  # \\input by the paper; never edited by hand
RUNS = RESULTS_DIR / "runs"
CONFIRMING, LEARNING = (2, 3, 4), (0, 1)
ARMS = (("nocarry", "No-carry rule"), ("agreement", "Agreement rule"), ("noselect", "No selection"),
        ("vote", "Sample and vote"))
LENGTHS = list(HARD_DIGITS)  # 8-15
# Categorical slots 1-4 of the reference palette (dataviz skill), plus a neutral for "before".
COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
NEUTRAL = "#8a8986"


# ---------------------------------------------------------------- reading logs

def events(run: str) -> list[dict]:
    path = RUNS / run / "rounds.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def evals(run: str, set_name: str) -> dict[int, dict[str, float]]:
    """Round -> accuracy dict for one test set (round 0 = before round 1)."""
    return {e["round"]: e["accuracy"] for e in events(run) if e["event"] == "eval" and e["set"] == set_name}


def addition_run(arm: str, seed: int, rnd: int) -> str:
    """The run that holds round ``rnd`` of an addition arm (five-cut reruns for seeds 0-1, rounds 3-4)."""
    if seed in LEARNING and rnd >= 3:
        return f"si-{arm}-seed{seed}-5cut"
    return f"si-{arm}-seed{seed}-authors"


def runf_run(seed: int) -> str:
    """Run f of a starting point; seed 3's original failed at start and its rerun is the confirming run."""
    name = f"si-noselect-after-nocarry-seed{seed}-authors"
    return name + "-r2" if (RUNS / (name + "-r2")).exists() else name


def addition_accuracy(run: str, rnd: int, digits: int, group: str = "mean") -> float | None:
    """Accuracy (%) at a length after round ``rnd``; ``group`` = 'mean' (of the two groups) or 'carry'."""
    acc = evals(run, "study5" if digits >= 13 else "study").get(rnd)
    if acc is None and digits >= 13:  # a run without the five-cut set: fall back to the original
        acc = evals(run, "study").get(rnd)
    if acc is None:
        return None
    carry, plain = acc.get(f"{digits}/carry"), acc.get(f"{digits}/no-carry")
    if carry is None or plain is None:
        return None
    return 100 * {"carry": carry, "no-carry": plain}.get(group, (carry + plain) / 2)


def fmt(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "–"
    return f"{x:+.1f}" if signed else f"{x:.1f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + \
        ["| " + " | ".join(r) + " |" for r in rows]


# ---------------------------------------------------------------- experiment 1

SPLIT_ARMS = ("noselect", "runf")  # shown by carry group in the main table (owner, 15 Nov 2026)


def exp1_final_rows(seeds: tuple[int, ...], group: str, split: bool = False) -> list[list[str]]:
    """Rows of arm x seed. With ``split``, no selection and run f get one row per carry group:
    their mean of about 50% hides that one group is near 100% and the other near 0%."""
    def rows_for(label, run_of, rnd, arm):
        out = []
        groups = (("carry", "carry group"), ("no-carry", "no-carry group")) if split and arm in SPLIT_ARMS \
            else ((group, None),)
        for s in seeds:
            for g, glabel in groups:
                out.append([label + (f", {glabel}" if glabel else ""), str(s)]
                           + [fmt(addition_accuracy(run_of(s), rnd, d, g)) for d in LENGTHS])
        return out
    rows = []
    for arm, label in ARMS:
        rows += rows_for(label, lambda s, arm=arm: addition_run(arm, s, 4), 4, arm)
    rows += rows_for("Run f (no selection, round 3)", runf_run, 3, "runf")
    return rows


def pass_mark(seeds: tuple[int, ...], group: str, original_set: bool = False):
    """Agreement minus no-carry after round 4, per length: (mean gap, worst gap, worst seed) in points."""
    out = {}
    for d in LENGTHS:
        gaps = []
        for s in seeds:
            if original_set and d >= 13:
                def acc(arm):
                    a = evals(addition_run(arm, s, 4), "study")[4]
                    return 100 * (a[f"{d}/carry"] if group == "carry" else (a[f"{d}/carry"] + a[f"{d}/no-carry"]) / 2)
                gaps.append(acc("agreement") - acc("nocarry"))
            else:
                gaps.append(addition_accuracy(addition_run("agreement", s, 4), 4, d, group)
                            - addition_accuracy(addition_run("nocarry", s, 4), 4, d, group))
        worst = min(gaps)
        out[d] = (sum(gaps) / len(gaps), worst, seeds[gaps.index(worst)])
    return out


def new_round(digits: int) -> int:
    """The round in which a length is first trained on (P3: two new lengths per round)."""
    return (digits - 8) // 2 + 1


def carry_shares(arm: str, seed: int, d: int) -> dict[str, float] | None:
    """One seed's share (%) of kept training answers carrying out of each kind of place, in the
    round length ``d`` is first trained on; None if that build has no carry count (see below).

    Places are counted from the units (0). A cut boundary at ``p`` digits from the
    left of an n-digit number is crossed by a carry out of place ``n - 1 - p``.
    Kinds: the cut-1 boundary; the boundaries of the other cuts in use; every
    other place below the top digit (the top place's carry only lengthens the
    answer and is left out).
    """
    cuts = ADDITION_CUTS[d]
    cut1 = {d - 1 - p for p in cut_positions(cuts[0])}
    others = {d - 1 - p for c in cuts[1:] for p in cut_positions(c)} - cut1
    rest = set(range(d - 1)) - cut1 - others
    rnd = new_round(d)
    build = next((e for e in events(addition_run(arm, seed, rnd))
                  if e["event"] == "build" and e["round"] == rnd and e["digits"] == d), None)
    if not build or "carries_by_place" not in build or not build["kept"]:
        return None
    counts, kept = build["carries_by_place"], build["kept"]
    return {kind: 100 * sum(counts[i] for i in places) / len(places) / kept
            for kind, places in (("cut 1", cut1), ("other cuts", others), ("no cut", rest)) if places}


def carries_per_place(arm: str, seeds: tuple[int, ...]):
    """``carry_shares`` over seeds: {digits: {kind: [share per seed that has one]}}."""
    out = {}
    for d in LENGTHS:
        shares = {"cut 1": [], "other cuts": [], "no cut": []}
        for s in seeds:
            one = carry_shares(arm, s, d)
            for kind, v in (one or {}).items():
                shares[kind].append(v)
        out[d] = shares
    return out


def mean_range(xs: list[float]) -> str:
    if not xs:
        return "–"
    return f"{sum(xs) / len(xs):.1f} ({min(xs):.1f}–{max(xs):.1f})" if len(xs) > 1 else f"{xs[0]:.1f}"


def exp1() -> str:
    head = ["Arm", "Seed"] + [f"{d}" for d in LENGTHS]
    lines = ["# Experiment 1: addition, five teachers", "",
             "Generated by `uv run python -m gainpred.writeup`; do not edit by hand.", "",
             "Accuracy after round 4 (%, mean of the carry and no-carry groups; 13-15 digits on the five-cut "
             "set). Run f: after its one round (round 3, from the no-carry run's round 2). No selection and run f "
             "are shown by carry group: their mean of about 50% is one group near 100% and the other near 0%.", "",
             "## Confirming runs (starting points 2-4)", ""]
    lines += table(head, exp1_final_rows(CONFIRMING, "mean", split=True))
    lines += ["", "Carry group alone:", ""] + table(head, exp1_final_rows(CONFIRMING, "carry"))
    lines += ["", "## Learning runs (starting points 0-1), shown separately", "",
              "Rounds 3-4 from the five-cut reruns (`-5cut`).", ""]
    lines += table(head, exp1_final_rows(LEARNING, "mean", split=True))

    lines += ["", "## Pass mark: agreement rule against the no-carry rule", "",
              "Agreement rule minus no-carry rule after round 4, in points, confirming runs. "
              "Mark (owner, decision point 2): the mean over the three seeds within 3 points at every length, "
              "and no single run more than 6 points worse.", ""]
    rows, met = [], True
    pm_mean, pm_carry = pass_mark(CONFIRMING, "mean"), pass_mark(CONFIRMING, "carry")
    for d in LENGTHS:
        m, w, ws = pm_mean[d]
        cm, cw, cws = pm_carry[d]
        ok = m >= -3 and w >= -6
        met &= ok
        rows.append([str(d), fmt(m, True), f"{fmt(w, True)} (seed {ws})", fmt(cm, True), f"{fmt(cw, True)} (seed {cws})",
                     "yes" if ok else "**no**"])
    lines += table(["Digits", "Mean gap", "Worst run", "Mean gap, carry group", "Worst run, carry group",
                    "Mark met (both groups mean)"], rows)
    orig = pass_mark(CONFIRMING, "mean", original_set=True)
    lines += ["", f"**Pass mark {'met' if met else 'NOT met'}** at every length. Largest mean gap "
              f"{fmt(min(v[0] for v in pm_mean.values()), True)} points; worst single run "
              f"{fmt(min(v[1] for v in pm_mean.values()), True)} points. On the original three-cut set at 13-15 "
              "(secondary) the worst single run is "
              + ", ".join(f"{d} digits {fmt(orig[d][1], True)} (seed {orig[d][2]})" for d in (13, 14, 15)) + ".", ""]

    lines += ["## Carries per place in the kept training answers", "",
              "Share (%) of kept answers that carry out of a place, averaged over the places of each kind, in the "
              "round the length is first trained on; mean over the seeds (lowest–highest). The cut-1 boundary is "
              "where the no-carry rule forbids a carry; the agreement rule's cuts include cut 1 and the other cuts.", ""]
    for seeds, title in ((CONFIRMING, "Confirming runs (starting points 2-4)"), (LEARNING, "Learning runs (starting points 0-1)")):
        nc, ag = carries_per_place("nocarry", seeds), carries_per_place("agreement", seeds)
        rows = []
        for d in LENGTHS:
            if not nc[d]["cut 1"] and not ag[d]["cut 1"]:
                rows.append([str(d)] + ["not counted (round run before the carry audit, 14 Nov)"] + [""] * 5)
                continue
            rows.append([str(d), mean_range(nc[d]["cut 1"]), mean_range(nc[d]["other cuts"]), mean_range(nc[d]["no cut"]),
                         mean_range(ag[d]["cut 1"]), mean_range(ag[d]["other cuts"]), mean_range(ag[d]["no cut"])])
        lines += [f"**{title}**", ""] + table(
            ["Digits", "No-carry: cut 1", "No-carry: other cuts", "No-carry: no cut",
             "Agreement: cut 1", "Agreement: other cuts", "Agreement: no cut"], rows) + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------- check 3 on addition

TEACHER_ROWS = (("no_carry", "No-carry rule"), ("agreement", "Agreement rule"), ("no_selection", "No selection"),
                ("sample_and_vote", "Sample and vote"))
CHECK3_CSV = RESULTS_DIR / "check3_addition.csv"


def check3_cells() -> list[dict]:
    if not CHECK3_CSV.exists():
        return []
    with CHECK3_CSV.open() as f:
        return [r for r in csv.DictReader(f) if r["check_3"] != ""]


def check3_range(cells: list[dict], teacher: str, d: int, seeds: tuple[int, ...]) -> str:
    """Check 3 (%) over seeds and rounds at one length: one value, or lowest-highest."""
    xs = [100 * float(c["check_3"]) for c in cells
          if c["teacher"] == teacher and int(c["digits"]) == d and int(c["seed"]) in seeds]
    if not xs:
        return "–"
    return f"{min(xs):.1f}" if min(xs) == max(xs) else f"{min(xs):.1f}–{max(xs):.1f}"


def check3_addition_rows() -> list[tuple[str, list[list[str]]]]:
    """(block title, rows of teacher x length) for the paper test, confirming and learning runs."""
    blocks = []
    pt = json.loads((RESULTS_DIR / "paper_test.json").read_text()) if (RESULTS_DIR / "paper_test.json").exists() else None
    if pt:
        rows = []
        for t, label in TEACHER_ROWS:
            vals = pt["table"][t]
            rows.append([label] + [fmt(None if vals[str(d)].get("check_3") is None else 100 * vals[str(d)]["check_3"])
                                   for d in LENGTHS])
        blocks.append((f"Paper test (perfect calculator, {pt['problems_per_length']:,} problems per length)", rows))
    cells = check3_cells()
    for seeds, title in ((CONFIRMING, "Real models, confirming runs (starting points 2-4)"),
                         (LEARNING, "Real models, learning runs (starting points 0-1)")):
        if cells:
            blocks.append((title, [[label] + [check3_range(cells, t, d, seeds) for d in LENGTHS]
                                   for t, label in TEACHER_ROWS]))
    return blocks


def pct_or(x: str) -> str:
    return "–" if x in ("", "None") else f"{100 * float(x):.1f}"


def check3_md() -> str:
    lines = ["# Check 3 on addition", "",
             "Generated by `uv run python -m gainpred.writeup`; do not edit by hand. Analysis defined after the "
             "confirming runs (amendment 22); it does not enter the verdict.", "",
             "Share (%) of a teacher's kept answers identical to one of the piece answers it asked the model for: "
             "the piece sums of every cut it built (no-carry rule and no selection: cut 1; agreement rule: all its "
             "cuts, three at 8-12 digits and five at 13-15), or the five sampled direct answers (sample and vote). Problems set aside are skipped; "
             "an unreadable piece answer never matches. Real models: each round's check sample redrawn (2,000 "
             "problems per length) with the model before the round; range over starting points and rounds. "
             "No-carry rows include the weak starting models; no-selection rows include run f.", ""]
    for title, rows in check3_addition_rows():
        lines += [f"## {title}", ""] + table(["Teacher"] + [str(d) for d in LENGTHS], rows) + [""]
    cells = check3_cells()
    if cells:
        greedy = [c for c in cells if c["teacher"] != "sample_and_vote"]
        same = sum(c["reproduced"] == "True" for c in greedy)
        lines += ["## Reproducing the logged checks", "",
                  f"Greedy teachers: the redrawn sample gives exactly the logged kept count, check 1 and check 2b in "
                  f"{same} of {len(greedy)} cells.", ""]
        bad = [c for c in greedy if c["reproduced"] != "True"]
        if bad:
            lines += ["Cells that do not reproduce. Every one is at 13-15 digits in a run made before the five-cut "
                      "change (decision point 1, commit e440447): those runs drew their samples under the three-cut "
                      "rule that no piece of any cut starts with 0, and built the agreement rule and check 2b from "
                      "three cuts. The redraw uses the five cuts now in use, so at these lengths it draws a "
                      "different sample (cut 1 is unchanged, hence the close but unequal kept counts of the "
                      "no-carry rule). P9b scores only the two run-f cells marked yes.", ""]
            lines += table(["Run", "Round", "Digits", "In P9b", "Kept (logged / redrawn)", "Check 1 (logged / redrawn)",
                            "Check 2b (logged / redrawn)"],
                           [[c["run"], c["round"], c["digits"], "yes" if c["in_p9b"] == "True" else "no",
                             f"{c['logged_kept']} / {c['kept']}",
                             f"{pct_or(c['logged_check_1'])} / {pct_or(c['redrawn_check_1'])}",
                             f"{pct_or(c['logged_check_2b'])} / {pct_or(c['redrawn_check_2b'])}"] for c in bad]) + [""]
        three = RESULTS_DIR / "check3_addition_3cut.csv"
        if bad and three.exists():
            t = list(csv.DictReader(three.open()))
            ok = sum(r["reproduced"] == "True" for r in t)
            lines += [f"**Redrawn under the three-cut rule they ran under** (owner, 18 November 2026; "
                      f"`results/check3_addition_3cut.csv`): {ok} of {len(t)} cells now reproduce the logged kept "
                      f"count, check 1 and check 2b exactly. Check 3 in these cells: "
                      + ", ".join(sorted({pct_or(r['check_3']) for r in t})) + "%.", ""]
            miss = [r for r in t if r["reproduced"] != "True"]
            if miss:
                lines += table(["Run", "Round", "Digits", "Kept (logged / redrawn)", "Check 1 (logged / redrawn)",
                                "Check 2b (logged / redrawn)"],
                               [[r["run"], r["round"], r["digits"], f"{r['logged_kept']} / {r['kept']}",
                                 f"{pct_or(r['logged_check_1'])} / {pct_or(r['redrawn_check_1'])}",
                                 f"{pct_or(r['logged_check_2b'])} / {pct_or(r['redrawn_check_2b'])}"] for r in miss]) + [""]
        vote = [c for c in cells if c["teacher"] == "sample_and_vote"]
        empty = [r for r in csv.DictReader(CHECK3_CSV.open()) if r["check_3"] == ""]
        if empty:
            lines += [f"Cells where the teacher kept nothing (check 3 not defined): {len(empty)}, all sample and "
                      "vote: " + ", ".join(f"{r['run']} round {r['round']} {r['digits']} digits" for r in empty)
                      + ".", ""]
        if vote:
            d1 = [abs(float(c["redrawn_check_1"]) - float(c["logged_check_1"])) for c in vote
                  if c["redrawn_check_1"] and c["logged_check_1"]]
            lines += [f"Sample and vote ({len(vote)} cells) samples its answers, and its sampling stream had been "
                      "advanced by the round's build before the original checks, so its redrawn checks are close, "
                      "not equal: largest difference in check 1 "
                      f"{100 * max(d1):.1f} points (median {100 * statistics.median(d1):.1f})." if d1 else "", ""]
            def v(x):
                return None if x in ("", "None") else float(x)
            every_vote = [r for r in csv.DictReader(CHECK3_CSV.open()) if r["teacher"] == "sample_and_vote"]
            cross = [r for r in every_vote if v(r["logged_check_1"]) is not None and v(r["redrawn_check_1"]) is not None
                     and (v(r["logged_check_1"]) < 0.5) != (v(r["redrawn_check_1"]) < 0.5)]
            lines += [f"Cells where logged and redrawn check 1 fall on opposite sides of the 0.50 level: {len(cross)} "
                      f"of {len(every_vote)}.", ""]
            if cross:
                lines += table(["Run", "Round", "Digits", "In P9b", "Kept (logged / redrawn)", "Check 1 logged",
                                "Check 1 redrawn"],
                               [[r["run"], r["round"], r["digits"], "yes" if r["in_p9b"] == "True" else "no",
                                 f"{r['logged_kept']} / {r['kept']}", pct_or(r["logged_check_1"]),
                                 pct_or(r["redrawn_check_1"])] for r in cross]) + [""]
        every = list(csv.DictReader(CHECK3_CSV.open()))
        unread = sum(int(c["unreadable_pieces"]) for c in every)
        total = sum(int(c["pieces"]) for c in every)
        lines += [f"Unreadable piece answers: {unread:,} of {total:,} ({100 * unread / total:.3f}%).", ""]
    return "\n".join(lines)


CHECK3_READING = {"0.0": "Builds a new answer", "100.0": "Hands back"}
CHECK3_HALF = "Hands back about half, builds the rest"  # join at the cut: a new answer only when the run crosses the join
CHECK3_CAPTION = (
    "The construction check. Each value is the share (%) of a procedure's kept answers that are identical to "
    "one of the answers it was given for the pieces. At 0% the procedure always builds an answer that none of "
    "its pieces contained, so it can show the student something new. At 100% it only hands back an answer the "
    "model already gives. Sample and vote does not split the problem: its \u201cpieces\u201d are five direct "
    "answers sampled from the model, and it returns one of them. Perfect solver: every piece answered "
    "correctly, before any training. Trained models: each round's check sample, answered by the model before "
    "that round. Addition values are the same at every length from 8 to 15 digits, in every round and from "
    "every starting model. Longest-run values are the lowest and highest over lengths 12 to 20 (trained "
    "models: evaluation runs). The addition rows were added after all experiments were complete, so they "
    "did not influence the predictions.")


def check3_one_value(title: str, label: str, values: list[str]) -> str:
    """The single check-3 value of one addition procedure over lengths 8-15 (and seeds, rounds).

    The paper table shows one number per addition procedure, so it may only do that when
    every length gives the same single value; otherwise generation fails (the per-length
    table stays in results/writeup/check3_addition.md)."""
    if len(set(values)) != 1 or "–" in values[0]:
        raise ValueError(f"check 3 differs across lengths for {label} ({title}): {values}")
    return values[0]


def lo_hi(xs: list[float]) -> str:
    """Lowest-highest (one value if they are equal)."""
    if not xs:
        return "–"
    return f"{min(xs):.1f}" if f"{min(xs):.1f}" == f"{max(xs):.1f}" else f"{min(xs):.1f}–{max(xs):.1f}"


def check3_merged_rows() -> list[list[str]]:
    """One row per procedure: (task, procedure, perfect solver, trained models, reading).

    Addition: one value per procedure, asserted identical across lengths 8-15, rounds and
    starting models (learning and confirming), and equal for the perfect solver and the
    trained models. Longest run: lowest-highest over lengths 12-20 (perfect solver: the
    paper test; trained models: the logged checks of starting points 2-4)."""
    rows = []
    blocks = {title.split(" (")[0]: dict((r[0], r[1:]) for r in rs) for title, rs in check3_addition_rows()}
    order = ("Paper test", "Real models, learning runs", "Real models, confirming runs")
    for i, (_, label) in enumerate(TEACHER_ROWS):
        vals = [check3_one_value(b, label, blocks[b][label]) for b in order]
        if len(set(vals)) != 1 or vals[0] not in CHECK3_READING:
            raise ValueError(f"check 3 for {label} is not one of 0 and 100 everywhere: {vals}")
        rows.append(["Addition" if i == 0 else "", label, vals[0], vals[0], CHECK3_READING[vals[0]]])
    pt = json.loads((RESULTS_DIR / "paper_test_longest_run.json").read_text())["table"]
    for i, (t, label) in enumerate(RUN_TEACHERS):
        perfect = [100 * v["check_3"] for v in pt[t]["all"].values() if v["check_3"] is not None]
        trained = [100 * e["check_3"] for s in CONFIRMING for e in events(f"runs-{t}-seed{s}-u30000")
                   if e["event"] == "checks" and e.get("check_3") is not None]
        if t == "join_at_the_cut":
            reading = CHECK3_HALF
        else:
            if {min(perfect + trained), max(perfect + trained)} != {100.0}:
                raise ValueError(f"check 3 for {label} is not 100 everywhere")
            reading = CHECK3_READING["100.0"]
        rows.append(["Longest run" if i == 0 else "", label, lo_hi(perfect), lo_hi(trained), reading])
    return rows


def check3_tex() -> str:
    """The construction check for both tasks, one row per procedure (section 5)."""
    return tex_table("check3", CHECK3_CAPTION, ["Task", "Procedure", "Perfect solver", "Trained models", "Reading"],
                     check3_merged_rows(), spec="llrrl", span=("Answers handed back (%)", 3, 4), fit=True)


# ---------------------------------------------------------------- what the filter removes

FILTER_ARMS = (("nocarry", "No-carry rule"), ("agreement", "Agreement rule"))


def filter_gap_rows(arm: str, seeds: tuple[int, ...]) -> list[dict]:
    """Per seed and length, in the round that first trains the length: the share of kept training
    answers with a carry out of the cut-1 place, and the student's accuracy on the study group
    "carry at cut 1" before and after that round (plus the no-carry group after, to compare).

    Final-recipe runs only (``addition_run``: ``-authors``, and ``-5cut`` for seeds 0-1 rounds
    3-4); si-their-seed1-r1 and the first si-nocarry-seed1 are never read. Analysis defined
    after the confirming runs (amendment 22)."""
    rows = []
    for s in seeds:
        for d in LENGTHS:
            rnd = new_round(d)
            run = addition_run(arm, s, rnd)
            shares = carry_shares(arm, s, d)
            rows.append({"seed": s, "digits": d, "round": rnd, "run": run,
                         "kept_carry": None if shares is None else shares["cut 1"],
                         "before": addition_accuracy(run, rnd - 1, d, "carry"),
                         "after": addition_accuracy(run, rnd, d, "carry"),
                         "after_plain": addition_accuracy(run, rnd, d, "no-carry")})
    return rows


def filter_gap_summary() -> list[str]:
    """The summary paragraph, computed from the same rows as the tables."""
    def lag(r):
        return None if r["after"] is None or r["after_plain"] is None else r["after"] - r["after_plain"]

    def worst(rows):
        r = min((r for r in rows if lag(r) is not None), key=lag)
        return f"{-lag(r):.1f} points (seed {r['seed']}, {r['digits']} digits)"

    def span(xs):
        lo, hi = f"{min(xs):.1f}", f"{max(xs):.1f}"
        return lo if lo == hi else f"{lo}-{hi}"

    nc_c, ag_c = filter_gap_rows("nocarry", CONFIRMING), filter_gap_rows("agreement", CONFIRMING)
    nc_l, ag_l = filter_gap_rows("nocarry", LEARNING), filter_gap_rows("agreement", LEARNING)
    kept_nc = [r["kept_carry"] for r in nc_c if r["kept_carry"] is not None]
    kept_ag = [r["kept_carry"] for r in ag_c if r["kept_carry"] is not None]
    after_nc = [r["after"] for r in nc_c if r["after"] is not None]
    mid = [-lag(r) for r in ag_l if r["digits"] in (10, 11) and lag(r) is not None]
    big = min((r for r in ag_l if lag(r) is not None), key=lag)
    return ["**Summary.** Confirming runs: the no-carry rule keeps "
            f"{span(kept_nc)}% of training answers with a carry at cut 1, yet in the same round "
            f"the student reaches {span(after_nc)}% on the carry-at-cut-1 group; its "
            f"largest lag behind the no-carry group is {worst(nc_c)}. The agreement rule keeps "
            f"{span(kept_ag)}% such answers and lags slightly more, at most {worst(ag_c)}. "
            f"Learning runs lag more: the no-carry rule at most {worst(nc_l)}; the agreement rule "
            f"{span(mid)} points at 10-11 digits, and seed {big['seed']} at {big['digits']} digits "
            f"reaches {big['after']:.1f}% on the carry group against {big['after_plain']:.1f}% on the no-carry group."]


def filter_gap_md() -> str:
    lines = ["# What the filter removes, and whether the student learns it anyway", "",
             "Generated by `uv run python -m gainpred.writeup`; do not edit by hand. Analysis defined after the "
             "confirming runs (amendment 22).", "",
             "For each length, the round that first trains it (8-9: round 1, 10-11: round 2, 12-13: round 3, "
             "14-15: round 4). **Kept, carry at cut 1**: share (%) of kept training answers with a carry out of "
             "the cut-1 place (as in the carries table; for 15 digits, cut 1 has two boundaries and the share is "
             "their mean). **Carry group**: the student's accuracy (%) on the study-set group with a carry across "
             "cut 1, before and after the round; **no-carry group after**: the other group, for comparison; "
             "**lag**: carry group minus no-carry group after the round, in points. 13-15 digits: the five-cut "
             "study set. Final-recipe runs only (`-authors`; `-5cut` for seeds 0-1, rounds 3-4).", ""]
    lines += filter_gap_summary() + [""]
    for seeds, title in ((CONFIRMING, "Confirming runs (starting points 2-4)"),
                         (LEARNING, "Learning runs (starting points 0-1), shown separately")):
        lines += [f"## {title}", ""]
        for arm, label in FILTER_ARMS:
            rows = []
            for r in filter_gap_rows(arm, seeds):
                lag = None if r["after"] is None or r["after_plain"] is None else r["after"] - r["after_plain"]
                kc = "not counted (built before the carry audit, 14 Nov)" if r["kept_carry"] is None \
                    else fmt(r["kept_carry"])
                rows.append([str(r["digits"]), str(r["seed"]), str(r["round"]), kc, fmt(r["before"]),
                             fmt(r["after"]), fmt(r["after_plain"]), fmt(lag, True)])
            lines += [f"**{label}**", ""] + table(
                ["Digits", "Seed", "Round", "Kept, carry at cut 1", "Carry group before", "Carry group after",
                 "No-carry group after", "Lag"], rows) + [""]
    return "\n".join(lines)


def filter_gap_tex() -> str:
    """Confirming runs: per length, each rule's kept carry share (mean) and the carry group before and
    after (mean, with the lowest seed after), next to the no-carry group after (mean)."""
    def m(xs):
        xs = [x for x in xs if x is not None]
        return f"{sum(xs) / len(xs):.1f}" if xs else "--"
    by = {arm: filter_gap_rows(arm, CONFIRMING) for arm, _ in FILTER_ARMS}
    rows = []
    for d in LENGTHS:
        row = [str(d)]
        for arm, _ in FILTER_ARMS:
            rs = [r for r in by[arm] if r["digits"] == d]
            after = [r["after"] for r in rs if r["after"] is not None]
            row += [m([r["kept_carry"] for r in rs]), m([r["before"] for r in rs]),
                    f"{m(after)} ({min(after):.1f})" if after else "--", m([r["after_plain"] for r in rs])]
        rows.append(row)
    return tex_table(
        "filter_gap", "What the filter removes and whether the student learns it anyway, evaluation runs, in "
        "the round that first trains each length. Kept: share (%) of kept training answers with a carry out of "
        "the split-1 place. Carry: accuracy (%) on the test problems with a carry across split 1, before and after "
        "the round (mean over the evaluation runs; lowest in brackets). No-carry: the other group after. "
        "NC: no-carry rule; AG: agreement rule. Defined after the evaluation runs.",
        ["Digits", "NC kept", "NC carry before", "NC carry after", "NC no-carry after",
         "AG kept", "AG carry before", "AG carry after", "AG no-carry after"], rows, fit=True)


# ---------------------------------------------------------------- experiment 2

RUN_TEACHERS = (("join_at_the_cut", "Join at the cut"), ("edges_differ", "Edges-differ rule"),
                ("pick_the_longer", "Pick the longer"))
JOINED = range(2, 9)


def joined(acc: dict[str, float], n: int) -> float | None:
    v = acc.get(f"joined_length_{n}")
    return None if v is None else 100 * v


def exp2_rows(seeds: tuple[int, ...]) -> list[list[str]]:
    rows = []
    for s in seeds:
        before = evals(f"runs-join_at_the_cut-seed{s}-u30000", "study").get(0)
        if before:
            rows.append(["Before (shared)", str(s), fmt(100 * before["all"])] + [fmt(joined(before, n)) for n in JOINED])
        for t, label in RUN_TEACHERS:
            after = evals(f"runs-{t}-seed{s}-u30000", "study").get(1)
            if after:
                rows.append([label, str(s), fmt(100 * after["all"])] + [fmt(joined(after, n)) for n in JOINED])
    return rows


def withheld_rows(seeds: tuple[int, ...]) -> list[list[str]]:
    rows = []
    for s in seeds:
        for name, label in ((f"runs-join_at_the_cut-withheld-seed{s}-u30000", "Withheld"),
                            (f"runs-join_at_the_cut-withheld-seed{s}-longruns", "Control arm")):
            ev = evals(name, "study")
            if 0 not in ev or 1 not in ev:
                continue
            row = [label, str(s)]
            for n in (4, 5, 6, 7, 8):
                row.append(f"{fmt(joined(ev[0], n))} → {fmt(joined(ev[1], n))}")
            rows.append(row)
        # Observation (owner, 15 Nov 2026): the edges-differ rule's full-pool round at the same joined
        # lengths, for comparison with the control arm. Not a test; the direction differs by seed.
        ev = evals(f"runs-edges_differ-seed{s}-u30000", "study")
        if 0 in ev and 1 in ev:
            rows.append(["Edges-differ rule, full pool (observation)", str(s)]
                        + [f"{fmt(joined(ev[0], n))} → {fmt(joined(ev[1], n))}" for n in (4, 5, 6, 7, 8)])
    return rows


def split_rows(seeds: tuple[int, ...]) -> list[list[str]]:
    data = json.loads((RESULTS_DIR / "withheld_split.json").read_text())
    rows = []
    for s in seeds:
        for arm in ("withheld", "full pool"):
            before, after = data.get(f"seed{s} before"), data.get(f"seed{s} {arm}")
            if not before or not after:
                continue
            row = [arm, str(s)]
            for n in (4, 5, 6):
                for a in ("1-2", "3-4"):
                    key = f"{n}|{a}"
                    row.append(f"{100 * before[key][0]:.1f} → {100 * after[key][0]:.1f}")
            rows.append(row)
    return rows


CHECK3_LENGTHS = range(12, 21)
CHECK3_ARMS = tuple((f"runs-{t}-seed{{s}}-u30000", label) for t, label in RUN_TEACHERS) + (
    ("runs-join_at_the_cut-withheld-seed{s}-u30000", "Join at the cut, withheld"),
    ("runs-join_at_the_cut-withheld-seed{s}-longruns", "Join at the cut, withheld, control arm"))


def check3_rows(seeds: tuple[int, ...]) -> list[list[str]]:
    rows = []
    for pattern, label in CHECK3_ARMS:
        for s in seeds:
            checks = {e["length"]: e.get("check_3") for e in events(pattern.format(s=s)) if e["event"] == "checks"}
            if checks:
                rows.append([label, str(s)] + [fmt(None if checks.get(n) is None else 100 * checks[n])
                                               for n in CHECK3_LENGTHS])
    return rows


def exp2() -> str:
    head = ["Teacher", "Seed", "All"] + [f"Joined {n}" for n in JOINED]
    lines = ["# Experiment 2: longest run", "",
             "Generated by `uv run python -m gainpred.writeup`; do not edit by hand.", "",
             "One round from the 30,000-update starting models. Study set, accuracy (%): all groups, and problems "
             "whose longest run crosses the join, by joined run length.", "",
             "## Confirming runs (starting points 2-4)", ""]
    lines += table(head, exp2_rows(CONFIRMING))
    lines += ["", "## Learning runs (starting points 0-1), shown separately", ""] + table(head, exp2_rows(LEARNING))

    whead = ["Arm", "Seed"] + [f"Joined {n}" for n in (4, 5, 6, 7, 8)]
    lines += ["", "## Withheld version and control arm, side by side", "",
              "Before → after the withheld round of join at the cut (joined runs up to 6 taught; run-across problems "
              "of left part 3-4 withheld). Control arm: the same round from a starting model whose training included "
              "strings with a longest run of 7-8 inside (P6). The control arm is an observation, not a test. The "
              "edges-differ rows (observation) show the same full-pool round of the edges-differ rule, for comparison "
              "at joined runs of 5.", ""]
    lines += ["**Confirming runs**", ""] + table(whead, withheld_rows(CONFIRMING))
    lines += ["", "**Learning runs**", ""] + table(whead, withheld_rows(LEARNING))
    ctrl = {s: evals(f"runs-join_at_the_cut-withheld-seed{s}-longruns", "study") for s in CONFIRMING + LEARNING}
    j6 = {s: joined(e[1], 6) for s, e in ctrl.items() if 1 in e}
    j7 = {s: (joined(e[0], 7), joined(e[1], 7)) for s, e in ctrl.items() if 1 in e}
    j8 = {s: (joined(e[0], 8), joined(e[1], 8)) for s, e in ctrl.items() if 1 in e}
    low = min(j6, key=j6.get)
    lines += ["", "Control arm across all five seeds: joined runs of 6 after the round "
              f"{min(j6.values()):.0f}–{max(j6.values()):.0f}% (lowest: seed {low}, {j6[low]:.0f}%; the others "
              f"{min(v for s, v in j6.items() if s != low):.0f}–{max(v for s, v in j6.items() if s != low):.0f}%); "
              f"joined runs of 7 from {min(b for b, _ in j7.values()):.0f}–{max(b for b, _ in j7.values()):.0f}% before "
              f"to {min(a for _, a in j7.values()):.0f}–{max(a for _, a in j7.values()):.0f}% after; joined runs of 8 "
              f"from {min(b for b, _ in j8.values()):.0f}–{max(b for b, _ in j8.values()):.0f}% to "
              f"{min(a for _, a in j8.values()):.0f}–{max(a for _, a in j8.values()):.0f}%.", ""]

    lines += ["## Check 3: does the teacher hand back a piece answer?", "",
              "Share (%) of the teacher's outputs identical to one of the piece answers it was given (P8; "
              "\"hands back\" if 100%), per length, before the round. Computed on the check sample.", ""]
    for seeds, title in ((CONFIRMING, "Confirming runs"), (LEARNING, "Learning runs")):
        lines += [f"**{title}**", ""] + table(["Teacher", "Seed"] + [str(n) for n in CHECK3_LENGTHS],
                                              check3_rows(seeds)) + [""]

    shead = ["Arm", "Seed"] + [f"J{n} a={a}" for n in (4, 5, 6) for a in ("1-2", "3-4")]
    lines += ["## Withheld split by left part", "",
              "Run-across study problems with joined run 4-6, split by the left part a (a = 3-4 is never taught in "
              "the withheld pool). Direct answers, before → after (%). From `scripts/withheld_split.py`.", "",
              "**Confirming runs**", ""] + table(shead, split_rows(CONFIRMING))
    lines += ["", "**Learning runs**", ""] + table(shead, split_rows(LEARNING)) + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------- experiment 3

def load_cells() -> list:
    """P9b's cells from results/p9b_cells.csv, as p9b.Cell objects with outcome and verdict."""
    from gainpred import p9b

    cells = []
    with (RESULTS_DIR / "p9b_cells.csv").open() as f:
        for r in csv.DictReader(f):
            def num(k):
                return float(r[k]) if r[k] not in ("", "None") else None
            c = p9b.Cell(run=r["run"], teacher=r["teacher"], seed=int(r["seed"]), round=int(r["round"]),
                         digits=int(r["digits"]), new_length=r["new_length"] == "True",
                         check_0={"True": True, "False": False}.get(r["check_0"]), check_1=num("check_1"),
                         check_2b=num("check_2b"), check_2_kept=num("check_2_kept"), teachable=num("teachable"),
                         kept_share=num("kept_share"), test_set=r["test_set"], before=num("before"),
                         after=num("after"), easy_change=num("easy_change"), best_after=num("best_after"),
                         mid_start=r["mid_start"] == "True")
            c.outcome, c.verdict = r["outcome"], r["verdict"]
            cells.append(c)
    return cells


def exp3() -> str:
    from gainpred import p9b

    cells = load_cells()
    levels = (0.38, p9b.LEVEL_1)
    lines = ["# Experiment 3: checks before training against outcomes (P9b)", "",
             "Generated by `uv run python -m gainpred.writeup` from `results/p9b_cells.csv`; do not edit by hand. "
             f"Frozen rule: 2b level {levels[0]}, check 1 level {levels[1]}.", ""]
    scored = [c for c in cells if c.outcome not in ("unclassified", "missing")]
    lines += p9b.support_lines(scored)[:4]
    lines += p9b.decision_point_2_lines(cells, *levels)
    mg = margins(evaluation_cases())
    lines += [f"**Margins (computed after the evaluation runs):** among cases that pass check 0 and keep at least 5%, "
              f"check 1 is at most {100 * mg['novelty_hi']:.1f}% where the verdict is does nothing and at least "
              f"{100 * mg['novelty_lo']:.1f}% elsewhere; check 2b is at most {100 * mg['confirmed_hi']:.1f}% where it "
              f"is does harm ({mg['n_harm']} cells) and at least {100 * mg['confirmed_lo']:.1f}% where it is pays off "
              f"({mg['n_pays']} cells). Any check-1 level in between, and any 2b level in between, gives the same "
              "verdicts (checked).", ""]
    learn = [c for c in scored if c.seed in LEARNING]
    hits = sum(c.outcome in p9b.MATCH[c.verdict] for c in learn)
    lines += ["## Learning runs (starting points 0-1), shown separately", "",
              f"Where the 2b level was set, so not a test of it: {hits} of {len(learn)} classified cells match "
              f"({100 * hits / max(len(learn), 1):.1f}%).", ""]
    rows = []
    for case in p9b.CASES:
        g = [c for c in learn if p9b.case_of(c) == case]
        if g:
            rows.append([case, str(len(g)), str(sum(c.outcome in p9b.MATCH[c.verdict] for c in g))])
    lines += table(["Case", "Cells", "Verdict matches"], rows) + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------- cost

def replay_tokens_estimate(per_length: int = 5_000) -> float:
    """Expected tokens the model writes answering the addition replay (easy problems, 3-7 digits).

    Not logged on its own: the round log sums replay and check tokens. Estimated
    as the length of the true answer plus the end token, averaged over random
    problems of each size (both numbers d digits, no leading 0), which is exact
    for a replay answered 100% right. Fixed seed, so the estimate never changes.
    """
    import random
    rng, total = random.Random(0), 0.0
    for d in range(3, 8):
        lo, hi = 10 ** (d - 1), 10 ** d - 1
        n = 20_000
        total += per_length * sum(len(str(rng.randint(lo, hi) + rng.randint(lo, hi))) + 1 for _ in range(n)) / n
    return total


def round_cost(run: str, rnd: int, addition: bool, replay_est: float) -> dict | None:
    """Seconds and tokens of one round, split into building, replay, checks and training."""
    ev = events(run)
    done = next((e for e in ev if e["event"] == "round_done" and e["round"] == rnd), None)
    if done is None:
        return None
    builds = [e for e in ev if e["event"] == "build" and e["round"] == rnd]
    build_s = sum(e["seconds"] for e in builds)
    build_t = sum(e["generated_tokens"] for e in builds)
    replay_s = done["seconds"].get("build", 0.0) - build_s  # replay is timed under "build"
    rest_t = done["generated_tokens"] - build_t  # replay + checks
    replay_t = replay_est if addition else 0.0  # longest run replays true answers: nothing generated
    train_log = RUNS / run / f"round_{rnd}" / "log.jsonl"
    train_s = train_t = None
    if train_log.exists():
        lines = [json.loads(x) for x in train_log.read_text().splitlines()]
        train_s = lines[-1].get("seconds", {}).get("train")
        trains = [x for x in lines if x.get("event") == "train"]
        train_t = trains[-1].get("answer_tokens") if trains else None
    return {"build_s": build_s, "build_t": build_t, "replay_s": replay_s, "replay_t": replay_t,
            "checks_s": done["seconds"].get("checks", 0.0), "checks_t": rest_t - replay_t,
            "train_s": train_s, "train_t": train_t}


COST_ARMS = tuple((f"si-{arm}-seed{{s}}-authors", label, True, range(1, 5)) for arm, label in ARMS) + (
    ("run f", "No selection after the no-carry rule", True, range(3, 4)),
    ("weak", "No-carry rule, weak starting models", True, range(1, 2)),
) + tuple((f"runs-{t}-seed{{s}}-u30000", label, False, range(1, 2)) for t, label in RUN_TEACHERS) + (
    ("runs-join_at_the_cut-withheld-seed{s}-u30000", "Join at the cut, withheld", False, range(1, 2)),
    ("runs-join_at_the_cut-withheld-seed{s}-longruns", "Join at the cut, withheld, control arm", False, range(1, 2)),
)


ADDITION_COST_LABELS = frozenset(label for _, label, addition, _ in COST_ARMS if addition)


def cost_runs(pattern: str, seeds: tuple[int, ...]) -> list[str]:
    if pattern == "run f":
        return [runf_run(s) for s in seeds]
    if pattern == "weak":
        return sorted(p.name for p in RUNS.iterdir() if "-weak-" in p.name
                      and any(p.name.endswith(f"-seed{s}") or p.name.endswith(f"-seed{s}-r2") for s in seeds)
                      and not (p.name == "si-nocarry-weak-s1425-seed2"))  # failed at start; its -r2 is the run
    return [pattern.format(s=s) for s in seeds]


def cost_md() -> str:
    replay_est = replay_tokens_estimate()
    lines = ["# Cost of a round, by part", "",
             "Generated by `uv run python -m gainpred.writeup` from the round logs; do not edit by hand.", "",
             "Per teacher and round: mean over the runs (seconds on one A10 GPU; tokens in thousands). "
             "*Build* = the teacher building its kept answers; *replay* = easy problems answered by the model "
             "(addition only; longest run replays true answers); *checks* = P8 checks at every length; *train* = "
             "3,000 updates. Check share = checks seconds as a share of build + replay + checks + train seconds. "
             "Evaluation on the test sets is not timed and not included. Generated tokens are what the model "
             "wrote; train tokens are answer tokens trained on (not generated).", "",
             f"Check tokens: exact for longest run. For addition the log sums replay and check tokens; the replay "
             f"share is estimated at {replay_est / 1000:,.0f} thousand tokens a round (true-answer lengths, exact "
             "when the replay is answered 100% right) and subtracted.", ""]
    head = ["Teacher", "Round", "Runs", "Build s", "Build tokens (k)", "Replay s", "Checks s", "Checks tokens (k)",
            "Train s", "Train answer tokens (k)", "Check share"]
    for seeds, title in ((CONFIRMING, "Confirming runs (starting points 2-4)"),
                         (LEARNING, "Learning runs (starting points 0-1)")):
        rows, tot = [], {"checks": 0.0, "all": 0.0}
        for pattern, label, addition, rounds in COST_ARMS:
            for rnd in rounds:
                runs = cost_runs(pattern, seeds)
                if seeds == LEARNING and addition and pattern.startswith("si-") and rnd >= 3:
                    runs = [r.replace("-authors", "-5cut") for r in runs]
                cs = [c for c in (round_cost(r, rnd, addition, replay_est) for r in runs) if c and c["train_s"]]
                if not cs:
                    continue
                m = {k: sum(c[k] for c in cs) / len(cs) for k in cs[0]}
                whole = m["build_s"] + m["replay_s"] + m["checks_s"] + m["train_s"]
                tot["checks"] += sum(c["checks_s"] for c in cs)
                tot["all"] += sum(c["build_s"] + c["replay_s"] + c["checks_s"] + c["train_s"] for c in cs)
                rows.append([label, str(rnd), str(len(cs)), f"{m['build_s']:.0f}", f"{m['build_t'] / 1000:,.0f}",
                             f"{m['replay_s']:.0f}", f"{m['checks_s']:.0f}", f"{m['checks_t'] / 1000:,.0f}",
                             f"{m['train_s']:.0f}", f"{m['train_t'] / 1000:,.0f}", f"{100 * m['checks_s'] / whole:.0f}%"])
        lines += [f"## {title}", ""] + table(head, rows)
        lines += ["", f"Over every round above: checks take {100 * tot['checks'] / max(tot['all'], 1):.0f}% of the "
                  f"timed round seconds ({tot['checks'] / 3600:.2f} of {tot['all'] / 3600:.2f} GPU hours).", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- run list

def first_commit(name: str) -> str:
    out = subprocess.run(["git", "log", "--diff-filter=A", "--format=%h", "--", f"results/runs/{name}"],
                         capture_output=True, text=True).stdout.split()
    return out[-1] if out else "?"


def runs_md() -> str:
    from gainpred.status import collect

    _, _, states, seconds, notes, launches = collect(RESULTS_DIR)
    lines = ["# Every run", "", "Generated by `uv run python -m gainpred.writeup`; do not edit by hand. Every run the "
             "project has started, including failed, superseded and parked ones. Commit = the commit the launcher "
             "recorded; runs from before the launcher (13 November 2026) have none recorded, and the commit that "
             "first saved their results is given instead.", ""]
    rows = []
    for name in sorted(states):
        e = launches.get(name)
        commit = e["commit"][:7] if e and e.get("commit") else f"not recorded (results first saved in {first_commit(name)})"
        rows.append([f"`{name}`", states[name], commit, f"{seconds.get(name, 0) / 60:.0f}",
                     notes.get(name, {}).get("note", "")])
    lines += table(["Run", "State", "Commit", "GPU minutes", "Note"], rows)
    lines += ["", f"{len(rows)} runs.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- figures

def figures() -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gainpred import p9b

    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42})
    made = []

    # Figure 1: experiment 1, accuracy per length after round 4, confirming seeds.
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    series1 = [(lambda s: addition_run("nocarry", s, 4), 4, "mean", COLOURS[0], "-", "No-carry rule"),
               (lambda s: addition_run("agreement", s, 4), 4, "mean", COLOURS[1], "-", "Agreement rule"),
               (lambda s: addition_run("noselect", s, 4), 4, "no-carry", COLOURS[2], "-", "No selection, no-carry group"),
               (lambda s: addition_run("noselect", s, 4), 4, "carry", COLOURS[2], "--", "No selection, carry group (drawn 2 points up)"),
               (lambda s: addition_run("vote", s, 4), 4, "mean", COLOURS[3], "-", "Sample and vote"),
               (runf_run, 3, "no-carry", "#6250d6", "-", "No selection after no-carry, no-carry group"),
               (runf_run, 3, "carry", "#6250d6", "--", "No selection after no-carry, carry group")]
    for run_of, rnd, group, colour, ls, label in series1:
        ys, lo, hi = [], [], []
        for d in LENGTHS:
            v = [addition_accuracy(run_of(s), rnd, d, group) for s in CONFIRMING]
            v = [x for x in v if x is not None]
            ys.append(sum(v) / len(v)); lo.append(min(v)); hi.append(max(v))
        # The no-selection carry line sits at 0%, under run f's carry line and sample and vote:
        # draw it 2 points up with square markers (said in its label) so it stays visible.
        lift, marker = (2.0, "s") if "drawn 2 points up" in label else (0.0, "o")
        ax.errorbar(LENGTHS, [y + lift for y in ys],
                    yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]],
                    color=colour, lw=2, ls=ls, marker=marker, ms=5 if lift else 4, capsize=2, label=label,
                    zorder=3 if lift else 2)
    ax.set_xlabel("Digits"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(-3, 103)
    ax.set_title("Experiment 1: evaluation runs after round 4 (no selection after no-carry: round 3); mean of three "
                 "starting models, range as bars",
                 loc="left", fontsize=9)
    ax.legend(frameon=False, ncol=2, fontsize=7, loc="center left", bbox_to_anchor=(0.02, 0.62))
    fig.tight_layout(); fig.savefig(OUT / "figures" / "exp1_accuracy.png", dpi=200); fig.savefig(PAPER / "figure_exp1.pdf"); plt.close(fig)
    made.append("exp1_accuracy.png")

    # Figure 2: experiment 2, joined-run accuracy before and after, confirming seeds.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    xs = list(JOINED)
    def series(run, rnd):
        out = []
        for n in xs:
            v = [joined(evals(run.format(s=s), "study")[rnd], n) for s in CONFIRMING]
            out.append(v)
        return out
    def draw(ax, data, colour, label, ls="-"):
        m = [sum(v) / len(v) for v in data]
        ax.errorbar(xs, m, yerr=[[a - min(v) for a, v in zip(m, data)], [max(v) - a for a, v in zip(m, data)]],
                    color=colour, lw=2, ls=ls, marker="o", ms=4, capsize=2, label=label)
    draw(axes[0], series("runs-join_at_the_cut-seed{s}-u30000", 0), NEUTRAL, "Before")
    for (t, label), colour in zip(RUN_TEACHERS, COLOURS):
        draw(axes[0], series(f"runs-{t}-seed{{s}}-u30000", 1), colour, paper_words(label))
    axes[0].set_title("Three teachers, full pool", loc="left", fontsize=9)
    draw(axes[1], series("runs-join_at_the_cut-withheld-seed{s}-u30000", 0), NEUTRAL, "Before")
    draw(axes[1], series("runs-join_at_the_cut-withheld-seed{s}-u30000", 1), COLOURS[0], "Withheld, after")
    draw(axes[1], series("runs-join_at_the_cut-withheld-seed{s}-longruns", 0), NEUTRAL, "Control, before", ls="--")
    draw(axes[1], series("runs-join_at_the_cut-withheld-seed{s}-longruns", 1), COLOURS[1], "Control, after")
    axes[1].axvspan(6.5, 8.4, color="#f1f0ec", zorder=0)
    axes[1].set_title("Withheld round (crossing runs of 7-8: answers never shown)", loc="left", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Crossing run length"); ax.set_xticks(xs); ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Accuracy (%), evaluation runs")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "exp2_joined.png", dpi=200); fig.savefig(PAPER / "figure_exp2.pdf"); plt.close(fig)
    made.append("exp2_joined.png")

    # Figure 3: experiment 3, novelty (all cases with an outcome) and confirmed share (cases that
    # reach the agreement check) by outcome. Dashed: the frozen levels. Shaded: the margins, the
    # range of levels that give the same predictions on the evaluation runs (computed after them).
    rows = list(csv.DictReader((RESULTS_DIR / "p9b_cells.csv").open()))
    mg = margins(evaluation_cases())
    groups = (("paid off", ("paid off",)), ("did nothing", ("did nothing",)),
              ("did harm or taught\nwrong answers", HARM))
    runs_label = {CONFIRMING: "Evaluation runs", LEARNING: "Calibration runs"}
    fig, (ax1, ax) = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)

    def panel(axis, key, keep, level, lo, hi, title, level_label):
        for i, (_, members) in enumerate(groups):
            for seeds, colour, dx in ((CONFIRMING, COLOURS[0], 0.12), (LEARNING, NEUTRAL, -0.12)):
                v = [100 * float(r[key]) for r in rows if int(r["seed"]) in seeds and r["outcome"] in members
                     and r[key] not in ("", "None") and keep(r)]
                jitter = [((k * 37) % 21 - 10) / 120 for k in range(len(v))]
                axis.scatter([i + dx + j for j in jitter], v, s=10, color=colour, alpha=0.65,
                             label=runs_label[seeds] if i == 0 else None, zorder=3)
        axis.axhspan(100 * lo, 100 * hi, color="#f1f0ec", zorder=0)
        axis.axhline(100 * level, color="#0b0b0b", lw=1, ls="--", zorder=2)
        axis.text(2.45, 100 * level + 2, level_label, fontsize=7, ha="right")
        axis.text(2.45, 100 * lo + 1, f"same predictions {100 * lo:.1f}-{100 * hi:.1f}", fontsize=6.5, ha="right",
                  color="#52514e")
        axis.set_xticks(range(3)); axis.set_xticklabels([g[0] for g in groups])
        axis.set_ylim(-3, 103)
        axis.set_title(title, loc="left", fontsize=9)

    panel(ax1, "check_1", lambda r: True, p9b.LEVEL_1, mg["novelty_hi"], mg["novelty_lo"], "Novelty",
          f"level {100 * p9b.LEVEL_1:.0f}%")
    ax1.set_ylabel("Share (%)")
    panel(ax, "check_2b", lambda r: (r["check_0"] == "True" and float(r["check_1"]) >= p9b.LEVEL_1
                                     and float(r["kept_share"]) >= p9b.MIN_KEPT_SHARE),
          0.38, mg["confirmed_hi"], mg["confirmed_lo"], "Confirmed share", "level 38%")
    fig.legend(*ax1.get_legend_handles_labels(), frameon=False, fontsize=7, ncol=2, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "figures" / "exp3_checks.png", dpi=200); fig.savefig(PAPER / "figure_checks.pdf"); plt.close(fig)
    made.append("exp3_checks.png")

    # Amendment 24: change against novelty (Part A) and teacher accuracy (Part B, once fetched).
    novelty_gain_figure(plt)
    made.append("novelty_gain.png")
    if teacher_accuracy_figure(plt):
        made.append("teacher_accuracy.png")
    return made


# ---------------------------------------------------------------- baseline ("train briefly and look")

def baseline_comparison():
    """Our verdicts and the baseline's on the same confirming cells, keyed by (run, round, digits)."""
    from gainpred import baseline, p9b

    res = baseline.score()
    ours = {(c.run, c.round, c.digits): c for c in load_cells()}
    out = {}
    for name in ("learning", "confirming"):
        rows = []
        for b in res[name]["scored"]:
            c = ours[(b["run"], b["round"], b["digits"])]
            rows.append({"case": p9b.case_of(c), "outcome": c.outcome, "ours": c.verdict,
                         "baseline": b["baseline_verdict"], "favour": b["favour_verdict"]})
        out[name] = rows
    lv = json.loads(baseline.LEVELS.read_text())
    lv["favour"] = json.loads(baseline.LEVELS_FAVOUR.read_text())
    return out, lv


def probe_log(run: str) -> list[dict]:
    """The scored probe of a run (its -r2 rerun if the first probe failed), merged over attempts."""
    for name in (f"probe-{run}-r2", f"probe-{run}"):
        path = RESULTS_DIR / "probes" / f"{name}.jsonl"
        if path.exists() and '"event": "done"' in path.read_text():
            return [json.loads(x) for x in path.read_text().splitlines()]
    return []


def baseline_cost(seeds: tuple[int, ...]):
    """Per teacher: probe seconds (build + replay + 200 updates + eval) and the full round's
    (build + replay + checks + 3,000 updates + eval), mean per round. The original rounds did not
    time their evaluation, so the probe's own evaluation time (same test sets) stands in for it."""
    est, rows = replay_tokens_estimate(), []
    for pattern, label, addition, rounds in COST_ARMS:
        probe_s, full_s, checks_s, n = 0.0, 0.0, 0.0, 0
        for rnd in rounds:
            runs = cost_runs(pattern, seeds)
            if seeds == LEARNING and addition and pattern.startswith("si-") and rnd >= 3:
                runs = [r.replace("-authors", "-5cut") for r in runs]
            for r in runs:
                c = round_cost(r, rnd, addition, est)
                done = next((e for e in probe_log(r) if e["event"] == "probe_done" and e["round"] == rnd), None)
                if not c or not c["train_s"] or not done:
                    continue
                p = done["seconds"]
                probe_s += p["build"] + p["replay"] + p["train"] + p["eval"]
                full_s += c["build_s"] + c["replay_s"] + c["checks_s"] + c["train_s"] + p["eval"]
                checks_s += c["checks_s"]
                n += 1
        if n:
            rows.append((label, n, probe_s / n, full_s / n, checks_s / n))
    return rows


RULES = (("ours", "Our checks (before training, no reference answers)"),
         ("baseline", "Baseline as run (200 updates, then look)"),
         ("favour", "Baseline variant, defined after the confirming runs, in the baseline's favour"))


def baseline_md() -> str:
    from gainpred import p9b

    comp, lv = baseline_comparison()
    lf = lv["favour"]
    cost = baseline_cost(CONFIRMING)
    tf = sum(n * f for _, n, _, f, _ in cost)
    probe_share = 100 * sum(n * p for _, n, p, _, _ in cost) / tf
    checks_share = 100 * sum(n * c for _, n, _, _, c in cost) / tf
    lines = ["# Baseline: train briefly and look", "",
             "Generated by `uv run python -m gainpred.writeup`; do not edit by hand. Protocol amendments 20 and 21.", "",
             "**The main claim is about access and cost, not only agreement.** The baseline trains 200 updates on "
             "the teacher's rebuilt training data and reads accuracy on the test problems, which needs their "
             "reference answers. Our checks are computed before any training and never look at a right answer. "
             f"Per round (confirming runs) our checks cost {checks_share:.0f}% of a full round; the baseline "
             f"{probe_share:.0f}%.", "",
             f"Levels, fitted on the learning runs only (starting points 0-1). As run (amendment 20): G = {lv['G']}, "
             f"H = {lv['H']}, D = {lv['D']}, committed before any confirming probe was fetched. Variant in the "
             f"baseline's favour (amendment 21, defined after the confirming runs): Z = {lf['Z']}, G = {lf['G']}, "
             f"H = {lf['H']}, D = {lf['D']}, committed before the confirming runs were scored with it. The variant "
             "asks whether a new length changed at all (Z) before testing the gap to the best teacher.", ""]
    head = ["Rule", "Cells matched", "Share", "Uses reference answers", "Cost per round"]
    for name, title in (("confirming", "Confirming runs (starting points 2-4)"),
                        ("learning", "Learning runs (starting points 0-1), where the levels were fitted")):
        rows = comp[name]
        n = len(rows)
        cost_n = baseline_cost(CONFIRMING if name == "confirming" else LEARNING)
        tfn = sum(k * f for _, k, _, f, _ in cost_n)
        ps = 100 * sum(k * p for _, k, p, _, _ in cost_n) / tfn
        cs_ = 100 * sum(k * c for _, k, _, _, c in cost_n) / tfn
        body = []
        for key, label in RULES:
            m = sum(r["outcome"] in p9b.MATCH[r[key]] for r in rows)
            body.append([label, f"{m}/{n}", f"{100 * m / n:.1f}%", "no" if key == "ours" else "yes",
                         f"{cs_:.0f}% of a round" if key == "ours" else f"{ps:.0f}% of a round"])
        lines += [f"## {title}", "", *table(head, body), ""]
        grp = []
        for gname, members in (("paid off", {"paid off"}), ("did nothing", {"did nothing"}),
                               ("did harm or taught wrong answers", {"did harm", "taught wrong answers"})):
            g = [r for r in rows if r["outcome"] in members]
            grp.append([gname, str(len(g))] + [str(sum(r["outcome"] in p9b.MATCH[r[k]] for r in g)) for k, _ in RULES])
        lines += ["By outcome group:", "", *table(["Outcome group", "Cells", "Ours", "As run", "Variant"], grp), ""]
        cs = []
        for case in p9b.CASES:
            g = [r for r in rows if r["case"] == case]
            if g:
                cs.append([case, str(len(g))] + [str(sum(r["outcome"] in p9b.MATCH[r[k]] for r in g)) for k, _ in RULES])
        lines += ["By case:", "", *table(["Case", "Cells", "Ours", "As run", "Variant"], cs), ""]
    lines += ["## Cost per round", "",
              "Mean seconds per round on one A10 GPU. Probe = rebuild the training data + replay + 200 updates + "
              "evaluation (the same for both baseline rules). Full round = build + replay + checks + 3,000 updates + "
              "evaluation (the original rounds did not time evaluation; the probe's evaluation time on the same test "
              "sets is used). Shares are of the full round.", ""]
    for seeds, title in ((CONFIRMING, "Confirming runs"), (LEARNING, "Learning runs")):
        rows = baseline_cost(seeds)
        tp = sum(n * p for _, n, p, _, _ in rows); tf2 = sum(n * f for _, n, _, f, _ in rows)
        tc = sum(n * c for _, n, _, _, c in rows)
        lines += [f"**{title}**", "", *table(
            ["Teacher", "Rounds", "Probe s", "Full round s", "Probe share", "Our checks s", "Checks share"],
            [[lab, str(n), f"{p:.0f}", f"{f:.0f}", f"{100 * p / f:.0f}%", f"{c:.0f}", f"{100 * c / f:.0f}%"]
             for lab, n, p, f, c in rows]),
            "", f"Over all rounds: probe {100 * tp / tf2:.0f}% of the full round; our checks {100 * tc / tf2:.0f}%.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- paper (LaTeX)

GENERATED_NOTE = "% Generated by `uv run python -m gainpred.writeup`; do not edit by hand.\n"


def tex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("→", r"$\rightarrow$"), ("–", "--"), ("−", "$-$"), ("\u201c", "``"), ("\u201d", "''")):
        text = text.replace(a, b)
    return text


def paper_words(text: str) -> str:
    """The paper's name for a procedure whose code name uses the old word "cut"."""
    return (text.replace("Join at the cut", "Join at the split").replace("join at the cut", "join at the split")
            .replace(" (run f)", ""))


def tex_table(label: str, caption: str, header: list[str], rows: list[list[str]], spec: str | None = None,
              small: bool = True, fit: bool = False, rule_before: set[int] = frozenset(),
              bold: set[tuple[int, int]] = frozenset(), span: tuple[str, int, int] | None = None) -> str:
    """A booktabs table; every cell is escaped. Captions say what is shown, not what it means.
    ``fit`` scales a wide table down to the text width. ``bold``: (row, column) cells whose leading
    value (the mean, not the range after it) is set in bold. ``span`` = (text, first, last): a header
    spanning columns first..last (1-based) above the column names, with a rule under it."""
    spec = spec or "l" + "r" * (len(header) - 1)
    lines = [GENERATED_NOTE.rstrip(), r"\begin{table}[htbp]", r"\centering", rf"\caption{{{tex_escape(caption)}}}",
             rf"\label{{tab:{label}}}"] + ([r"\small"] if small else []) + ([r"\resizebox{\linewidth}{!}{%"] if fit else []) + [
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    if span:
        text, first, last = span
        cells = [""] * (first - 1) + [rf"\multicolumn{{{last - first + 1}}}{{c}}{{{tex_escape(text)}}}"] + \
            [""] * (len(header) - last)
        lines += [" & ".join(cells) + r" \\", rf"\cmidrule(lr){{{first}-{last}}}"]
    lines += [" & ".join(tex_escape(h) for h in header) + r" \\", r"\midrule"]
    def cell(i, j, c):
        text = tex_escape(paper_words(c))
        return re.sub(r"^(\S+)", r"\\textbf{\1}", text, count=1) if (i, j) in bold else text
    for i, r in enumerate(rows):
        lines += ([r"\midrule"] if i in rule_before else []) + [" & ".join(cell(i, j, c) for j, c in enumerate(r)) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}"] + (["}"] if fit else []) + [r"\end{table}"]
    return "\n".join(lines) + "\n"


def tex_figure(label: str, pdf: str, caption: str, width: str = r"\linewidth") -> str:
    return "\n".join([GENERATED_NOTE.rstrip(), r"\begin{figure}[htbp]", r"\centering",
                      rf"\includegraphics[width={width}]{{generated/{pdf}}}", rf"\caption{{{tex_escape(caption)}}}",
                      rf"\label{{fig:{label}}}", r"\end{figure}"]) + "\n"


# ---------------------------------------------------------------- section 4: numbers and tables
#
# Everything here reads results/p9b_cells.csv (through load_cells) and the logs; nothing is
# rescored. "Evaluation cases" are the cells of starting models 2-4 (the confirming runs);
# "with an outcome" leaves out the unclassified ones. The paper's words: prediction (verdict),
# case (cell), basics / novelty / agreement check (checks 0 / 1 / 2b), confirmed share (2b).

HARM = ("did harm", "taught wrong answers")
OUTCOME_ORDER = ("paid off", "did nothing", "did harm", "taught wrong answers")
TEACHER_LABEL = dict(TEACHER_ROWS)


def evaluation_cases(with_outcome: bool = True) -> list:
    """Cases of the evaluation runs (starting models 2-4); by default only those with an outcome."""
    cells = [c for c in load_cells() if c.seed in CONFIRMING]
    return [c for c in cells if c.outcome not in ("unclassified", "missing")] if with_outcome else cells


def correct(c) -> bool:
    from gainpred import p9b
    return c.outcome in p9b.MATCH[c.verdict]


def decided_by(c) -> str:
    """Which check of the frozen rule settles the prediction: basics, novelty (which includes the
    5% minimum kept share) or agreement. Mirrors ``p9b.verdict_of`` step by step."""
    from gainpred import p9b
    if c.check_0 is False:
        return "basics"
    if (c.kept_share is not None and c.kept_share < p9b.MIN_KEPT_SHARE) or c.check_1 is None or c.check_1 < p9b.LEVEL_1:
        return "novelty"
    return "agreement"


def by_min_kept(c) -> bool:
    """Decided by the 5% minimum: the teacher kept under 5% of the check sample."""
    from gainpred import p9b
    return decided_by(c) == "novelty" and c.kept_share is not None and c.kept_share < p9b.MIN_KEPT_SHARE


def margins(conf: list) -> dict[str, float | int]:
    """How far the two levels could move without changing any prediction (computed after the
    evaluation runs). Among cases that pass the basics check and keep at least 5%:

    * novelty: the highest value predicted "does nothing" and the lowest value predicted
      anything else -- any novelty level above the first and up to the second gives the
      same predictions;
    * confirmed share, among cases that reach the agreement check: the highest value
      predicted "does harm" and the lowest predicted "pays off".

    Raises if a level inside either interval changes a prediction.
    """
    from gainpred import p9b

    el = [c for c in conf if c.check_0 is not False and c.kept_share is not None and c.kept_share >= p9b.MIN_KEPT_SHARE]
    nothing = [c.check_1 for c in el if c.verdict == "does nothing" and c.check_1 is not None]
    rest = [c.check_1 for c in el if c.verdict != "does nothing"]
    harm = [c.check_2b for c in el if c.verdict == "does harm" and decided_by(c) == "agreement"]
    pays = [c.check_2b for c in el if c.verdict == "pays off"]
    m = {"novelty_hi": max(nothing), "novelty_lo": min(rest), "confirmed_hi": max(harm), "n_harm": len(harm),
         "confirmed_lo": min(pays), "n_pays": len(pays)}
    frozen = [c.verdict for c in conf]

    def levels(lo, hi, k=40):
        return [lo + (hi - lo) * (i + 1) / k for i in range(k)]  # (lo, hi], both ends of the open side tested
    for x in levels(m["novelty_hi"], m["novelty_lo"]):
        if [p9b.verdict_of(c, 0.38, x) for c in conf] != frozen:
            raise ValueError(f"novelty level {x} changes a prediction")
    for x in levels(m["confirmed_hi"], m["confirmed_lo"]):
        if [p9b.verdict_of(c, x, p9b.LEVEL_1) for c in conf] != frozen:
            raise ValueError(f"confirmed-share level {x} changes a prediction")
    return m


def section4_numbers() -> dict[str, str]:
    """Every number section 4's text cites, as LaTeX macro name -> value (``generated/numbers.tex``)."""
    from gainpred import p9b

    allc, conf = evaluation_cases(False), evaluation_cases()
    n, hits = len(conf), sum(map(correct, conf))
    nums = {"gpCases": n, "gpCorrect": hits, "gpCorrectPct": f"{100 * hits / n:.1f}",
            "gpCasesAll": len(allc), "gpNoOutcome": len(allc) - n, "gpCorrectAllPct": f"{100 * hits / len(allc):.1f}"}
    for s, word in zip(CONFIRMING, ("Two", "Three", "Four")):
        g = [c for c in conf if c.seed == s]
        nums[f"gpCorrectModel{word}"], nums[f"gpCasesModel{word}"] = sum(map(correct, g)), len(g)
    nums |= {"gpHarmful": sum(c.outcome in HARM for c in conf),
             "gpDidHarm": sum(c.outcome == "did harm" for c in conf),
             "gpTaughtWrong": sum(c.outcome == "taught wrong answers" for c in conf)}
    for k, word in (("basics", "Basics"), ("novelty", "Novelty"), ("agreement", "Agreement")):
        g = [c for c in conf if decided_by(c) == k]
        nums[f"gpBy{word}"], nums[f"gpBy{word}Correct"] = len(g), sum(map(correct, g))
    nums["gpByMinKept"] = sum(map(by_min_kept, conf))
    nob = sum(c.outcome in p9b.MATCH[p9b.verdict_of(c, 0.38, p9b.LEVEL_1, use_check_0=False)] for c in conf)
    nums |= {"gpCorrectNoBasics": nob, "gpCorrectNoBasicsPct": f"{100 * nob / n:.1f}"}

    comp, _ = baseline_comparison()
    rows = comp["confirming"]
    for key, word in (("ours", "Ours"), ("baseline", "Baseline"), ("favour", "Variant")):
        ok = sum(r["outcome"] in p9b.MATCH[r[key]] for r in rows)
        nums[f"gp{word}Correct"], nums[f"gp{word}CorrectPct"] = ok, f"{100 * ok / len(rows):.1f}"
        nums[f"gp{word}HarmFlagged"] = sum(r[key] == "does harm" for r in rows if r["outcome"] in HARM)
        weak = [r for r in rows if r["case"] == p9b.CASES[5]]
        nums[f"gp{word}WeakCorrect"] = sum(r["outcome"] in p9b.MATCH[r[key]] for r in weak)
    nums["gpWeakCases"] = len([r for r in rows if r["case"] == p9b.CASES[5]])
    cost = baseline_cost(CONFIRMING)
    tf = sum(k * f for _, k, _, f, _ in cost)
    nums["gpBaselineCostPct"] = f"{100 * sum(k * p for _, k, p, _, _ in cost) / tf:.0f}"
    nums["gpChecksCostPct"] = f"{100 * sum(k * c for _, k, _, _, c in cost) / tf:.0f}"
    # The same two shares over the addition rounds only (amendment 31): the paper reports addition alone.
    cost_add = [row for row in cost if row[0] in ADDITION_COST_LABELS]
    tfa = sum(k * f for _, k, _, f, _ in cost_add)
    nums["gpBaselineCostAdditionPct"] = f"{100 * sum(k * p for _, k, p, _, _ in cost_add) / tfa:.0f}"
    nums["gpChecksCostAdditionPct"] = f"{100 * sum(k * c for _, k, _, _, c in cost_add) / tfa:.0f}"

    m = margins(conf)
    nums |= {"gpNoveltyLevel": f"{100 * p9b.LEVEL_1:.0f}", "gpConfirmedLevel": "38",
             "gpNoveltyMarginLow": f"{100 * m['novelty_hi']:.1f}", "gpNoveltyMarginHigh": f"{100 * m['novelty_lo']:.1f}",
             "gpConfirmedMarginLow": f"{100 * m['confirmed_hi']:.1f}", "gpConfirmedHarmCases": m["n_harm"],
             "gpConfirmedMarginHigh": f"{100 * m['confirmed_lo']:.1f}", "gpConfirmedPaysCases": m["n_pays"]}

    rise = rise_groups(conf)
    nums["gpRiseNothingUnderTwo"] = sum(x < 2 for x in rise["Did little or nothing"])
    rr = p9b.round_rows(conf)
    nums |= {"gpRoundsCorrect": sum(o in p9b.MATCH[v] for _, _, v, o in rr), "gpRounds": len(rr)}
    # Added after the evaluation runs (amendment 28, part 4): the accuracy of always predicting
    # "does nothing", i.e. the share of the cases whose outcome is "did nothing".
    nums["gpDidNothingPct"] = f"{100 * sum(c.outcome == 'did nothing' for c in conf) / n:.1f}"
    # Internal consistency: the parts add up.
    assert nums["gpCorrect"] == nums["gpOursCorrect"] == sum(nums[f"gpCorrectModel{w}"] for w in ("Two", "Three", "Four"))
    assert nums["gpHarmful"] == nums["gpDidHarm"] + nums["gpTaughtWrong"]
    assert n == nums["gpByBasics"] + nums["gpByNovelty"] + nums["gpByAgreement"]
    return {k: str(v) for k, v in nums.items()}


# Which numbers were computed after the evaluation runs (said in numbers.tex, and in the text).
AFTER_EVALUATION = ("gpCorrectNoBasics", "gpVariant", "gpNoveltyMargin", "gpConfirmedMargin", "gpConfirmedHarmCases",
                    "gpConfirmedPaysCases", "gpRounds", "gpCtrlAfterSeven", "gpCtrlBeforeSeven", "gpWithheldSeven", "gpEdgesFive",
                    "gpKeptNoCarry", "gpCarryAfter", "gpNaturalCarry", "gpNovGain", "gpTaNoCarry", "gpTaAgreement",
                    "gpTaNoSelect", "gpTaWeak", "gpTaFull", "gpTaP", "gpSteer", "gpReh", "gpSched", "gpDidNothing",
                    "gpLrEdgesRandom")


def numbers_tex(nums: dict[str, str]) -> str:
    lines = [GENERATED_NOTE.rstrip(), "% Numbers cited in the text, so the prose cannot drift from the data.",
             "% Marked 'after': computed after the evaluation runs (not part of the frozen analysis)."]
    for k, v in nums.items():
        note = "  % after" if k.startswith(AFTER_EVALUATION) else ""
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}{note}")
    return "\n".join(lines) + "\n"


RISE_ROWS = (("Did little or nothing", {"did nothing"}), ("Taught wrong answers", {"taught wrong answers"}),
             ("Paid off", {"paid off"}))


def rise_groups(conf: list) -> dict[str, list[float]]:
    """New lengths (score before at most 50%): the rise over the round, in points, by outcome."""
    new = [c for c in conf if c.new_length and c.before <= 0.50]
    out = {label: [100 * (c.after - c.before) for c in new if c.outcome in members] for label, members in RISE_ROWS}
    if sum(map(len, out.values())) != len(new):
        raise ValueError("a new length has an outcome outside the rise table's rows")
    return out


def rise_tex(conf: list) -> str:
    def signed(x):
        return f"{x:.1f}".replace("-", "−")  # a typeset minus
    rows = [[label, str(len(v)), signed(min(v)), signed(statistics.median(v)), signed(max(v))]
            for label, v in rise_groups(conf).items()]
    return tex_table("rise", "Rise in accuracy over the round at new lengths (score before at most 50%), "
                     "evaluation runs, by outcome; points. Taught wrong answers is counted as harm.",
                     ["Outcome", "Cases", "Smallest rise", "Median", "Largest rise"], rows)


def outcome_counts(g: list) -> str:
    counts = {o: sum(c.outcome == o for c in g) for o in OUTCOME_ORDER}
    return ", ".join(f"{o} {k}" for o, k in counts.items() if k)


def cases_tex(conf: list) -> str:
    from gainpred import p9b

    rows = []
    for case in p9b.CASES:
        g = [c for c in conf if p9b.case_of(c) == case]
        if g:
            by = [k for k in ("basics", "novelty", "agreement") if any(decided_by(c) == k for c in g)]
            rows.append([case, str(len(g)), str(sum(map(correct, g))), outcome_counts(g), " / ".join(by)])
    return tex_table("p9b-cases", "Cases correctly predicted, by kind of case, in the evaluation runs. Decided "
                     "by: the check that settles the prediction (novelty includes the 5% minimum kept share). "
                     "Taught wrong answers is counted as harm.",
                     ["Kind of case", "Cases", "Correctly predicted", "Outcomes", "Decided by"], rows,
                     spec="lrrll", fit=True)


def weak_steps(run: str) -> str:
    m = re.search(r"-s(\d+)-", run)
    return f"{int(m.group(1)):,} updates" if m else ""


def easy_lowest(c) -> float:
    """Lowest easy-problem accuracy over sizes 3-7 of the model before the round (the basics
    check needs 98% at every size)."""
    acc = evals(c.run, "easy")[c.round - 1]
    return 100 * min(v for k, v in acc.items() if k != "all")


def weak_tex() -> str:
    from gainpred import p9b

    weak = sorted((c for c in evaluation_cases(False) if p9b.case_of(c) == p9b.CASES[5]),
                  key=lambda c: (c.seed, int(re.search(r"-s(\d+)-", c.run).group(1)), c.digits))
    rows = [[f"{c.seed} ({weak_steps(c.run)})", f"{easy_lowest(c):.1f}", str(c.digits), pct(c.before), pct(c.after),
             pct(c.best_after), "without an outcome" if c.outcome == "unclassified" else c.outcome, c.verdict,
             p9b.verdict_of(c, 0.38, p9b.LEVEL_1, use_check_0=False)] for c in weak]
    return tex_table("weak", f"The {len(rows)} cases from weak starting models (stopped early), evaluation runs, "
                     "including those without an outcome. Easy: lowest accuracy (%) over easy sizes 3-7 before the "
                     "round (the basics check needs 98% at every size). Before, after: accuracy (%) at that length; "
                     "best: the best teacher's after, same starting model and round.",
                     ["Starting model", "Easy", "Length", "Before", "After", "Best", "Outcome", "Prediction",
                      "Without basics"], rows, spec="lrrrrrlll", fit=True)


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def outcome_gap(c) -> str:
    """Which gap in the outcome rule (P9b) leaves a case without an outcome.

    Learned lengths, and new ones starting above 50%, have outcomes only for a change of
    at most 3 points either way or a fall of more than 3: a rise of more than 3 has none.
    New lengths starting at or below 50% have none for a rise of 10-20 points that is not
    30 points below the best teacher, or a rise of 20 or more that is 10-30 points below it.
    """
    if not c.new_length or c.before > 0.50:
        return "started above 50%, rose more than 3 points" if c.new_length else \
            "learned length, rose more than 3 points"
    if c.after - c.before < 0.20:
        return "rose 10-20 points, under 30 below the best"
    return "rose 20 points or more, 10-30 below the best"


def no_outcome_tex() -> str:
    cs = [c for c in evaluation_cases(False) if c.outcome == "unclassified"]
    rows = [[TEACHER_LABEL.get(c.teacher, c.teacher), f"{c.seed}" + (" (weak)" if "-weak-" in c.run else ""),
             str(c.round), str(c.digits), pct(c.before), pct(c.after), pct(c.check_1), c.verdict, outcome_gap(c)]
            for c in sorted(cs, key=lambda c: (c.teacher, c.seed, c.round, c.digits))]
    return tex_table("no-outcome", f"The {len(rows)} evaluation cases without an outcome, and the gap in the "
                     "outcome rule each falls in. Before, after: accuracy (%) at that length; novelty in %.",
                     ["Teacher", "Starting model", "Round", "Length", "Before", "After", "Novelty", "Prediction",
                      "Gap in the outcome rule"], rows, spec="llrrrrrll", fit=True)


def round_level_tex(conf: list) -> str:
    from gainpred import p9b

    rr = p9b.round_rows(conf)
    ok = sum(o in p9b.MATCH[v] for _, _, v, o in rr)
    pooled = {"paid off": "Paid off", "did nothing": "Did nothing", "did harm": "Did harm"}
    rows = [[label, *(str(sum(o == out and v == p for _, _, v, o in rr)) for p in PREDICTIONS)]
            for out, label in pooled.items()]
    return tex_table("round-level", f"Round-level score, defined after the evaluation runs: {ok} of {len(rr)} "
                     "rounds correctly predicted. A round's prediction is the prediction at its newest length; its "
                     "outcome is did harm if any length did harm or taught wrong answers, else paid off if any new "
                     "length paid off, else did nothing. Rows: outcome; columns: prediction.",
                     ["Outcome", "Pays off", "Does nothing", "Does harm"], rows)


# Shin et al.'s exact addition figures (arXiv 2611.00452v1): (mean over new lengths 8-23,
# accuracy at 23 digits), in %. Starting model and filtered final round: their Table 10;
# unfiltered final round: their appendix C.1 text. Their per-length results are heatmaps only.
SHIN_ADDITION = (("Starting model", "0.0", "0.0"),
                 ("Filtered composition, last round (8)", "93.7", "80.0"),
                 ("Unfiltered composition, last round (8)", "33.5", "11.0"))
REPRO_TEACHERS = (("nocarry", "No-carry rule"), ("noselect", "No selection"))


def reproduction_rows(events: list[dict]) -> list[list[str]]:
    """Our rows: (model, mean over 8-15 digits, accuracy at 15 digits), each as the mean of the
    three evaluation runs with the range in brackets. The 8-15 mean averages the per-length
    accuracies, as Shin et al. do ("rows with a size range average the accuracy per size")."""
    def cell(values: list[float]) -> str:
        values = [100 * v for v in values]
        return f"{statistics.mean(values):.1f} ({min(values):.1f}–{max(values):.1f})"

    def new_lengths(e):
        return statistics.mean(e["reproduction"][str(d)] for d in range(8, 16))

    def row(label, picked):
        assert len(picked) == 3, (label, len(picked))
        return [label, cell([new_lengths(e) for e in picked]), cell([e["reproduction"]["15"] for e in picked])]

    rows = [row("Starting model", [e for e in events if e["round"] == 0])]
    for key, label in REPRO_TEACHERS:
        for k in range(1, 5):
            rows.append(row(f"{label}, round {k}", [e for e in events if e["round"] == k and f"-{key}-" in e["run"]]))
    return rows


def reproduction_tex() -> str | None:
    """Appendix: the reproduction test set, set beside Shin et al.'s reported figures.
    None until ``python -m gainpred.reproduction`` has written results/reproduction.jsonl."""
    from gainpred.reproduction import RESULTS

    if not RESULTS.exists():
        return None
    events = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    ours = reproduction_rows(events)
    rows = [["This paper (8–15 digits; 15 digits)", "", ""], *ours,
            ["Shin et al. (8–23 digits; 23 digits)", "", ""], *([a, b, c] for a, b, c in SHIN_ADDITION)]
    return tex_table(
        "reproduction",
        "Accuracy (%) on the reproduction test set: 100 problems per length, 3 to 15 digits, greedy answers, "
        "scored after all experiments on the saved models of the evaluation runs (mean of three runs, range in "
        "brackets). Shin et al.'s figures are those they report (their Table 10 and appendix C.1); they train "
        "8 rounds up to 23 digits and test up to 23 digits, so their columns cover 8-23 and 23 digits, ours "
        "8-15 and 15 digits. The no-carry rule corresponds to their filtered composition, no selection to "
        "their unfiltered composition.",
        ["Model", "Mean over new lengths", "Longest length"], rows, rule_before={len(ours) + 1})


# ---------------------------------------------------------------- section 5: experiment 2 table

EXP2_ROWS = (("Before the round", "runs-join_at_the_cut-seed{s}-u30000", 0),
             ("Pick the longer", "runs-pick_the_longer-seed{s}-u30000", 1),
             ("Edges-differ rule (Shin et al.)", "runs-edges_differ-seed{s}-u30000", 1),
             ("Join at the split (ours)", "runs-join_at_the_cut-seed{s}-u30000", 1))


def mean_range_or_zero(xs: list[float]) -> str:
    """``mean_range``, but a column of zeros prints "0.0" (no "(0.0–0.0)")."""
    return "0.0" if f"{min(xs):.1f}" == f"{max(xs):.1f}" == "0.0" else mean_range(xs)


def exp2_paper_rows() -> tuple[list[list[str]], set[tuple[int, int]]]:
    """(rows, bold cells): all groups and crossing runs of 5-8, evaluation runs. Bold: the highest
    mean in each column, as printed (ties all bold)."""
    values = []
    for label, run, rnd in EXP2_ROWS:
        evs = [evals(run.format(s=s), "study")[rnd] for s in CONFIRMING]
        values.append((label, [[100 * e["all"] for e in evs]] + [[joined(e, k) for e in evs] for k in (5, 6, 7, 8)]))
    rows = [[label, *(mean_range_or_zero(c) for c in cols)] for label, cols in values]
    bold = set()
    for j in range(5):
        means = [f"{statistics.mean(cols[j]):.1f}" for _, cols in values]
        best = max(means, key=float)
        bold |= {(i, j + 1) for i, m in enumerate(means) if m == best}
    return rows, bold


def exp2_tex() -> str:
    rows, bold = exp2_paper_rows()
    groups = all_groups()
    return tex_table(
        "exp2", f"Longest run, one round: accuracy (%) on the test problems, all groups and crossing runs of 5-8; "
        f"mean over the evaluation runs (lowest-highest). All pools {NUMBER_WORDS[len(groups)]} groups of "
        f"1,000 problems, {NUMBER_WORDS[sum(g != 'random_join' for g in groups)]} of which contain a crossing run. "
        "Best in each column in bold.",
        ["Teacher", "All", "Crossing run 5", "Crossing run 6", "Crossing run 7", "Crossing run 8"], rows, fit=True,
        rule_before={3}, bold=bold)


# ---------------------------------------------------------------- section 5: slim withheld table and numbers

NUMBER_WORDS = {n: w for n, w in enumerate(("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                                            "nine", "ten", "eleven", "twelve"))}


def all_groups() -> list[str]:
    """The longest-run test groups the logged "all" pools (every group of test_study, in file order)."""
    seen = []
    for line in (Path("data/longest_run/test_study.jsonl")).read_text().splitlines():
        g = json.loads(line)["group"]
        if g not in seen:
            seen.append(g)
    return seen


def span(values: list[float]) -> str:
    return f"{min(values):.1f}–{max(values):.1f}"


def withheld_slim_rows() -> list[list[str]]:
    """The withheld version, evaluation runs, before -> after (lowest-highest over starting models).

    Crossing runs of 6 by left part come from results/withheld_split.json (every test problem with
    a crossing run of 6, both groupings; scripts/withheld_split.py); crossing runs of 7-8 from the
    round logs (the two groups pooled, 1,000 problems each)."""
    split = json.loads((RESULTS_DIR / "withheld_split.json").read_text())
    rows = []
    for label, key in (("Shown combinations: crossing run of 6, left part 1-2", "6|1-2"),
                       ("Withheld combinations: crossing run of 6, left part 3-4", "6|3-4")):
        before = [100 * split[f"seed{s} before"][key][0] for s in CONFIRMING]
        after = [100 * split[f"seed{s} withheld"][key][0] for s in CONFIRMING]
        n = {split[f"seed{s} {w}"][key][1] for s in CONFIRMING for w in ("before", "withheld")}
        assert len(n) == 1, (key, n)
        rows.append([label, f"{n.pop():,}", f"{span(before)} → {span(after)}"])
    evs = [evals(f"runs-join_at_the_cut-withheld-seed{s}-u30000", "study") for s in CONFIRMING]
    before = [(joined(e[0], 7) + joined(e[0], 8)) / 2 for e in evs]
    after = [(joined(e[1], 7) + joined(e[1], 8)) / 2 for e in evs]
    rows.append(["Answers never shown: crossing runs of 7-8", "2,000", f"{span(before)} → {span(after)}"])
    return rows


def withheld_slim_tex() -> str:
    return tex_table(
        "withheld", "The withheld version (join at the split, trained without crossing runs whose left part is "
        "3-4), evaluation runs: accuracy (%) before → after the round, lowest-highest over the three starting "
        "models. Left part: how many symbols of the crossing run lie in the left piece. The teacher shows "
        "crossing runs of 6 with left part 1-2, never with left part 3-4, and never an answer of 7 or 8 on "
        "long strings.", ["Test problems", "Problems", "Before → after"], withheld_slim_rows())


def natural_carry(d: int, n: int = 20_000) -> float:
    """Share (%) of freshly drawn hard d-digit problems that carry out of the split-1 place(s) (mean
    over its boundaries, as in ``carry_shares``): the rate before any selection. Seeded, so fixed."""
    import random

    from gainpred.data import draw_hard

    places = {d - 1 - p for p in cut_positions(ADDITION_CUTS[d][0])}
    rng, hits = random.Random(1_000 + d), 0
    for _ in range(n):
        p = draw_hard(rng, d)
        a, b, carry = p.a, p.b, 0
        for place in range(d):
            carry = 1 if a % 10 + b % 10 + carry >= 10 else 0
            hits += carry and place in places
            a, b = a // 10, b // 10
    return 100 * hits / n / len(places)


def section5_numbers() -> dict[str, str]:
    """Every number section 5's text cites (evaluation runs), as LaTeX macro name -> value."""
    nums = {}
    # Agreement minus no-carry after round 4: every gap is a shortfall, so the macros give its size.
    for group, suffix in (("mean", ""), ("carry", "Carry")):
        pm = pass_mark(CONFIRMING, group)
        assert all(v[0] <= 0 and v[1] <= 0 for v in pm.values()), f"agreement ahead somewhere ({group})"
        d_mean = min(pm, key=lambda d: pm[d][0])
        d_worst = min(pm, key=lambda d: pm[d][1])
        nums |= {f"gpGapMean{suffix}": f"{-pm[d_mean][0]:.1f}", f"gpGapMean{suffix}Digits": str(d_mean),
                 f"gpGapWorst{suffix}": f"{-pm[d_worst][1]:.1f}", f"gpGapWorst{suffix}Digits": str(d_worst),
                 f"gpGapWorst{suffix}Model": str(pm[d_worst][2])}
    # Longest run, one round: all groups and crossing runs of 6-8, mean over the evaluation runs.
    for name, run, rnd in (("Before", "runs-join_at_the_cut-seed{s}-u30000", 0),
                           ("Join", "runs-join_at_the_cut-seed{s}-u30000", 1),
                           ("Edges", "runs-edges_differ-seed{s}-u30000", 1),
                           ("Longer", "runs-pick_the_longer-seed{s}-u30000", 1)):
        evs = [evals(run.format(s=s), "study")[rnd] for s in CONFIRMING]
        nums[f"gpLr{name}All"] = f"{statistics.mean(100 * e['all'] for e in evs):.1f}"
        nums[f"gpLr{name}Long"] = f"{statistics.mean(statistics.mean(joined(e, k) for k in (6, 7, 8)) for e in evs):.1f}"
    # The edges-differ rule's starting models (round 0 of its runs), crossing runs of 5 (section 5's
    # "fell from ... to \gpEdgesFiveCorrect").
    nums["gpLrEdgesBeforeFive"] = (
        f"{statistics.mean(joined(evals(f'runs-edges_differ-seed{s}-u30000', 'study')[0], 5) for s in CONFIRMING):.1f}")
    # Added after the evaluation runs (amendment 28, part 4): students taught with the edges-differ
    # rule (after its one round), on the test group of randomly joined strings: lowest and highest
    # over the starting models of the evaluation runs.
    rj = [100 * evals(f"runs-edges_differ-seed{s}-u30000", "study")[1]["random_join"] for s in CONFIRMING]
    nums["gpLrEdgesRandomJoinLow"], nums["gpLrEdgesRandomJoinHigh"] = f"{min(rj):.1f}", f"{max(rj):.1f}"
    # Filter gap (defined after the evaluation runs): kept share and the carry group after.
    for arm, name in FILTER_ARMS:
        rows = filter_gap_rows(arm, CONFIRMING)
        after = [r["after"] for r in rows if r["after"] is not None]
        key = "NoCarry" if arm == "nocarry" else "Agreement"
        nums[f"gpCarryAfter{key}Mean"] = f"{statistics.mean(after):.1f}"
        nums[f"gpCarryAfter{key}Lowest"] = f"{min(after):.1f}"
    kept_nc = [r["kept_carry"] for r in filter_gap_rows("nocarry", CONFIRMING) if r["kept_carry"] is not None]
    # Not exactly 0: the rule reads the model's piece answers, and a wrong piece sum can let a carry
    # through (1-4 answers in 10,000 in a few runs). "0.0" holds to one decimal; the maximum is given too.
    assert f"{max(kept_nc):.1f}" == "0.0", "the no-carry rule kept a visible share of split-1 carries"
    nums["gpKeptNoCarry"] = "0.0"
    nums["gpKeptNoCarryMax"] = f"{max(kept_nc):.2f}"
    ag = carries_per_place("agreement", CONFIRMING)
    means = [statistics.mean(ag[d]["cut 1"]) for d in LENGTHS if ag[d]["cut 1"]]
    nums["gpKeptAgreementLow"], nums["gpKeptAgreementHigh"] = f"{min(means):.1f}", f"{max(means):.1f}"
    natural = [natural_carry(d) for d in LENGTHS]
    nums["gpNaturalCarryLow"], nums["gpNaturalCarryHigh"] = f"{min(natural):.1f}", f"{max(natural):.1f}"
    # Build time, all four rounds of every evaluation run: agreement rule over no-carry rule.
    def build_seconds(arm):
        return sum(round_cost(addition_run(arm, s, r), r, True, 0.0)["build_s"] for s in CONFIRMING for r in range(1, 5))
    nums["gpBuildRatio"] = f"{build_seconds('agreement') / build_seconds('nocarry'):.1f}"
    return nums


# ---------------------------------------------------------------- control-arm answers (amendment 23)
#
# Defined after all experiments. Reads results/control_answers.jsonl (gainpred.control_answers);
# nothing here exists until that file does.

CONTROL_COLUMNS = (("Correct", "correct"), ("Right symbol, length 6", "right symbol, length 6"),
                   ("Right symbol, other length", "right symbol, other wrong length"),
                   ("Anything else", "anything else"), ("Pick the longer", "pick the longer"))


def control_pooled(rows: list[dict], role: str, crossing: int, seeds, key: str) -> tuple[float, float, float]:
    """(pooled %, lowest %, highest %) of one category over the given starting models."""
    from gainpred.control_answers import tabulate

    per = [tabulate([r for r in rows if r["role"] == role and r["seed"] == s and r["crossing"] == crossing])
           for s in seeds]
    pooled = 100 * sum(t[key] for t in per) / sum(t["n"] for t in per)
    shares = [100 * t[key] / t["n"] for t in per]
    return pooled, min(shares), max(shares)


def control_md() -> str | None:
    from gainpred.control_answers import (CATEGORIES, EVALUATION_SEEDS, RESULTS, ROLES, SAVED_LENGTHS, load_rows,
                                          prediction, tabulate)

    if not RESULTS.exists():
        return None
    rows = load_rows()
    out = ["# What the control-arm students answer on long crossing runs", "",
           "Generated by `uv run python -m gainpred.writeup`; do not edit by hand. Protocol amendment 23, "
           "**defined after all experiments**; inference only on saved models (`gainpred.control_answers`). "
           "Every answer is in `results/control_answers.jsonl`.", "",
           "Models: " + "; ".join(f"**{k}** {v.lower()}" for k, v in ROLES.items()) + ". Categories do not overlap, "
           "except *pick the longer* (the answer equals the longer piece's own longest run), which is counted "
           "separately and can overlap the others. Counts out of 1,000 problems per model and crossing-run length.", ""]
    held = []
    for s in EVALUATION_SEEDS:
        six, wrong, ok = prediction(rows, (s,))
        held.append(f"starting model {s}: {six} of {wrong} wrong answers ({'held' if ok else 'did not hold'})")
    six, wrong, ok = prediction(rows, EVALUATION_SEEDS)
    out += [f"**Prediction** (amendment 23: after the round, at a crossing run of 7, at least half of the "
            f"control-arm students' wrong answers give the right symbol with length 6): "
            f"**{'held' if ok else 'did not hold'}** on the evaluation runs pooled, {six} of {wrong} "
            f"({100 * six / wrong:.1f}%); " + "; ".join(held) + ".", ""]
    cal = [prediction(rows, (s,)) for s in (0, 1)]
    out += ["Calibration runs: " + "; ".join(f"starting model {s}: {a} of {w} ({'held' if ok else 'did not hold'})"
                                           for s, (a, w, ok) in zip((0, 1), cal)) + ".", ""]
    for title, seeds in (("Evaluation runs (starting models 2-4)", EVALUATION_SEEDS),
                         ("Calibration runs (starting models 0-1)", (0, 1))):
        out += [f"## {title}", ""]
        for n in SAVED_LENGTHS:
            out += [f"### Crossing run of {n}", "",
                    "| Model | Starting model | Correct | Right symbol, length 6 | Right symbol, other wrong length "
                    "(lengths) | Anything else | Pick the longer |", "| --- | --- | ---: | ---: | --- | ---: | ---: |"]
            for role in ROLES:
                for s in seeds:
                    t = tabulate([r for r in rows if r["role"] == role and r["seed"] == s and r["crossing"] == n])
                    dist = ", ".join(f"{k}: {v}" for k, v in t["lengths"].items())
                    out.append(f"| {role} | {s} | {t['correct']} | {t['right symbol, length 6']} | "
                               f"{t['right symbol, other wrong length']}{f' ({dist})' if dist else ''} | "
                               f"{t['anything else']} | {t['pick the longer']} |")
            out.append("")
        out += ["### After the round, crossing runs of 7 and 8, by left part", "",
                "The left part is how many symbols of the crossing run lie in the left piece (1-4 by "
                "construction, so a crossing run of 7 has left part 3-4 and one of 8 has left part 4; "
                "the 1-2 and 5-or-more rows are empty).", "",
                "| Model | Crossing run | Left part | Problems | " + " | ".join(c for c, _ in CONTROL_COLUMNS) + " |",
                "| --- | --- | --- | ---: | " + " | ".join("---:" for _ in CONTROL_COLUMNS) + " |"]
        for role in ("b", "c", "d"):
            for n in (7, 8):
                for label, lo, hi in (("1-2", 1, 2), ("3-4", 3, 4), ("5 or more", 5, 99)):
                    t = tabulate([r for r in rows if r["role"] == role and r["seed"] in seeds and r["crossing"] == n
                                  and lo <= r["left_part"] <= hi])
                    cells = [str(t[k]) for _, k in CONTROL_COLUMNS] if t["n"] else ["" for _ in CONTROL_COLUMNS]
                    out.append(f"| {role} | {n} | {label} | {t['n']} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["## Most common wrong answers, evaluation runs pooled, crossing run of 7", "",
            "| Model | Wrong | Five most common wrong answers (count) |", "| --- | ---: | --- |"]
    from collections import Counter
    for role in ROLES:
        wrong = [r for r in rows if r["role"] == role and r["seed"] in EVALUATION_SEEDS and r["crossing"] == 7
                 and r["answer"] != r["correct"]]
        top = Counter(("unreadable" if r["answer"].startswith("?") else
                       ("right symbol" if r["answer"][0] == r["correct"][0] else "other symbol")
                       + ", length " + r["answer"].split("|")[1]) for r in wrong).most_common(5)
        out.append(f"| {role} | {len(wrong)} | " + "; ".join(f"{k} ({v})" for k, v in top) + " |")
    return "\n".join(out) + "\n"


def control_tex() -> str | None:
    from gainpred.control_answers import EVALUATION_SEEDS, RESULTS, ROLES, load_rows

    if not RESULTS.exists():
        return None
    rows = load_rows()
    labels = {"a": "Control arm, before the round", "b": "Control arm, after the round",
              "c": "Withheld version, after the round", "d": "Edges-differ rule, after the round"}
    body = []
    for role, n in (("a", 7), ("b", 7), ("c", 7), ("d", 7), ("d", 5)):
        cells = []
        for _, key in CONTROL_COLUMNS:
            pooled, lo, hi = control_pooled(rows, role, n, EVALUATION_SEEDS, key)
            cells.append(f"{pooled:.1f} ({lo:.1f}–{hi:.1f})")
        body.append([f"{labels[role]}, crossing run of {n}", *cells])
    assert set(ROLES) == set(labels)
    return tex_table(
        "control-answers",
        "Answers on long crossing runs, computed after all experiments: share of the 1,000 test problems per "
        "model (%), evaluation runs pooled, range over the three starting models in brackets. The control arm's "
        "starting training includes short strings with a run of 7-8 inside one piece; the withheld round's "
        "teacher never shows an answer above 6 on long strings. Columns other than the last do not overlap; "
        "pick the longer is the share equal to the longer piece's own longest run and can overlap them.",
        ["Model", "Correct", "Right symbol, length 6", "Right symbol, other length", "Anything else",
         "Pick the longer"], body, fit=True)


def control_numbers() -> dict[str, str]:
    """Macros for the text: pooled evaluation-run shares (%) for b and c at 7, d at 5."""
    from gainpred.control_answers import EVALUATION_SEEDS, RESULTS, load_rows

    if not RESULTS.exists():
        return {}
    rows = load_rows()
    out = {}
    out["gpCtrlBeforeSevenCorrect"] = f"{control_pooled(rows, 'a', 7, EVALUATION_SEEDS, 'correct')[0]:.1f}"
    for role, n, name in (("b", 7, "CtrlAfterSeven"), ("c", 7, "WithheldSeven"), ("d", 5, "EdgesFive")):
        for key, suffix in (("correct", "Correct"), ("right symbol, length 6", "LengthSix"),
                            ("pick the longer", "PickLonger")):
            out[f"gp{name}{suffix}"] = f"{control_pooled(rows, role, n, EVALUATION_SEEDS, key)[0]:.1f}"
    return out


PREDICTIONS = ("pays off", "does nothing", "does harm")
OUTCOME_ROWS = (("Paid off", {"paid off"}), ("Did nothing", {"did nothing"}),
                ("Did harm", {"did harm", "taught wrong answers"}))


def p9b_tables(conf: list) -> tuple[str, str]:
    """Section 4: (summary table, confusion table) over the evaluation cases ``conf``.

    The summary compares the prediction rule with always giving one prediction. The
    confusion table counts cases by outcome (rows) and prediction (columns); "Did harm"
    pools "did harm" and "taught wrong answers", exactly as ``p9b.MATCH`` does. The two
    tables must agree, or nothing is written: the grid sums to the number of cases, its
    diagonal is the rule's count, and each row is the matching "Always ..." count. The
    caption's second sentence is added only when every error lies in the "Does harm"
    column.
    """
    from gainpred import p9b

    n = len(conf)
    hits = sum(c.outcome in p9b.MATCH[c.verdict] for c in conf)
    always = {v: sum(c.outcome in p9b.MATCH[v] for c in conf) for v in PREDICTIONS}
    rows = [["Our prediction rule", f"{hits}/{n}", f"{100 * hits / n:.1f}"]]
    rows += [[f"Always “{v}”", f"{always[v]}/{n}", f"{100 * always[v] / n:.1f}"]
             for v in ("does nothing", "does harm", "pays off")]
    summary = tex_table("p9b-summary", "Cases correctly predicted in the evaluation runs, for our prediction rule "
                        "and for always giving the same prediction.", ["Rule", "Cases correctly predicted", "%"], rows)

    grid = [[sum(c.outcome in members and c.verdict == v for c in conf) for v in PREDICTIONS]
            for _, members in OUTCOME_ROWS]
    if sum(map(sum, grid)) != n:
        raise ValueError(f"confusion grid sums to {sum(map(sum, grid))}, not {n} cases")
    if sum(grid[i][i] for i in range(3)) != hits:
        raise ValueError(f"confusion diagonal {sum(grid[i][i] for i in range(3))} != the rule's {hits}")
    for i, v in enumerate(PREDICTIONS):
        if sum(grid[i]) != always[v]:
            raise ValueError(f"row {OUTCOME_ROWS[i][0]} sums to {sum(grid[i])}, not always-{v}'s {always[v]}")
    errors = [(i, j) for i in range(3) for j in range(3) if i != j and grid[i][j]]
    caption = f"Predictions before training against outcomes after training, for the {n} evaluation cases."
    if all(j == PREDICTIONS.index("does harm") for _, j in errors):
        caption += " Every error is a prediction of harm that did not occur."
    lines = [GENERATED_NOTE.rstrip(), r"\begin{table}[htbp]", r"\centering", rf"\caption{{{tex_escape(caption)}}}",
             r"\label{tab:p9b-confusion}", r"\small", r"\begin{tabular}{lrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{" + tex_escape("Prediction") + r"} \\", r"\cmidrule(l){2-4}",
             " & ".join(tex_escape(h) for h in ("Outcome", "Pays off", "Does nothing", "Does harm")) + r" \\",
             r"\midrule"]
    lines += [" & ".join(tex_escape(c) for c in (label, *map(str, row))) + r" \\"
              for (label, _), row in zip(OUTCOME_ROWS, grid)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return summary, "\n".join(lines) + "\n"


# ---------------------------------------------------------------- amendment 24: novelty and teacher accuracy
#
# Both analyses were defined after all experiments (protocol amendment 24); nothing here enters a
# prediction. Part A reads results/p9b_cells.csv only. Part B reads results/teacher_accuracy.csv
# (gainpred.teacher_accuracy), which holds, per case, how accurate the teacher's kept answers were
# against the true sum.

KIND_ORDER = ("No-carry rule", "Agreement rule", "No selection", "No selection after the no-carry rule",
              "Sample and vote", "Weak starting models")
# Colour AND marker per kind, so the figure reads in greyscale; the weak starting models are hollow.
KIND_STYLE = {"No-carry rule": (COLOURS[0], "o"), "Agreement rule": (COLOURS[1], "s"),
              "No selection": (COLOURS[2], "^"), "No selection after the no-carry rule": ("#6250d6", "v"),
              "Sample and vote": (COLOURS[3], "D"), "Weak starting models": ("#0b0b0b", "X")}
SOUND = ("No-carry rule", "Agreement rule")


def novelty_cases() -> list[dict]:
    """All 270 evaluation cases (with or without an outcome): kind, novelty, change and error before, in %.

    Novelty is check 1, measured on the teacher's kept answers before the round; change is accuracy
    after minus before on the test problems of that length, in points; error before = 100 - before.
    """
    from gainpred.teacher_accuracy import kind
    out = []
    for r in csv.DictReader((RESULTS_DIR / "p9b_cells.csv").open()):
        if int(r["seed"]) not in CONFIRMING:
            continue
        before, after = 100 * float(r["before"]), 100 * float(r["after"])
        out.append({"run": r["run"], "kind": kind(r["run"], r["teacher"]), "seed": int(r["seed"]),
                    "round": int(r["round"]), "digits": int(r["digits"]), "novelty": 100 * float(r["check_1"]),
                    "before": before, "after": after, "change": after - before, "error": 100 - before})
    return out


def within_bound(c: dict, slack: float = 3.0) -> bool:
    """|change| <= novelty + 3 points: a teacher that repeats the student cannot move it much."""
    return abs(c["change"]) <= c["novelty"] + slack


def sound_between(cases: list[dict]) -> list[dict]:
    """Sound teachers (no-carry and agreement rules, fully trained models) with novelty strictly in 5-95%."""
    return sorted((c for c in cases if c["kind"] in SOUND and 5 < c["novelty"] < 95),
                  key=lambda c: (c["novelty"], c["run"], c["round"], c["digits"]))


def novelty_numbers() -> dict[str, str]:
    """Part A macros (all 'after'); tests/test_amendment24.py checks each against the expected value."""
    cases = novelty_cases()
    outside = [c for c in cases if not within_bound(c)]
    exc = outside[0] if len(outside) == 1 else None
    sound = [c for c in cases if c["kind"] in SOUND]
    err_gap = [abs(c["novelty"] - c["error"]) for c in sound]
    chg_gap = [abs(c["change"] - c["novelty"]) for c in sound]
    runf = [c for c in cases if c["kind"] == "No selection after the no-carry rule" and c["before"] >= 98]
    vote = [c for c in cases if c["kind"] == "Sample and vote"]
    weak = [c for c in cases if c["kind"] == "Weak starting models"]
    out = {"gpNovGainCases": str(len(cases)), "gpNovGainWithin": str(len(cases) - len(outside))}
    if exc:
        out |= {"gpNovGainExcModel": str(exc["seed"]), "gpNovGainExcRound": str(exc["round"]),
                "gpNovGainExcDigits": str(exc["digits"]), "gpNovGainExcBefore": f"{exc['before']:.1f}",
                "gpNovGainExcAfter": f"{exc['after']:.1f}", "gpNovGainExcNovelty": f"{exc['novelty']:.1f}"}
    out |= {"gpNovGainSound": str(len(sound)),
            "gpNovGainSoundError": str(sum(g <= 5 for g in err_gap)), "gpNovGainSoundErrorGap": f"{max(err_gap):.1f}",
            "gpNovGainSoundChange": str(sum(g <= 5 for g in chg_gap)), "gpNovGainSoundChangeGap": f"{max(chg_gap):.1f}",
            "gpNovGainSoundBetween": str(len(sound_between(cases))),
            "gpNovGainRunfCases": str(len(runf)), "gpNovGainRunfLow": f"{min(c['novelty'] for c in runf):.1f}",
            "gpNovGainRunfHigh": f"{max(c['novelty'] for c in runf):.1f}",
            "gpNovGainVote": str(len(vote)), "gpNovGainVoteMax": f"{max(abs(c['change']) for c in vote):.1f}",
            "gpNovGainWeak": str(len(weak)), "gpNovGainWeakLow": f"{min(c['change'] for c in weak):.1f}",
            "gpNovGainWeakHigh": f"{max(c['change'] for c in weak):.1f}"}
    return out


def novelty_gain_tex() -> str:
    rows = [[c["kind"], str(c["seed"]), str(c["round"]), str(c["digits"]), f"{c['error']:.1f}",
             f"{c['novelty']:.1f}", fmt(c["change"], True)] for c in sound_between(novelty_cases())]
    return tex_table(
        "novelty-gain", "The evaluation cases of the no-carry and agreement rules (fully trained starting models) "
        "with novelty strictly between 5% and 95%, sorted by novelty; computed after the evaluation runs. Student "
        "error before: 100 minus accuracy before the round on the test problems of that length (%). Novelty: on "
        "the teacher's kept answers before the round (%). Change: accuracy after minus before (points).",
        ["Teacher", "Starting model", "Round", "Digits", "Student error before", "Novelty", "Change"], rows)


def novelty_gain_figure(plt) -> None:
    cases = novelty_cases()
    fig, ax = plt.subplots(figsize=(5.5, 3.7))  # the width of the paper's text block: included unscaled
    xs = [0, 100]
    # The axes cover every case: the y-axis is symmetric, so both bound lines run the full width.
    lim = max(105.0, max(abs(c["change"]) for c in cases) + 5)
    ax.fill_between(xs, [0, 100], [lim, lim], color="#f1f0ec", zorder=0, lw=0)
    ax.fill_between(xs, [-lim, -lim], [0, -100], color="#f1f0ec", zorder=0, lw=0)
    ax.plot(xs, xs, color="#52514e", lw=1, ls="--", zorder=1)
    ax.plot(xs, [0, -100], color="#52514e", lw=1, ls="--", zorder=1)
    ax.axhline(0, color="#c3c2be", lw=0.6, zorder=1)
    for k in KIND_ORDER:
        colour, marker = KIND_STYLE[k]
        g = [c for c in cases if c["kind"] == k]
        hollow = k == "Weak starting models"
        ax.scatter([c["novelty"] for c in g], [c["change"] for c in g], s=22, marker=marker, zorder=3,
                   facecolors="none" if hollow else colour, edgecolors=colour if hollow else "white",
                   linewidths=1.0 if hollow else 0.4, label=f"{k} ({len(g)})")
    for c in (c for c in cases if not within_bound(c)):
        ax.scatter([c["novelty"]], [c["change"]], s=200, facecolors="none", edgecolors="#0b0b0b", lw=1.0, zorder=4)
        ax.annotate(f"No-carry rule, starting model {c['seed']},\nround {c['round']}, {c['digits']} digits",
                    (c["novelty"], c["change"]), xytext=(c["novelty"] + 6, c["change"] - 4), fontsize=8,
                    va="top", arrowprops={"arrowstyle": "-", "color": "#52514e", "lw": 0.6})
    ax.set_xlim(-2, 102); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Novelty (%)"); ax.set_ylabel("Change in accuracy (points)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "novelty_gain.png", dpi=200)
    fig.savefig(PAPER / "figure_novelty_gain.pdf"); plt.close(fig)


NOVELTY_GAIN_CAPTION = (
    "Change in accuracy against novelty, one point per evaluation case (all 270, with or without an outcome); "
    "computed after the evaluation runs. Novelty is measured on the teacher's kept answers before the round; "
    "accuracy on the test problems of that length, after the round minus before (points). Dashed: change = "
    "novelty and change = minus novelty, the bound a teacher's novelty sets on how far it can move the student; "
    "shaded: outside it. Circled: the one case more than 3 points outside.")


# Part B: teacher accuracy (one inference run; results/teacher_accuracy.csv).

TA_KINDS = ("No-carry rule", "Agreement rule", "No selection", "No selection after the no-carry rule",
            "Weak starting models")
TA_COLUMNS = (("teacher_accuracy", "Teacher accuracy"), ("confirmed_share", "Confirmed share"),
              ("wrong_confirmed", "Wrong but confirmed"), ("accuracy_confirmed", "Accuracy among confirmed"),
              ("accuracy_unconfirmed", "Accuracy among unconfirmed"), ("student_accuracy", "Student accuracy"))
TA_MACRO = {"No-carry rule": "NoCarry", "Agreement rule": "Agreement", "No selection": "NoSelect",
            "No selection after the no-carry rule": "NoSelectAfter", "Weak starting models": "Weak"}
TA_QUANTITY = {"teacher_accuracy": "Acc", "confirmed_share": "Confirmed", "wrong_confirmed": "WrongConfirmed",
               "accuracy_confirmed": "AccConfirmed"}


def ta_rows() -> list[dict] | None:
    from gainpred.teacher_accuracy import CSV, load_rows
    return load_rows() if CSV.exists() else None


def ta_range(rows: list[dict], key: str, decimals: int = 1) -> tuple[str, str, int]:
    """(lowest, highest, cases with a value) of a share over cases, in %."""
    v = [100 * r[key] for r in rows if r[key] is not None]
    return (f"{min(v):.{decimals}f}", f"{max(v):.{decimals}f}", len(v)) if v else ("--", "--", 0)


def ta_cell(rows, key) -> str:
    lo, hi, _ = ta_range(rows, key)
    return lo if lo == hi else f"{lo}–{hi}"


def teacher_accuracy_md() -> str | None:
    from gainpred.teacher_accuracy import predictions
    rows = ta_rows()
    if rows is None:
        return None
    lines = ["# How accurate each teacher's kept answers are (amendment 24, Part B)", "",
             "Generated by `uv run python -m gainpred.writeup`; do not edit by hand. Defined after all experiments; "
             "inference only (`gainpred.teacher_accuracy`). Evaluation runs (starting models 2-4); every case of the "
             "teachers that split. Each case's check sample (2,000 problems) was redrawn from the model before the "
             "round; kept count, novelty and confirmed share reproduce the logged values exactly in every case. "
             "Shares are of the case's kept answers (%), scored against the true sum; range over cases.", ""]
    header = ["Teacher", "Cases", *[label for _, label in TA_COLUMNS]]
    for title, kinds in (("Fully trained starting models", TA_KINDS[:4]), ("Weak starting models (reported separately)",
                                                                           TA_KINDS[4:])):
        lines += [f"## {title}", ""]
        lines += table(header, [[k, str(len(g := [r for r in rows if r["kind"] == k])),
                                 *[ta_cell(g, key) for key, _ in TA_COLUMNS]] for k in kinds])
        lines += [""]
    lines += ["Accuracy among unconfirmed: over cases with at least one unconfirmed kept answer. Student accuracy: the "
              "student's greedy direct answer on the same kept problems.", "",
              "## Predictions (recorded before any answer was computed)", "",
              "P1: in every case from a fully trained starting model, wrong-but-confirmed is at most 2% of kept "
              "answers. P2: in every case from a fully trained starting model, accuracy among confirmed answers is "
              "at least 98%.", ""]
    preds = predictions(rows)
    prow = []
    for label, p in preds.items():
        full = [r for r in rows if r["kind"] != "Weak starting models" and (label == "pooled" or f"seed {r['seed']}" == label)]
        prow.append([label, str(p["cases"]), "held" if p["p1"] else f"did not hold ({len(p['p1_fail'])} cases)",
                     ta_range(full, "wrong_confirmed", 2)[1],
                     "held" if p["p2"] else f"did not hold ({len(p['p2_fail'])} cases)",
                     ta_range(full, "accuracy_confirmed", 2)[0]])
    lines += table(["Starting model", "Cases", "P1", "Largest wrong but confirmed", "P2", "Lowest accuracy among confirmed"],
                   prow)
    fails = {id(r): r for p in preds.values() for r in p["p1_fail"] + p["p2_fail"]}
    if fails:
        lines += ["", "Cases where a prediction failed:", ""]
        lines += table(["Run", "Round", "Digits", "Kept", "Wrong but confirmed", "Accuracy among confirmed"],
                       [[r["run"], str(r["round"]), str(r["digits"]), str(r["kept"]), pct(r["wrong_confirmed"]),
                         pct(r["accuracy_confirmed"])] for r in fails.values()])
    weak = [r for r in rows if r["kind"] == "Weak starting models"]
    lines += ["", f"Weak starting models (not part of P1 or P2): wrong but confirmed {ta_cell(weak, 'wrong_confirmed')}%, "
              f"accuracy among confirmed {ta_cell(weak, 'accuracy_confirmed')}%.", "",
              "Consistency check (true by logic): teacher accuracy >= confirmed share - wrong but confirmed, in "
              f"every case: {'yes' if all(r['teacher_accuracy'] is None or r['teacher_accuracy'] >= r['confirmed_share'] - r['wrong_confirmed'] - 1e-12 for r in rows) else 'NO'}.",
              ""]
    return "\n".join(lines)


def teacher_accuracy_tex() -> str | None:
    rows = ta_rows()
    if rows is None:
        return None
    body = [[k, str(len(g := [r for r in rows if r["kind"] == k])), *[ta_cell(g, key) for key, _ in TA_COLUMNS[:4]]]
            for k in TA_KINDS]
    return tex_table(
        "teacher-accuracy", "How accurate the teachers' kept answers are, evaluation runs, computed after all "
        "experiments: share of a case's kept answers (%), lowest to highest over cases, against the true sum. Each "
        "case's check sample was redrawn from the model before the round and reproduces the logged novelty and "
        "confirmed share exactly. Wrong but confirmed: wrong, and built identically by at least one other split.",
        ["Teacher", "Cases", "Teacher accuracy", "Confirmed share", "Wrong but confirmed", "Accuracy among confirmed"],
        body, rule_before={4}, fit=True)


def teacher_accuracy_numbers() -> dict[str, str]:
    from gainpred.teacher_accuracy import predictions
    rows = ta_rows()
    if rows is None:
        return {}
    out = {}
    for k in TA_KINDS:
        g = [r for r in rows if r["kind"] == k]
        for key, q in TA_QUANTITY.items():
            lo, hi, _ = ta_range(g, key)
            out[f"gpTa{TA_MACRO[k]}{q}Low"], out[f"gpTa{TA_MACRO[k]}{q}High"] = lo, hi
    p = predictions(rows)["pooled"]
    full = [r for r in rows if r["kind"] != "Weak starting models"]
    out |= {"gpTaFullCases": str(p["cases"]),
            "gpTaPOne": "held" if p["p1"] else "did not hold", "gpTaPOneFail": str(len(p["p1_fail"])),
            "gpTaPOneWorst": ta_range(full, "wrong_confirmed", 2)[1],
            "gpTaPTwo": "held" if p["p2"] else "did not hold", "gpTaPTwoFail": str(len(p["p2_fail"])),
            "gpTaPTwoWorst": ta_range(full, "accuracy_confirmed", 2)[0]}
    return out


def teacher_accuracy_figure(plt) -> bool:
    rows = ta_rows()
    if rows is None:
        return False
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 100], [0, 100], color="#52514e", lw=1, ls="--", zorder=1)
    for k in TA_KINDS:
        colour, marker = KIND_STYLE[k]
        g = [r for r in rows if r["kind"] == k and r["kept"]]
        hollow = k == "Weak starting models"
        ax.scatter([100 * r["confirmed_share"] for r in g], [100 * r["teacher_accuracy"] for r in g], s=22,
                   marker=marker, zorder=3, facecolors="none" if hollow else colour,
                   edgecolors=colour if hollow else "white", linewidths=1.0 if hollow else 0.4, label=f"{k} ({len(g)})")
    ax.set_xlim(-2, 102); ax.set_ylim(-2, 102)
    ax.set_xlabel("Confirmed share (%)"); ax.set_ylabel("Teacher accuracy (%)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "teacher_accuracy.png", dpi=200)
    fig.savefig(PAPER / "figure_teacher_accuracy.pdf"); plt.close(fig)
    return True


TEACHER_ACCURACY_CAPTION = (
    "Teacher accuracy against confirmed share, one point per evaluation case of the teachers that split; computed "
    "after all experiments. Both are shares of the case's kept answers, accuracy against the true sum. Dashed: "
    "accuracy = confirmed share. A point lies below the line by at most its share of wrong but confirmed answers.")


# ---------------------------------------------------------------- amendment 25: steered rounds
#
# Defined after all experiments; new runs (gainpred.steer reads them). The paper's words: a length
# is taught or skipped; novelty (check 1); starting model (starting point).

def steer_ready() -> bool:
    from gainpred import steer
    return {"fixed", "steer3", "frozen"} <= set(steer.arms_present())


def steer_arm_summary(arm: str) -> dict:
    from gainpred import steer as st
    fixed_kept = st.kept_total("fixed")
    secs = {s: st.seconds(arm, s) for s in st.SEEDS}
    fsecs = {s: st.seconds("fixed", s) for s in st.SEEDS}
    lowest = [min(st.final(st.load(arm, s))[d] for d in st.HARD) for s in st.SEEDS]
    hard_gap = {s: st.gaps(arm, s, st.HARD) for s in st.SEEDS}
    easy_gap = {s: st.gaps(arm, s, st.EASY) for s in st.SEEDS}
    worst = max(((s, d, g) for s in st.SEEDS for d, g in hard_gap[s].items()), key=lambda x: x[2])
    fall = st.worst_fall(arm) if arm != "fixed" else None
    return {
        "taught": [st.taught(arm, s) if arm != "fixed" else st.LENGTH_ROUNDS for s in st.SEEDS],
        "repair_taught": [st.taught(arm, s, repairs=True) if arm != "fixed" else 0 for s in st.SEEDS],
        "repairs": [st.repair_rounds(arm, s) if arm != "fixed" else 0 for s in st.SEEDS],
        "kept_pct": 100 * st.kept_total(arm) / fixed_kept,
        "build_pct": 100 * sum(x["build"] for x in secs.values()) / sum(x["build"] for x in fsecs.values()),
        "all_pct": 100 * sum(sum(x.values()) for x in secs.values()) / sum(sum(x.values()) for x in fsecs.values()),
        "lowest": lowest, "worst_gap": worst, "worst_easy": max(max(g.values()) for g in easy_gap.values()),
        "fall": fall}


def lo_hi_int(xs: list[int]) -> str:
    return str(xs[0]) if min(xs) == max(xs) else f"{min(xs)}–{max(xs)}"


def steer_table_rows() -> list[list[str]]:
    from gainpred import steer as st
    rows = []
    for arm in [a for a in ("fixed", "steer3", "frozen", "rehearse") if a in st.arms_present()]:
        m = steer_arm_summary(arm)
        fixed = arm == "fixed"
        rows.append([st.ARM_LABELS[arm], lo_hi_int(m["taught"]),
                     "–" if fixed else f"{sum(m['repairs'])} ({sum(m['repair_taught'])} lengths)",
                     f"{m['kept_pct']:.0f}", f"{m['build_pct']:.0f}", f"{m['all_pct']:.0f}",
                     f"{statistics.mean(m['lowest']):.1f} ({min(m['lowest']):.1f}–{max(m['lowest']):.1f})",
                     "–" if fixed else f"{m['worst_gap'][2]:.1f}",
                     "–" if fixed or m["fall"] is None else f"{m['fall']['fall']:.1f}"])
    return rows


STEER_HEADER = ["Schedule", "Lengths taught (of 20)", "Repair rounds", "Answers built %", "Build time %",
                "All timed %", "Lowest final accuracy", "Largest shortfall", "Largest fall when skipped"]


def steer_tex() -> str | None:
    if not steer_ready():
        return None
    return tex_table(
        "steer", "Letting the checks decide which lengths to teach (steered rounds), agreement rule, evaluation "
        "runs; defined after all experiments, predictions recorded before the runs. Lengths taught: of the 20 "
        "length-rounds per starting model in rounds 1-4, lowest to highest over starting models; repair rounds "
        "after round 4 (lengths taught in them) over the three starting models. Answers built and time: % of the "
        "fixed schedule, pooled. Lowest final accuracy over lengths 8-15 (%): mean (range) over starting models. "
        "Largest shortfall: points below the fixed schedule after its round 4, at any length 8-15. Largest fall "
        "when skipped: accuracy before minus after a round in which the length was skipped (points).",
        STEER_HEADER, steer_table_rows(), fit=True)


def steer_lengths_tex() -> str | None:
    from gainpred import steer as st
    if not steer_ready():
        return None
    arms = [a for a in ("fixed", "steer3", "frozen", "rehearse") if a in st.arms_present()]
    finals = {a: [st.final(st.load(a, s)) if a != "fixed" else st.accuracy_by_round(st.load(a, s))[4]
                  for s in st.SEEDS] for a in arms}
    rows = [[str(d)] + [f"{statistics.mean(f[d] for f in finals[a]):.1f} "
                        f"({min(f[d] for f in finals[a]):.1f}–{max(f[d] for f in finals[a]):.1f})" for a in arms]
            for d in st.HARD]
    return tex_table(
        "steer-lengths", "Final accuracy at each length (%), agreement rule, evaluation runs: mean (range) over "
        "the three starting models; the fixed schedule after round 4, the steered schedules after their last "
        "round (repair rounds included). Defined after all experiments.",
        ["Digits"] + [st.ARM_LABELS[a] for a in arms], rows, fit=True)


def steer_numbers() -> dict[str, str]:
    from gainpred import steer as st
    if not steer_ready():
        return {}
    b, c = steer_arm_summary("steer3"), steer_arm_summary("frozen")
    s, d, g = b["worst_gap"]
    out = {"gpSteerLevel": "3", "gpSteerLengthRounds": str(st.LENGTH_ROUNDS * len(st.SEEDS)),
           "gpSteerTaught": str(sum(b["taught"])), "gpSteerRepairRounds": str(sum(b["repairs"])),
           "gpSteerKeptPct": f"{b['kept_pct']:.0f}", "gpSteerBuildSecondsPct": f"{b['build_pct']:.0f}",
           "gpSteerSecondsPct": f"{b['all_pct']:.0f}",
           "gpSteerWorstGap": f"{g:.1f}", "gpSteerWorstGapDigits": str(d), "gpSteerWorstGapModel": str(s),
           "gpSteerWorstEasyGap": f"{b['worst_easy']:.1f}"}
    f = b["fall"]
    out |= {"gpSteerWorstFall": f"{f['fall']:.1f}" if f else "--",
            "gpSteerWorstFallDigits": str(f["digits"]) if f else "--",
            "gpSteerWorstFallRound": str(f["round"]) if f else "--"}
    out |= {"gpSteerSOne": "held" if st.s1()["pooled"]["held"] else "failed",
            "gpSteerSTwo": "held" if st.s2()["held"] else "failed"}
    out |= {"gpSteerFrozenTaught": str(sum(c["taught"])), "gpSteerFrozenKeptPct": f"{c['kept_pct']:.0f}",
            "gpSteerFrozenWorstGap": f"{c['worst_gap'][2]:.1f}", "gpSteerFrozenWorstGapDigits": str(c["worst_gap"][1])}
    rehearsed = "rehearse" in st.arms_present()
    out["gpSteerRehearsalRun"] = "yes" if rehearsed else "no"
    if rehearsed:
        r = steer_arm_summary("rehearse")
        out |= {"gpSteerRehearseKeptPct": f"{r['kept_pct']:.0f}", "gpSteerRehearseWorstGap": f"{r['worst_gap'][2]:.1f}",
                "gpSteerRehearseWorstFall": f"{r['fall']['fall']:.1f}" if r["fall"] else "--"}
    return out


def steer_md() -> str | None:
    from gainpred import steer as st
    if not steer_ready():
        return None
    lines = ["# Steered rounds (amendment 25)", "",
             "**Read this first.** The arms \"novelty at least 3%\" and \"rule as frozen\" below left skipped lengths out "
             "of training altogether. That was an error in the design of the experiment (see amendment 26): the checks "
             "say that teaching a learned length adds nothing, not that it can be left out. The collapses reported here "
             "are catastrophic forgetting (McCloskey and Cohen, 1989), which rehearsal prevents (Robins, 1995). The arm "
             "\"with rehearsal\" is the pilot for the experiment of amendment 26, written up in "
             "`results/writeup/steer_rehearsal.md`.", "",
             "Generated by `uv run python -m gainpred.writeup`; do not edit by hand. Defined after all experiments; "
             "predictions recorded before any steered run was launched. Agreement rule, evaluation runs (starting "
             "models 2-4). Per-length rows: `results/steer.csv`.", "", "## Summary", ""]
    lines += table(STEER_HEADER, steer_table_rows())
    lines += ["", "## Identity check (round 1 build log against the fixed schedule)", ""]
    for arm in ("steer3", "frozen", "rehearse"):
        if arm not in st.arms_present():
            continue
        for s in st.SEEDS:
            ok, diffs = st.identity(arm, s)
            lines.append(f"- `{st.run_name(arm, s)}`: " + ("passed" if ok else "not applicable (a length was skipped)"
                                                            if ok is None else "FAILED: " + "; ".join(diffs)))
            lines += [f"  - round-1 check differs: {x}" for x in st.check_differences(arm, s)]
    p1, p2 = st.s1(), st.s2()
    lines += ["", "## Predictions", "",
              "S1: for each starting model, after the last round of the main steered arm, accuracy at every length "
              "8-15 at most 3 points below the fixed schedule's after round 4, and at every easy length 3-7 at most "
              "3 points below. S2: summed over the three starting models, the main steered arm builds at most half "
              "as many training answers as the fixed schedule.", ""]
    lines += table(["Starting model", "S1", "Largest shortfall 8-15 (digits)", "Largest shortfall 3-7 (digits)",
                    "Answers built (steered / fixed)"],
                   [[str(s), "held" if p1[s]["held"] else "failed",
                     f"{p1[s]['worst_hard']:+.1f} ({p1[s]['worst_hard_digits']})",
                     f"{p1[s]['worst_easy']:+.1f} ({p1[s]['worst_easy_digits']})",
                     f"{p2['per_seed'][s][0]:,} / {p2['per_seed'][s][1]:,}"] for s in st.SEEDS] +
                   [["pooled", "held" if p1["pooled"]["held"] else "failed", f"{p1['pooled']['worst_hard']:+.1f}",
                     f"{p1['pooled']['worst_easy']:+.1f}", f"{p2['kept']:,} / {p2['fixed']:,}"]])
    lines += ["", f"S2: {'held' if p2['held'] else 'failed'} ({100 * p2['share']:.1f}% of the fixed schedule's "
              "answers).", "", "## Trigger for the rehearsal arm", ""]
    trig = st.trigger()
    lines += ["Fired: " + ("yes. " + "; ".join(f"starting model {r['seed']}, round {r['round']}, {r['digits']} digits: "
                                               f"{r['before']:.1f} -> {r['after']:.1f}" for r in trig) if trig else
                           "no (no skipped length fell more than 3 points in a round in which it was skipped)."), ""]
    for arm in ("steer3", "frozen", "rehearse"):
        if arm not in st.arms_present():
            continue
        lines += [f"## {st.ARM_LABELS[arm]}: decisions", ""]
        rows = []
        for s in st.SEEDS:
            for r in st.round_rows(arm, s):
                rows.append([str(s), str(r["round"]) + (" (repair)" if r["repair"] else ""), str(r["digits"]),
                             r["decision"] + (f" ({r['verdict']})" if r["verdict"] else ""),
                             pct(r["novelty"]),
                             pct(r["confirmed"]), f"{r['before']:.1f}", f"{r['after']:.1f}",
                             "–" if r["fixed_after"] is None else f"{r['fixed_after']:.1f}", f"{r['kept']:,}"])
        lines += table(["Starting model", "Round", "Digits", "Decision", "Novelty %", "Confirmed %", "Before",
                        "After", "Fixed after", "Kept"], rows)
        lines += [""]
    lines += ["## Time (seconds, summed over the three starting models)", ""]
    trows = []
    for arm in [a for a in ("fixed", "steer3", "frozen", "rehearse") if a in st.arms_present()]:
        t = {k: sum(st.seconds(arm, s)[k] for s in st.SEEDS) for k in ("checks", "build", "replay", "train")}
        trows.append([st.ARM_LABELS[arm]] + [f"{t[k]:.0f}" for k in ("checks", "build", "replay", "train")] +
                     [f"{sum(t.values()):.0f}"])
    lines += table(["Schedule", "Checks", "Build", "Replay", "Train", "All timed"], trows)
    lines += ["", "Evaluations are not timed in either schedule. The steered schedules' checks include the closing "
              "checks.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- amendment 26: steered rounds with rehearsal
#
# Defined after all other experiments. Arm (e) fixed schedule (starting points 0-1 stitched from two
# runs), (f) steered with rehearsal, (g) schedule only. Paper words: rebuilt, rehearsed, novelty,
# starting model, fixed schedule.

REH_GROUPS = (("Starting models 0–1 (predictions recorded)", (0, 1)),
              ("Starting models 2–4 (pilot, no prediction)", (2, 3, 4)))
REH_LABELS = {"fixed": "Fixed schedule", "rehearse": "Steered, rehearsal", "schedule": "Schedule only"}


def rehearsal_ready() -> bool:
    from gainpred import steer as st
    return set(st.REH_ARMS) <= set(st.arms_present(st.ALL_SEEDS))


def reh_final(arm: str, seed: int) -> dict[int, float]:
    from gainpred import steer as st
    return st.accuracy_by_round(st.load(arm, seed))[4] if arm == "fixed" else st.final(st.load(arm, seed))


def reh_summary(arm: str, seeds) -> dict:
    from gainpred import steer as st
    fixed = arm == "fixed"
    secs = sum(sum(st.seconds(arm, s).values()) for s in seeds)
    fsecs = sum(sum(st.seconds("fixed", s).values()) for s in seeds)
    lowest = [min(reh_final(arm, s)[d] for d in st.HARD) for s in seeds]
    hard = [(s, d, g) for s in seeds for d, g in st.gaps(arm, s, st.HARD).items()] if not fixed else []
    easy = [g for s in seeds for g in st.gaps(arm, s, st.EASY).values()] if not fixed else []
    fall = None if fixed else st.worst_not_rebuilt_fall(arm, seeds)
    rows = {s: st.round_rows(arm, s) for s in seeds}
    return {"rebuilt": [sum(r["rebuilt"] for r in rows[s] if not r["repair"]) for s in seeds],
            "repairs": sum(st.repair_rounds(arm, s) for s in seeds) if not fixed else 0,
            "repair_rebuilt": sum(r["rebuilt"] for s in seeds for r in rows[s] if r["repair"]),
            "kept": st.kept_total(arm, seeds), "kept_pct": 100 * st.kept_total(arm, seeds) / st.kept_total("fixed", seeds),
            "secs_pct": 100 * secs / fsecs, "lowest": lowest,
            "worst_gap": max(hard, key=lambda x: x[2]) if hard else None, "worst_easy": max(easy) if easy else None,
            "fall": fall}


def reh_row(arm: str, seeds) -> list[str]:
    m = reh_summary(arm, seeds)
    fixed = arm == "fixed"
    return [REH_LABELS[arm], lo_hi_int(m["rebuilt"]),
            "–" if fixed or arm == "schedule" else (f"{m['repairs']} ({m['repair_rebuilt']})" if m["repairs"] else "0"),
            f"{m['kept_pct']:.0f}", f"{m['secs_pct']:.0f}",
            f"{statistics.mean(m['lowest']):.1f} ({min(m['lowest']):.1f}–{max(m['lowest']):.1f})",
            "–" if fixed else f"{m['worst_gap'][2]:.1f}",
            "–" if fixed or m["fall"] is None else f"{m['fall']['fall']:.1f}"]


REH_HEADER = ["Schedule", "Rebuilt", "Repairs", "Built %", "Time %", "Lowest final", "Shortfall", "Fall"]


def blocked_table(label: str, caption: str, header: list[str], blocks: list[tuple[str, list[list[str]]]]) -> str:
    """A booktabs table in row blocks, each under an italic block heading; no scaling (it must fit
    the text width at \\small as it is)."""
    spec = "l" + "r" * (len(header) - 1)
    lines = [GENERATED_NOTE.rstrip(), r"\begin{table}[htbp]", r"\centering", rf"\caption{{{tex_escape(caption)}}}",
             rf"\label{{tab:{label}}}", r"\small", r"\setlength{\tabcolsep}{4pt}", rf"\begin{{tabular}}{{{spec}}}",
             r"\toprule", " & ".join(tex_escape(h) for h in header) + r" \\"]
    for title, rows in blocks:
        lines += [r"\midrule", rf"\multicolumn{{{len(header)}}}{{l}}{{\emph{{{tex_escape(title)}}}}} \\"]
        lines += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def rehearsal_tex() -> str | None:
    if not rehearsal_ready():
        return None
    blocks = [(t, [reh_row(a, seeds) for a in ("fixed", "rehearse", "schedule")]) for t, seeds in REH_GROUPS]
    return blocked_table(
        "rehearsal", "Letting the checks decide where new answers are built, with every length already taught "
        "rehearsed (agreement rule); defined after all other experiments. Predictions for starting models 0-1 were "
        "recorded before their runs; starting models 2-4 are the pilot. Steered: a length's answers are rebuilt only "
        "if novelty is at least 3%; other lengths already taught are rehearsed with the answers kept the last time "
        "they were built. Schedule only: each round rebuilds its two newest lengths and rehearses the rest, with no "
        "checks. Rebuilt: lengths rebuilt in rounds 1-4, of 20 per starting model (lowest-highest). Repairs: repair "
        "rounds after round 4 (lengths rebuilt in them). Built %, Time %: answers built and all timed seconds, % of "
        "the fixed schedule. Lowest final: lowest accuracy over lengths 8-15 after the last round (%), mean (range) "
        "over starting models. Shortfall: largest amount by which a length ends below the fixed schedule after its "
        "round 4 (points). Fall: largest fall in a round at a length already taught and not rebuilt (points).",
        REH_HEADER, blocks)


def rehearsal_lengths_tex() -> str | None:
    from gainpred import steer as st
    if not rehearsal_ready():
        return None
    blocks = []
    for title, seeds in REH_GROUPS:
        finals = {a: [reh_final(a, s) for s in seeds] for a in st.REH_ARMS}
        blocks.append((title, [[str(d)] + [f"{statistics.mean(f[d] for f in finals[a]):.1f} "
                                           f"({min(f[d] for f in finals[a]):.1f}–{max(f[d] for f in finals[a]):.1f})"
                                           for a in st.REH_ARMS] for d in st.HARD]))
    return blocked_table(
        "rehearsal-lengths", "Final accuracy at each length (%), agreement rule: mean (range) over starting models; "
        "the fixed schedule after round 4, the others after their last round (repairs included). Defined after all "
        "other experiments.", ["Digits"] + [REH_LABELS[a] for a in st.REH_ARMS], blocks)


def rehearsal_numbers() -> dict[str, str]:
    from gainpred import steer as st
    if not rehearsal_ready():
        return {}
    f, pilot = reh_summary("rehearse", st.LEARNING), reh_summary("rehearse", st.SEEDS)
    sched = reh_summary("schedule", st.ALL_SEEDS)
    t = st.t_predictions()
    out = {"gpRehKeptPct": f"{f['kept_pct']:.0f}", "gpRehSecondsPct": f"{f['secs_pct']:.0f}",
           "gpRehRebuilt": str(sum(f["rebuilt"])), "gpRehLengthRounds": str(st.LENGTH_ROUNDS * len(st.LEARNING)),
           "gpRehRepairRounds": str(f["repairs"]), "gpRehWorstGap": f"{f['worst_gap'][2]:.1f}",
           "gpRehWorstGapDigits": str(f["worst_gap"][1]), "gpRehWorstEasyGap": f"{f['worst_easy']:.1f}",
           "gpRehWorstFall": f"{f['fall']['fall']:.1f}" if f["fall"] else "--",
           "gpRehTOne": "held" if t["T1"]["pooled"]["held"] else "failed",
           "gpRehTTwo": "held" if t["T2"]["held"] else "failed",
           "gpRehTThree": "held" if t["T3"]["pooled"]["held"] else "failed",
           "gpRehPilotKeptPct": f"{pilot['kept_pct']:.0f}", "gpRehPilotSecondsPct": f"{pilot['secs_pct']:.0f}",
           "gpRehPilotWorstGap": f"{pilot['worst_gap'][2]:.1f}",
           "gpRehPilotWorstFall": f"{pilot['fall']['fall']:.1f}" if pilot["fall"] else "--",
           "gpSchedKeptPct": f"{sched['kept_pct']:.0f}", "gpSchedSecondsPct": f"{sched['secs_pct']:.0f}",
           "gpSchedWorstGap": f"{sched['worst_gap'][2]:.1f}",
           "gpSchedWorstFall": f"{sched['fall']['fall']:.1f}" if sched["fall"] else "--"}
    sched01, sched24 = reh_summary("schedule", st.LEARNING), reh_summary("schedule", st.SEEDS)
    out["gpRehVsSchedKeptDiff"] = f"{(f['kept'] - sched01['kept']) / 1000:.0f}"
    out["gpRehPilotVsSchedKeptDiff"] = f"{(pilot['kept'] - sched24['kept']) / 1000:.0f}"
    out["gpRehVsSchedWorstGap"] = f"{reh_vs_sched_gap()[0]:.1f}"
    return out


def reh_vs_sched_gap() -> tuple[float, int, int, float]:
    """Largest |steered - schedule only| final accuracy at any length 8-15, any starting model:
    (size, starting model, digits, signed difference steered minus schedule)."""
    from gainpred import steer as st
    best = max(((abs(reh_final("rehearse", s)[d] - reh_final("schedule", s)[d]), s, d,
                 reh_final("rehearse", s)[d] - reh_final("schedule", s)[d]) for s in st.ALL_SEEDS for d in st.HARD))
    return best


def steer_rehearsal_md() -> str | None:
    from gainpred import steer as st
    if not rehearsal_ready():
        return None
    lines = ["# Steered rounds with rehearsal (amendment 26)", "",
             "Generated by `uv run python -m gainpred.writeup`; do not edit by hand. Defined after all other "
             "experiments. Agreement rule. Arms: (e) fixed schedule (starting models 0-1: rounds 1-2 of "
             "`si-agreement-seedN-authors`, rounds 3-4 of `si-agreement-seedN-5cut`; 2-4: `si-agreement-seedN-authors`); "
             "(f) steered with rehearsal (novelty at least 3%; starting models 0-1 new, 2-4 the pilot of amendment 25); "
             "(g) schedule only (two newest lengths rebuilt per round, the rest rehearsed; no checks). Per-length rows: "
             "`results/steer_rehearsal.csv`. All runs were timed on the same GPU type (A10).", "", "## Summary", ""]
    for title, seeds in REH_GROUPS:
        lines += [f"### {title}", ""]
        lines += table(REH_HEADER, [reh_row(a, seeds) for a in ("fixed", "rehearse", "schedule")]) + [""]
    lines += ["Fall: largest fall in a round at a length already taught and not rebuilt (its stored answers "
              "rehearsed). Shortfall: largest amount below the fixed schedule after its round 4, lengths 8-15.", "",
              "## Identity check (round 1 build log against the fixed schedule)", ""]
    for arm in ("rehearse", "schedule"):
        for s in st.ALL_SEEDS:
            ok, diffs = st.identity(arm, s)
            missing = st.identity_not_compared(arm, s)
            lines.append(f"- `{st.run_name(arm, s)}`: " + ("passed" if ok else "not applicable" if ok is None
                                                            else "FAILED: " + "; ".join(diffs))
                         + (f" ({', '.join(missing)} not logged by the older fixed run, so not compared)" if missing else ""))
    t = st.t_predictions()
    lines += ["", "## Predictions (arm f, starting models 0-1; recorded before the runs)", "",
              "T1: for each starting model, after the last round, accuracy at every length 8-15 at most 3 points below "
              "the fixed schedule's after round 4, and at every easy length 3-7 at most 3 points below. T2: summed over "
              "the two starting models, at most half as many answers built as the fixed schedule. T3: in no round does "
              "accuracy at a length already taught and not rebuilt in that round fall by more than 3 points.", ""]
    rows = []
    for s in st.LEARNING:
        w = t["T3"][s]["worst"]
        rows.append([str(s), "held" if t["T1"][s]["held"] else "failed",
                     f"{t['T1'][s]['worst_hard']:.1f} ({t['T1'][s]['worst_hard_digits']})",
                     f"{t['T1'][s]['worst_easy']:.1f} ({t['T1'][s]['worst_easy_digits']})",
                     f"{t['T2']['per_seed'][s][0]:,} / {t['T2']['per_seed'][s][1]:,}",
                     "held" if t["T3"][s]["held"] else "failed",
                     "–" if w is None else f"{w['fall']:.1f} (round {w['round']}, {w['digits']} digits)"])
    w = t["T3"]["pooled"]["worst"]
    rows.append(["pooled", "held" if t["T1"]["pooled"]["held"] else "failed", f"{t['T1']['pooled']['worst_hard']:.1f}",
                 f"{t['T1']['pooled']['worst_easy']:.1f}", f"{t['T2']['kept']:,} / {t['T2']['fixed']:,}",
                 "held" if t["T3"]["pooled"]["held"] else "failed", "–" if w is None else f"{w['fall']:.1f}"])
    lines += table(["Starting model", "T1", "Largest shortfall 8-15 (digits)", "Largest shortfall 3-7 (digits)",
                    "Answers built (steered / fixed)", "T3", "Largest fall not rebuilt"], rows)
    lines += ["", f"T2: {'held' if t['T2']['held'] else 'failed'} ({100 * t['T2']['share']:.1f}% of the fixed "
              "schedule's answers). Shortfalls are signed: negative means the steered arm ended higher.", "",
              "## Steered against schedule only", "",
              "Where the two differ in whether a length's answers were rebuilt (rounds 1-4; repair rounds exist only in "
              "the steered arm):", ""]
    diffs = [d for s in st.ALL_SEEDS for d in st.decision_differences(s)]
    lines += table(["Starting model", "Round", "Digits", "Steered", "Schedule only", "Novelty %"],
                   [[str(d["seed"]), str(d["round"]), str(d["digits"]),
                     {True: "rebuilt", False: "not rebuilt", None: "–"}[d["steered"]],
                     {True: "rebuilt", False: "not rebuilt", None: "–"}[d["schedule"]], pct(d["novelty"])]
                    for d in diffs])
    rows = []
    for s in st.ALL_SEEDS:
        a, b = reh_final("rehearse", s), reh_final("schedule", s)
        ka, kb = st.kept_total("rehearse", (s,)), st.kept_total("schedule", (s,))
        dmax = max(st.HARD, key=lambda d: abs(a[d] - b[d]))
        rows.append([str(s), f"{ka:,}", f"{kb:,}", f"{a[dmax] - b[dmax]:+.1f} ({dmax})"])
    lines += ["", "Answers built and the largest difference in final accuracy (steered minus schedule only, points; "
              "digits):", ""]
    lines += table(["Starting model", "Steered built", "Schedule built", "Largest accuracy difference"], rows)
    for arm in ("rehearse", "schedule"):
        lines += ["", f"## {REH_LABELS[arm]}: per-length decisions", ""]
        rows = [[str(s), str(r["round"]) + (" (repair)" if r["repair"] else ""), str(r["digits"]),
                 "rebuilt" if r["rebuilt"] else ("rehearsed" if r["rehearsed"] else "not rebuilt, nothing stored"),
                 pct(r["novelty"]), f"{r['before']:.1f}", f"{r['after']:.1f}",
                 "–" if r["fixed_after"] is None else f"{r['fixed_after']:.1f}", f"{r['kept']:,}"]
                for s in st.ALL_SEEDS for r in st.round_rows(arm, s)]
        lines += table(["Starting model", "Round", "Digits", "Answers", "Novelty %", "Before", "After", "Fixed after",
                        "Built"], rows)
    lines += [""]
    return "\n".join(lines)


def paper_files() -> dict[str, str]:
    """Main-text tables and figure wrappers. Confirming runs are the evidence throughout."""
    from gainpred import p9b

    cells = load_cells()
    conf = [c for c in cells if c.seed in CONFIRMING and c.outcome not in ("unclassified", "missing")]
    n = len(conf)
    out = {}

    # Section 4: the prediction rule against always giving one prediction, and the confusion grid.
    out["table_p9b_summary.tex"], out["table_p9b_confusion.tex"] = p9b_tables(conf)

    out["table_p9b_cases.tex"] = cases_tex(conf)
    out["numbers.tex"] = numbers_tex(section4_numbers() | section5_numbers() | control_numbers()
                                     | novelty_numbers() | teacher_accuracy_numbers() | steer_numbers()
                                     | rehearsal_numbers())
    if (t := rehearsal_tex()) is not None:
        out["table_rehearsal.tex"], out["table_rehearsal_lengths.tex"] = t, rehearsal_lengths_tex()
    if (t := steer_tex()) is not None:
        out["table_steer.tex"], out["table_steer_lengths.tex"] = t, steer_lengths_tex()
    out["table_novelty_gain.tex"] = novelty_gain_tex()
    out["figure_novelty_gain.tex"] = tex_figure("novelty-gain", "figure_novelty_gain.pdf", NOVELTY_GAIN_CAPTION)
    if (ta := teacher_accuracy_tex()) is not None:
        out["table_teacher_accuracy.tex"] = ta
        out["figure_teacher_accuracy.tex"] = tex_figure("teacher-accuracy", "figure_teacher_accuracy.pdf",
                                                        TEACHER_ACCURACY_CAPTION, width=r"0.75\linewidth")
    if (ctrl := control_tex()) is not None:
        out["table_control_answers.tex"] = ctrl
    out["table_rise.tex"] = rise_tex(conf)
    out["table_weak.tex"] = weak_tex()
    out["table_no_outcome.tex"] = no_outcome_tex()
    out["table_round_level.tex"] = round_level_tex(conf)
    if (repro := reproduction_tex()) is not None:
        out["table_reproduction.tex"] = repro

    # Section 4: cost of the checks, per teacher, confirming runs (all rounds pooled per teacher).
    est, rows = replay_tokens_estimate(), []
    for pattern, label, addition, rounds in COST_ARMS:
        cs = [c for rnd in rounds for c in (round_cost(r, rnd, addition, est) for r in cost_runs(pattern, CONFIRMING))
              if c and c["train_s"]]
        if not cs:
            continue
        whole = sum(c["build_s"] + c["replay_s"] + c["checks_s"] + c["train_s"] for c in cs)
        checks = sum(c["checks_s"] for c in cs)
        rows.append([label, str(len(cs)), f"{whole / len(cs):.0f}", f"{checks / len(cs):.0f}", f"{100 * checks / whole:.0f}"])
    out["table_cost.tex"] = tex_table(
        "cost", "Seconds per round on one A10 GPU, evaluation runs: whole round (build, replay, checks, 3,000 updates; "
        "evaluation not timed) and the checks alone.", ["Teacher", "Rounds", "Round s", "Checks s", "Checks %"], rows)
    out["table_cost_addition.tex"] = tex_table(
        "cost-addition", "Seconds per round on one A10 GPU, evaluation runs on addition: whole round (build, replay, "
        "checks, 3,000 updates; evaluation not timed) and the checks alone.",
        ["Teacher", "Rounds", "Round s", "Checks s", "Checks %"], [r for r in rows if r[0] in ADDITION_COST_LABELS])

    # Section 5: agreement rule against the no-carry rule.
    pm, pc = pass_mark(CONFIRMING, "mean"), pass_mark(CONFIRMING, "carry")
    rows = [[str(d), fmt(pm[d][0], True), f"{fmt(pm[d][1], True)} ({pm[d][2]})", fmt(pc[d][0], True),
             f"{fmt(pc[d][1], True)} ({pc[d][2]})"] for d in LENGTHS]
    out["table_passmark.tex"] = tex_table(
        "passmark", "Agreement rule minus no-carry rule after round 4, points, evaluation runs: mean over three "
        "starting models and worst single run (its starting model).",
        ["Digits", "Mean gap", "Worst run", "Mean gap, carry", "Worst run, carry"], rows)

    nc, ag = carries_per_place("nocarry", CONFIRMING), carries_per_place("agreement", CONFIRMING)
    def m(xs):
        return f"{sum(xs) / len(xs):.1f}" if xs else "--"
    rows = [[str(d), m(nc[d]["cut 1"]), m(nc[d]["other cuts"]), m(nc[d]["no cut"]), m(ag[d]["cut 1"]),
             m(ag[d]["other cuts"]), m(ag[d]["no cut"])] for d in LENGTHS]
    out["table_filter_gap.tex"] = filter_gap_tex()
    out["table_carries.tex"] = tex_table(
        "carries", "Share (%) of kept training answers that carry out of a place, by kind of place, in the round "
        "the length is first trained on; mean over the evaluation runs. NC: no-carry rule; AG: agreement rule.",
        ["Digits", "NC: split 1", "NC: other splits", "NC: no split", "AG: split 1", "AG: other splits",
         "AG: no split"], rows)

    # Section 5: longest run.
    out["table_exp2.tex"] = exp2_tex()

    out["table_check3.tex"] = check3_tex()

    out["table_withheld_slim.tex"] = withheld_slim_tex()
    out["table_withheld.tex"] = tex_table(
        "withheld-full", "Crossing runs of 4-8 before and after one round (%), evaluation runs. Withheld: join at "
        "the split without crossing runs of left part 3-4. Control arm and edges-differ rows are observations.",
        ["Arm", "Starting model", "4", "5", "6", "7", "8"], withheld_rows(CONFIRMING), fit=True)

    # Section 4: our checks, the baseline as run, and the variant in its favour (confirming runs).
    comp, lv = baseline_comparison()
    lf = lv["favour"]
    rows_c = comp["confirming"]
    cost = baseline_cost(CONFIRMING)
    tf = sum(n * f for _, n, _, f, _ in cost)
    probe_share = 100 * sum(n * p for _, n, p, _, _ in cost) / tf
    checks_share = 100 * sum(n * c for _, n, _, _, c in cost) / tf
    # The table's cases are all addition, so its cost column is taken over the addition rounds (amendment 31).
    cost = [row for row in cost if row[0] in ADDITION_COST_LABELS]
    tf = sum(n * f for _, n, _, f, _ in cost)
    probe_share = 100 * sum(n * p for _, n, p, _, _ in cost) / tf
    checks_share = 100 * sum(n * c for _, n, _, _, c in cost) / tf
    labels = {"ours": "Our checks", "baseline": "Baseline, as run",
              "favour": "Baseline, variant in its favour (defined after the evaluation runs)"}

    def rule_rows(rs, cost_cols=True):
        out_rows = []
        for key in ("ours", "baseline", "favour"):
            def share(sel, k=key, pred=None):
                g = [r for r in rs if sel(r)]
                return f"{sum((r[k] == pred) if pred else r['outcome'] in p9b.MATCH[r[k]] for r in g)}/{len(g)}"
            allm = sum(r["outcome"] in p9b.MATCH[r[key]] for r in rs)
            out_rows.append([labels[key], f"{allm}/{len(rs)} ({100 * allm / len(rs):.1f})",
                             share(lambda r: r["outcome"] == "paid off"), share(lambda r: r["outcome"] == "did nothing"),
                             share(lambda r: r["outcome"] in HARM, pred="does harm"),
                             ("no" if key == "ours" else "yes") if cost_cols else "",
                             (f"{checks_share:.0f}" if key == "ours" else f"{probe_share:.0f}") if cost_cols else ""])
        return out_rows
    weak = [r for r in rows_c if r["case"] == p9b.CASES[5]]
    rows = rule_rows(rows_c) + [[f"Weak starting models only ({len(weak)} cases)", "", "", "", "", "", ""]] + [
        r for r in rule_rows(weak, cost_cols=False)]
    out["table_baseline.tex"] = tex_table(
        "baseline", f"Our checks against the train-briefly-and-look baseline, evaluation runs (cases correctly "
        f"predicted, %). Harmful cases flagged: harmful cases (did harm, or taught wrong answers) predicted to do "
        f"harm. The baseline trains 200 updates and reads accuracy on the test problems, so it needs their correct "
        f"answers; our checks do not. Cost: share of a full round, over the addition rounds. Levels fitted on the calibration runs: as run "
        f"G = {lv['G']}, H = {lv['H']}, D = {lv['D']}; variant Z = {lf['Z']}, G = {lf['G']}, H = {lf['H']}, D = {lf['D']}.",
        ["Rule", "All", "Paid off", "Did nothing", "Harmful cases flagged", "Correct answers", "Cost %"], rows,
        fit=True, rule_before={3})

    out["figure_exp1.tex"] = tex_figure("exp1", "figure_exp1.pdf",
                                        "Accuracy after round 4 (no selection after the no-carry rule: after its one "
                                        "round, round 3), evaluation runs; mean over the "
                                        "three starting models, range as bars.")
    out["figure_exp2.tex"] = tex_figure("exp2", "figure_exp2.pdf",
                                        "Longest run: accuracy by crossing-run length before and after one round, "
                                        "evaluation runs.")
    out["figure_checks.tex"] = tex_figure("checks", "figure_checks.pdf",
                                          "Novelty (all cases with an outcome) and confirmed share (cases that "
                                          "reach the agreement check) by outcome. Dashed: the levels fixed before the "
                                          "evaluation runs. Shaded: the range within which a level gives the same "
                                          "predictions on the evaluation runs (computed after them).")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in (("exp1.md", exp1), ("exp2.md", exp2), ("exp3.md", exp3), ("cost.md", cost_md),
                        ("baseline.md", baseline_md), ("filter_gap.md", filter_gap_md),
                        ("check3_addition.md", check3_md), ("control_answers.md", control_md),
                        ("teacher_accuracy.md", teacher_accuracy_md), ("steer.md", steer_md),
                        ("steer_rehearsal.md", steer_rehearsal_md),
                        ("runs.md", runs_md)):
        text = build()
        if text is None:  # an analysis whose results are not fetched yet
            continue
        (OUT / name).write_text(text)
        print(f"results/writeup/{name}")
    for f in figures():
        print(f"results/writeup/figures/{f}")
    from gainpred.texwrap import wrap_text  # short lines for the editor; the PDF is unchanged

    for name, text in paper_files().items():
        (PAPER / name).write_text(wrap_text(text))
        print(f"paper/generated/{name}")


if __name__ == "__main__":
    main()
