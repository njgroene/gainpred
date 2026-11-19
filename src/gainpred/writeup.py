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
CHECK3_CAPTION = (
    "Check 3 on addition: share (%) of a teacher's kept answers identical to one of the piece answers it "
    "asked for (piece sums of every cut it built; for sample and vote, its five samples). Real models: each "
    "round's check sample redrawn with the model before the round; range over starting points and rounds. "
    "Defined after the confirming runs.")


def check3_one_value(title: str, label: str, values: list[str]) -> str:
    """The single check-3 value of one teacher over lengths 8-15 (and seeds, rounds).

    The paper table shows one number per teacher, so it may only do that when every
    length gives the same single value; otherwise generation fails (the per-length
    table stays in results/writeup/check3_addition.md)."""
    if len(set(values)) != 1 or "–" in values[0]:
        raise ValueError(f"check 3 differs across lengths for {label} ({title}): {values}")
    return values[0]


def check3_tex() -> str:
    """One row per teacher: paper test, learning runs, confirming runs, and the reading."""
    blocks = {title.split(" (")[0]: dict((r[0], r[1:]) for r in rows) for title, rows in check3_addition_rows()}
    order = ("Paper test", "Real models, learning runs", "Real models, confirming runs")
    rows = []
    for _, label in TEACHER_ROWS:
        vals = [check3_one_value(b, label, blocks[b][label]) for b in order]
        if len(set(vals)) != 1 or vals[0] not in CHECK3_READING:
            raise ValueError(f"check 3 for {label} is not one of 0 and 100 everywhere: {vals}")
        rows.append([label, *vals, CHECK3_READING[vals[0]]])
    return tex_table("check3_addition", CHECK3_CAPTION,
                     ["Teacher", "Paper test", "Learning runs", "Confirming runs", "Reading"], rows, spec="lrrrl")


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
        "filter_gap", "What the filter removes and whether the student learns it anyway, confirming runs, in "
        "the round that first trains each length. Kept: share (%) of kept training answers with a carry out of "
        "the cut-1 place. Carry: accuracy (%) on the study group with a carry across cut 1, before and after "
        "the round (mean over starting points 2-4; lowest in brackets). No-carry: the other group after. "
        "NC: no-carry rule; AG: agreement rule. Defined after the confirming runs.",
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
    ("run f", "Run f (no selection)", True, range(3, 4)),
    ("weak", "No-carry rule, weak starting models", True, range(1, 2)),
) + tuple((f"runs-{t}-seed{{s}}-u30000", label, False, range(1, 2)) for t, label in RUN_TEACHERS) + (
    ("runs-join_at_the_cut-withheld-seed{s}-u30000", "Join at the cut, withheld", False, range(1, 2)),
    ("runs-join_at_the_cut-withheld-seed{s}-longruns", "Join at the cut, withheld, control arm", False, range(1, 2)),
)


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
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6})
    made = []

    # Figure 1: experiment 1, accuracy per length after round 4, confirming seeds.
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    series1 = [(lambda s: addition_run("nocarry", s, 4), 4, "mean", COLOURS[0], "-", "No-carry rule"),
               (lambda s: addition_run("agreement", s, 4), 4, "mean", COLOURS[1], "-", "Agreement rule"),
               (lambda s: addition_run("noselect", s, 4), 4, "no-carry", COLOURS[2], "-", "No selection, no-carry group"),
               (lambda s: addition_run("noselect", s, 4), 4, "carry", COLOURS[2], "--", "No selection, carry group (drawn 2 points up)"),
               (lambda s: addition_run("vote", s, 4), 4, "mean", COLOURS[3], "-", "Sample and vote"),
               (runf_run, 3, "no-carry", "#6250d6", "-", "Run f, no-carry group"),
               (runf_run, 3, "carry", "#6250d6", "--", "Run f, carry group")]
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
    ax.set_title("Experiment 1: confirming runs after round 4 (run f: round 3); mean of seeds 2-4, range as bars",
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
        draw(axes[0], series(f"runs-{t}-seed{{s}}-u30000", 1), colour, label)
    axes[0].set_title("Three teachers, full pool", loc="left", fontsize=9)
    draw(axes[1], series("runs-join_at_the_cut-withheld-seed{s}-u30000", 0), NEUTRAL, "Before")
    draw(axes[1], series("runs-join_at_the_cut-withheld-seed{s}-u30000", 1), COLOURS[0], "Withheld, after")
    draw(axes[1], series("runs-join_at_the_cut-withheld-seed{s}-longruns", 0), NEUTRAL, "Control, before", ls="--")
    draw(axes[1], series("runs-join_at_the_cut-withheld-seed{s}-longruns", 1), COLOURS[1], "Control, after")
    axes[1].axvspan(6.5, 8.4, color="#f1f0ec", zorder=0)
    axes[1].set_title("Withheld round (joined runs 7-8 never taught)", loc="left", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Joined run length"); ax.set_xticks(xs); ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Accuracy (%), seeds 2-4")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "exp2_joined.png", dpi=200); fig.savefig(PAPER / "figure_exp2.pdf"); plt.close(fig)
    made.append("exp2_joined.png")

    # Figure 3: experiment 3, check 1 (all classified cells) and check 2b (cells that reach it) by outcome group.
    rows = list(csv.DictReader((RESULTS_DIR / "p9b_cells.csv").open()))
    groups = (("paid off", ("paid off",)), ("did nothing", ("did nothing",)),
              ("did harm or\ntaught wrong", ("did harm", "taught wrong answers")))
    fig, (ax1, ax) = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for i, (label, members) in enumerate(groups):
        for seeds, colour, dx in ((CONFIRMING, COLOURS[0], 0.12), (LEARNING, NEUTRAL, -0.12)):
            v = [100 * float(r["check_1"]) for r in rows
                 if int(r["seed"]) in seeds and r["outcome"] in members and r["check_1"] not in ("", "None")]
            jitter = [((k * 37) % 21 - 10) / 120 for k in range(len(v))]
            ax1.scatter([i + dx + j for j in jitter], v, s=10, color=colour, alpha=0.6,
                        label=("Confirming (2-4)" if seeds == CONFIRMING else "Learning (0-1)") if i == 0 else None)
    ax1.axhline(100 * p9b.LEVEL_1, color="#0b0b0b", lw=1, ls="--")
    ax1.text(-0.45, 100 * p9b.LEVEL_1 + 2, f"check 1 level {p9b.LEVEL_1:.2f}", fontsize=7, ha="left")
    ax1.set_xticks(range(3)); ax1.set_xticklabels([g[0] for g in groups])
    ax1.set_ylabel("Share (%)"); ax1.set_ylim(-3, 103)
    ax1.set_title("Check 1, all classified cells", loc="left", fontsize=9)
    for i, (label, members) in enumerate(groups):
        for seeds, colour, dx in ((CONFIRMING, COLOURS[0], 0.12), (LEARNING, NEUTRAL, -0.12)):
            v = [100 * float(r["check_2b"]) for r in rows
                 if int(r["seed"]) in seeds and r["outcome"] in members and r["check_0"] == "True"
                 and r["check_2b"] not in ("", "None") and float(r["check_1"]) >= p9b.LEVEL_1
                 and float(r["kept_share"]) >= p9b.MIN_KEPT_SHARE]
            jitter = [((k * 37) % 21 - 10) / 120 for k in range(len(v))]
            ax.scatter([i + dx + j for j in jitter], v, s=10, color=colour, alpha=0.7,
                       label=("Confirming (2-4)" if seeds == CONFIRMING else "Learning (0-1)") if i == 0 else None)
    ax.axhline(38, color="#0b0b0b", lw=1, ls="--")
    ax.text(-0.45, 40, "2b level 0.38", fontsize=7, ha="left")
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(-3, 103)
    ax.set_title("Check 2b, cells that reach it", loc="left", fontsize=9)
    fig.legend(*ax1.get_legend_handles_labels(), frameon=False, fontsize=7, ncol=2, loc="lower center")
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "figures" / "exp3_checks.png", dpi=200); fig.savefig(PAPER / "figure_checks.pdf"); plt.close(fig)
    made.append("exp3_checks.png")
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


def tex_table(label: str, caption: str, header: list[str], rows: list[list[str]], spec: str | None = None,
              small: bool = True, fit: bool = False) -> str:
    """A booktabs table; every cell is escaped. Captions say what is shown, not what it means.
    ``fit`` scales a wide table down to the text width."""
    spec = spec or "l" + "r" * (len(header) - 1)
    lines = [GENERATED_NOTE.rstrip(), r"\begin{table}[t]", r"\centering", rf"\caption{{{tex_escape(caption)}}}",
             rf"\label{{tab:{label}}}"] + ([r"\small"] if small else []) + ([r"\resizebox{\linewidth}{!}{%"] if fit else []) + [
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(tex_escape(h) for h in header) + r" \\",
             r"\midrule"]
    lines += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"] + (["}"] if fit else []) + [r"\end{table}"]
    return "\n".join(lines) + "\n"


def tex_figure(label: str, pdf: str, caption: str, width: str = r"\linewidth") -> str:
    return "\n".join([GENERATED_NOTE.rstrip(), r"\begin{figure}[t]", r"\centering",
                      rf"\includegraphics[width={width}]{{generated/{pdf}}}", rf"\caption{{{tex_escape(caption)}}}",
                      rf"\label{{fig:{label}}}", r"\end{figure}"]) + "\n"


def paper_files() -> dict[str, str]:
    """Main-text tables and figure wrappers. Confirming runs are the evidence throughout."""
    from gainpred import p9b

    cells = load_cells()
    conf = [c for c in cells if c.seed in CONFIRMING and c.outcome not in ("unclassified", "missing")]
    n = len(conf)
    out = {}

    # Section 4: the verdict rule against always giving one verdict, and by outcome group.
    hits = sum(c.outcome in p9b.MATCH[c.verdict] for c in conf)
    rows = [["Verdict rule (fixed in advance)", f"{hits}/{n}", f"{100 * hits / n:.1f}"]]
    for v in ("does nothing", "pays off", "does harm"):
        m = sum(c.outcome in p9b.MATCH[v] for c in conf)
        rows.append([f"Always “{v}”", f"{m}/{n}", f"{100 * m / n:.1f}"])
    for name, members in (("paid off", {"paid off"}), ("did nothing", {"did nothing"}),
                          ("did harm or taught wrong answers", {"did harm", "taught wrong answers"})):
        g = [c for c in conf if c.outcome in members]
        m = sum(c.outcome in p9b.MATCH[c.verdict] for c in g)
        rows.append([f"Verdict rule, cells that {name}", f"{m}/{len(g)}", f"{100 * m / max(len(g), 1):.1f}"])
    out["table_p9b_summary.tex"] = tex_table(
        "p9b-summary", "Verdicts given before training against outcomes after training, confirming runs "
        "(starting points 2-4, classified cells).", ["Rule", "Cells matched", "%"], rows)

    rows = []
    for case in p9b.CASES:
        g = [c for c in conf if p9b.case_of(c) == case]
        if g:
            rows.append([case, str(len(g)), str(sum(c.outcome in p9b.MATCH[c.verdict] for c in g))])
    out["table_p9b_cases.tex"] = tex_table("p9b-cases", "Verdict matches by case, confirming runs.",
                                           ["Case", "Cells", "Matches"], rows)

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
        "cost", "Seconds per round on one A10 GPU, confirming runs: whole round (build, replay, checks, 3,000 updates; "
        "evaluation not timed) and the checks alone.", ["Teacher", "Rounds", "Round s", "Checks s", "Checks %"], rows)

    # Section 5: agreement rule against the no-carry rule.
    pm, pc = pass_mark(CONFIRMING, "mean"), pass_mark(CONFIRMING, "carry")
    rows = [[str(d), fmt(pm[d][0], True), f"{fmt(pm[d][1], True)} ({pm[d][2]})", fmt(pc[d][0], True),
             f"{fmt(pc[d][1], True)} ({pc[d][2]})"] for d in LENGTHS]
    out["table_passmark.tex"] = tex_table(
        "passmark", "Agreement rule minus no-carry rule after round 4, points, confirming runs: mean over three "
        "starting points and worst single run (its starting point).",
        ["Digits", "Mean gap", "Worst run", "Mean gap, carry", "Worst run, carry"], rows)

    nc, ag = carries_per_place("nocarry", CONFIRMING), carries_per_place("agreement", CONFIRMING)
    def m(xs):
        return f"{sum(xs) / len(xs):.1f}" if xs else "--"
    rows = [[str(d), m(nc[d]["cut 1"]), m(nc[d]["other cuts"]), m(nc[d]["no cut"]), m(ag[d]["cut 1"]),
             m(ag[d]["other cuts"]), m(ag[d]["no cut"])] for d in LENGTHS]
    out["table_filter_gap.tex"] = filter_gap_tex()
    out["table_check3_addition.tex"] = check3_tex()
    out["table_carries.tex"] = tex_table(
        "carries", "Share (%) of kept training answers that carry out of a place, by kind of place, in the round "
        "the length is first trained on; mean over starting points 2-4. NC: no-carry rule; AG: agreement rule.",
        ["Digits", "NC: cut 1", "NC: other cuts", "NC: no cut", "AG: cut 1", "AG: other cuts", "AG: no cut"], rows)

    # Section 5: longest run.
    rows = []
    for label, run, rnd in (("Before", "runs-join_at_the_cut-seed{s}-u30000", 0),
                            *((lab, f"runs-{t}-seed{{s}}-u30000", 1) for t, lab in RUN_TEACHERS),
                            ("Join at the cut, withheld", "runs-join_at_the_cut-withheld-seed{s}-u30000", 1)):
        evs = [evals(run.format(s=s), "study").get(rnd) for s in CONFIRMING]
        row = [label, mean_range([100 * e["all"] for e in evs])]
        row += [mean_range([joined(e, k) for e in evs]) for k in (5, 6, 7, 8)]
        rows.append(row)
    out["table_exp2.tex"] = tex_table(
        "exp2", "Longest run, one round: study-set accuracy (%), all groups and joined run lengths 5-8; mean over "
        "starting points 2-4 (lowest-highest).", ["Teacher", "All", "Joined 5", "Joined 6", "Joined 7", "Joined 8"],
        rows, fit=True)

    rows = []
    for pattern, label in CHECK3_ARMS:
        v = [100 * e["check_3"] for s in CONFIRMING for e in events(pattern.format(s=s))
             if e["event"] == "checks" and e.get("check_3") is not None]
        rows.append([label, mean_range(v)])
    out["table_check3.tex"] = tex_table(
        "check3", "Check 3: share (%) of teacher outputs identical to a piece answer, over lengths 12-20 and "
        "starting points 2-4 (lowest-highest).", ["Teacher", "Check 3"], rows)

    out["table_withheld.tex"] = tex_table(
        "withheld", "Joined runs of 4-8 before and after one round (%), confirming runs. Withheld: join at the cut "
        "without run-across problems of left part 3-4. Control arm and edges-differ rows are observations.",
        ["Arm", "Seed", "4", "5", "6", "7", "8"], withheld_rows(CONFIRMING), fit=True)

    # Section 4: our checks, the baseline as run, and the variant in its favour (confirming runs).
    comp, lv = baseline_comparison()
    lf = lv["favour"]
    rows_c = comp["confirming"]
    cost = baseline_cost(CONFIRMING)
    tf = sum(n * f for _, n, _, f, _ in cost)
    probe_share = 100 * sum(n * p for _, n, p, _, _ in cost) / tf
    checks_share = 100 * sum(n * c for _, n, _, _, c in cost) / tf
    labels = {"ours": "Our checks", "baseline": "Baseline, as run",
              "favour": "Baseline, variant in its favour (defined after the confirming runs)"}
    rows = []
    for key in ("ours", "baseline", "favour"):
        def matched(sel, k=key):
            g = [r for r in rows_c if sel(r)]
            return f"{sum(r['outcome'] in p9b.MATCH[r[k]] for r in g)}/{len(g)}"
        allm = sum(r["outcome"] in p9b.MATCH[r[key]] for r in rows_c)
        rows.append([labels[key], f"{allm}/{len(rows_c)} ({100 * allm / len(rows_c):.1f})",
                     matched(lambda r: r["outcome"] == "paid off"), matched(lambda r: r["outcome"] == "did nothing"),
                     matched(lambda r: r["outcome"] in ("did harm", "taught wrong answers")),
                     "no" if key == "ours" else "yes",
                     f"{checks_share:.0f}" if key == "ours" else f"{probe_share:.0f}"])
    out["table_baseline.tex"] = tex_table(
        "baseline", f"Our checks against the train-briefly-and-look baseline, confirming runs (cells matched, %). "
        f"The baseline trains 200 updates and reads accuracy on the test problems, so it needs their reference "
        f"answers; our checks do not. Cost: share of a full round. Levels fitted on starting points 0-1: as run "
        f"G = {lv['G']}, H = {lv['H']}, D = {lv['D']}; variant Z = {lf['Z']}, G = {lf['G']}, H = {lf['H']}, D = {lf['D']}.",
        ["Rule", "All", "Paid off", "Did nothing", "Did harm", "Reference answers", "Cost %"], rows, fit=True)

    out["figure_exp1.tex"] = tex_figure("exp1", "figure_exp1.pdf",
                                        "Accuracy after round 4 (run f: round 3), confirming runs; mean over starting "
                                        "points 2-4, range as bars.")
    out["figure_exp2.tex"] = tex_figure("exp2", "figure_exp2.pdf",
                                        "Longest run: accuracy by joined run length before and after one round, "
                                        "confirming runs.")
    out["figure_checks.tex"] = tex_figure("checks", "figure_checks.pdf",
                                          "Check 1 (all classified cells) and check 2b (cells that reach it) by "
                                          "outcome group; dashed: the levels fixed before the confirming runs.")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in (("exp1.md", exp1), ("exp2.md", exp2), ("exp3.md", exp3), ("cost.md", cost_md),
                        ("baseline.md", baseline_md), ("filter_gap.md", filter_gap_md),
                        ("check3_addition.md", check3_md), ("runs.md", runs_md)):
        (OUT / name).write_text(build())
        print(f"results/writeup/{name}")
    for f in figures():
        print(f"results/writeup/figures/{f}")
    from gainpred.texwrap import wrap_text  # short lines for the editor; the PDF is unchanged

    for name, text in paper_files().items():
        (PAPER / name).write_text(wrap_text(text))
        print(f"paper/generated/{name}")


if __name__ == "__main__":
    main()
