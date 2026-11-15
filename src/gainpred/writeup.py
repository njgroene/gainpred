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
import subprocess
from pathlib import Path

from gainpred.cuts import ADDITION_CUTS, HARD_DIGITS, cut_positions
from gainpred.report import RESULTS_DIR

OUT = RESULTS_DIR / "writeup"
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
    return 100 * (carry if group == "carry" else (carry + plain) / 2)


def fmt(x: float | None, signed: bool = False) -> str:
    if x is None:
        return "–"
    return f"{x:+.1f}" if signed else f"{x:.1f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"] + \
        ["| " + " | ".join(r) + " |" for r in rows]


# ---------------------------------------------------------------- experiment 1

def exp1_final_rows(seeds: tuple[int, ...], group: str) -> list[list[str]]:
    rows = []
    for arm, label in ARMS:
        for s in seeds:
            rows.append([label, str(s)] + [fmt(addition_accuracy(addition_run(arm, s, 4), 4, d, group))
                                           for d in LENGTHS])
    for s in seeds:
        run = runf_run(s)
        rows.append(["Run f (no selection, round 3)", str(s)] + [fmt(addition_accuracy(run, 3, d, group))
                                                                  for d in LENGTHS])
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


def carries_per_place(arm: str, seeds: tuple[int, ...]):
    """Share of kept training answers carrying out of each kind of place, in the round the length is new.

    Places are counted from the units (0). A cut boundary at ``p`` digits from the
    left of an n-digit number is crossed by a carry out of place ``n - 1 - p``.
    Kinds: the cut-1 boundary; the boundaries of the other cuts in use; every
    other place below the top digit (the top place's carry only lengthens the
    answer and is left out). Returns {digits: {kind: [share per seed]}}.
    """
    out = {}
    for d in LENGTHS:
        cuts = ADDITION_CUTS[d]
        cut1 = {d - 1 - p for p in cut_positions(cuts[0])}
        others = {d - 1 - p for c in cuts[1:] for p in cut_positions(c)} - cut1
        rest = set(range(d - 1)) - cut1 - others
        shares = {"cut 1": [], "other cuts": [], "no cut": []}
        for s in seeds:
            rnd = new_round(d)
            build = next((e for e in events(addition_run(arm, s, rnd))
                          if e["event"] == "build" and e["round"] == rnd and e["digits"] == d), None)
            if not build or "carries_by_place" not in build or not build["kept"]:
                continue
            counts, kept = build["carries_by_place"], build["kept"]
            for kind, places in (("cut 1", cut1), ("other cuts", others), ("no cut", rest)):
                if places:
                    shares[kind].append(100 * sum(counts[i] for i in places) / len(places) / kept)
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
             "set). Run f: after its one round (round 3, from the no-carry run's round 2).", "",
             "## Confirming runs (starting points 2-4)", ""]
    lines += table(head, exp1_final_rows(CONFIRMING, "mean"))
    lines += ["", "Carry group alone:", ""] + table(head, exp1_final_rows(CONFIRMING, "carry"))
    lines += ["", "## Learning runs (starting points 0-1), shown separately", "",
              "Rounds 3-4 from the five-cut reruns (`-5cut`).", ""]
    lines += table(head, exp1_final_rows(LEARNING, "mean"))

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
              "strings with a longest run of 7-8 inside (P6). The control arm is an observation, not a test.", ""]
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

    shead = ["Arm", "Seed"] + [f"J{n} a={a}" for n in (4, 5, 6) for a in ("1-2", "3-4")]
    lines += ["## Withheld split by left part", "",
              "Run-across study problems with joined run 4-6, split by the left part a (a = 3-4 is never taught in "
              "the withheld pool). Direct answers, before → after (%). From `scripts/withheld_split.py`.", "",
              "**Confirming runs**", ""] + table(shead, split_rows(CONFIRMING))
    lines += ["", "**Learning runs**", ""] + table(shead, split_rows(LEARNING)) + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------- experiment 3

def exp3() -> str:
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
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6})
    made = []

    # Figure 1: experiment 1, accuracy per length after round 4, confirming seeds.
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for (arm, label), colour in zip(ARMS, COLOURS):
        ys, lo, hi = [], [], []
        for d in LENGTHS:
            v = [addition_accuracy(addition_run(arm, s, 4), 4, d) for s in CONFIRMING]
            v = [x for x in v if x is not None]
            ys.append(sum(v) / len(v)); lo.append(min(v)); hi.append(max(v))
        ax.errorbar(LENGTHS, ys, yerr=[[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]],
                    color=colour, lw=2, marker="o", ms=4, capsize=2, label=label)
    ax.set_xlabel("Digits"); ax.set_ylabel("Accuracy after round 4 (%)"); ax.set_ylim(-3, 103)
    ax.set_title("Experiment 1: confirming runs (mean of seeds 2-4, range as bars)", loc="left", fontsize=9)
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0.12, 0.62))
    fig.tight_layout(); fig.savefig(OUT / "figures" / "exp1_accuracy.png", dpi=200); plt.close(fig)
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
    fig.tight_layout(); fig.savefig(OUT / "figures" / "exp2_joined.png", dpi=200); plt.close(fig)
    made.append("exp2_joined.png")

    # Figure 3: experiment 3, check 2b by outcome group, for cells that reach the 2b test.
    rows = list(csv.DictReader((RESULTS_DIR / "p9b_cells.csv").open()))
    groups = (("paid off", ("paid off",)), ("did nothing", ("did nothing",)),
              ("did harm or\ntaught wrong", ("did harm", "taught wrong answers")))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
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
    ax.text(2.45, 40, "2b level 0.38", fontsize=7, ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Check 2b (%)"); ax.set_ylim(-3, 103)
    ax.set_title("Experiment 3: check 2b in cells that reach it", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="center right")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "exp3_2b.png", dpi=200); plt.close(fig)
    made.append("exp3_2b.png")
    return made


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in (("exp1.md", exp1), ("exp2.md", exp2), ("exp3.md", exp3), ("runs.md", runs_md)):
        (OUT / name).write_text(build())
        print(f"results/writeup/{name}")
    for f in figures():
        print(f"results/writeup/figures/{f}")


if __name__ == "__main__":
    main()
