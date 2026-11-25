"""Analyses of the pilot's logs (protocol amendment 27, part 3). Made after the runs; not predictions.

Every number comes from ``results/pilot/runs/*/rounds.jsonl``; nothing is typed by hand.

Which runs
==========
* The solid agreement-rule runs are repeats 0-2. For repeat 1, floors 2 and 3 come from
  ``solid-agr-r1`` and all of floor 4 from ``solid-agr-r1-again`` (its floor-4 call was restarted by
  Modal and refused to overwrite its round 1; the floor-4 events in ``solid-agr-r1`` are ignored).
* "Checks before the round" are the ``checks`` events: 2,000 fresh problems per new length, with
  the model as it was before the round. ``audit_direct_right`` there is how often the model's own
  direct answer was right; novelty, kept share and confirmed share are the teacher's.

A. What the checks would have built (agreement rule)
====================================================
For each repeat, in the order the lengths were taught (floor 2 round 1 = 16-17, ... floor 4 round 4
= 38-39): the first round where novelty at both new lengths is below 50% (below 3%), and how many
of the 24 lengths 16-39 have novelty of 50% (3%) or more. With the agreement rule the frozen verdict
(P9b) reduces to "novelty of 50% or more": the confirmed share is 100% by construction (two splits
agree on every kept answer) and the kept share is always above 5%.

B. Error rate before the round
==============================
100% minus ``audit_direct_right`` per length (fresh problems), mean of the three repeats, and the
test-set error after the round that first taught the length.

C. The frozen rule scored on the pilot's rounds
===============================================
One case = one new length in one round of one run (solid-agr, solid-nc, corrupt-agr, corrupt-nc;
three repeats each). Verdict: ``p9b.verdict`` with its frozen levels on the checks' kept share,
novelty and confirmed share; the basics check passes if every easy length (3-7) was at least 98%
on the test set before the round (P9b's check 0). Before = ``audit_direct_right``; after = test-set
accuracy at that length after the round (mean of the two carry groups); best = the highest "after"
at that length among all pilot runs with the same repeat number. Outcome, as in the task:

* before above 50%: "did nothing" if after is within 3 points of before; "did harm" if more than 3
  points lower; otherwise unclassified;
* before 50% or lower: "paid off" if after is at least 20 points higher and within 10 points of
  best; "did nothing" if after is less than 10 points higher; "taught wrong answers" if higher but
  30 or more points below best; otherwise unclassified.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from gainpred import p9b
from gainpred.floors import FLOORS, PILOT_RESULTS, round_lengths
from gainpred.floors_report import RUNS, accuracy, events

REPEATS = (0, 1, 2)
LENGTHS = range(16, 40)
CLIMB_FLOORS = (2, 3, 4)
CONDITIONS = {"solid / agreement": "solid-agr", "solid / no-carry": "solid-nc",
              "corrupted / agreement": "corrupt-agr", "corrupted / no-carry": "corrupt-nc"}
OUT = PILOT_RESULTS


# ---------------------------------------------------------------------------- which events

def run_events(prefix: str, repeat: int) -> list[dict]:
    """Events of one condition and repeat, with repeat 1 of the solid agreement runs stitched:
    floors 2-3 from solid-agr-r1, floor 4 from solid-agr-r1-again."""
    if prefix == "solid-agr" and repeat == 1:
        return ([e for e in events("solid-agr-r1") if e.get("floor") in (2, 3)]
                + [e for e in events("solid-agr-r1-again") if e.get("floor") == 4])
    return events(f"{prefix}-r{repeat}")


def checks_of(evs: list[dict]) -> dict[tuple[int, int, int], dict]:
    """(floor, round, digits) -> the checks event before that round."""
    return {(e["floor"], e["round"], e["digits"]): e for e in evs if e["event"] == "checks"}


def taught_rounds(floors=CLIMB_FLOORS):
    """(floor, round, (two new lengths)) in teaching order."""
    return [(f, k, round_lengths(f, k)) for f in floors for k in range(1, 5)]


def after_of(evs: list[dict], floor: int, round_: int, digits: int) -> float | None:
    """Test-set accuracy (%) at ``digits`` after ``round_`` of ``floor``."""
    return accuracy(evs, floor, round_).get(digits)


def easy_before_passes(evs: list[dict], floor: int, round_: int) -> bool | None:
    """P9b's check 0: every easy length (3-7) at least 98% on the test set before the round."""
    sets = [e for e in evs if e["event"] == "eval" and e.get("floor") == floor and e["round"] == round_ - 1
            and e["set"] == "easy"]
    if not sets:
        return None
    return all(v >= p9b.EASY_PASS for k, v in sets[-1]["accuracy"].items() if k != "all")


# ---------------------------------------------------------------------------- A

def skip_table() -> dict:
    rows = []
    lowest_kept = (2.0, None)
    confirmed = []
    for r in REPEATS:
        ch = checks_of(run_events("solid-agr", r))
        nov = {}
        first50 = first3 = None
        for f, k, new in taught_rounds():
            vals = [ch[(f, k, d)]["novelty"] for d in new]
            for d in new:
                e = ch[(f, k, d)]
                nov[d] = e["novelty"]
                confirmed.append(e["confirmed_share"])
                if e["kept_share"] < lowest_kept[0]:
                    lowest_kept = (e["kept_share"], (r, f, k, d))
            if first50 is None and all(v < 0.50 for v in vals):
                first50 = (f, k, new)
            if first3 is None and all(v < 0.03 for v in vals):
                first3 = (f, k, new)
        rows.append({"repeat": r, "first_below_50": first50, "first_below_3": first3,
                     "at_least_50": [d for d in LENGTHS if nov[d] >= 0.50],
                     "at_least_3": [d for d in LENGTHS if nov[d] >= 0.03], "novelty": nov})
    return {"rows": rows, "lowest_kept_share": lowest_kept[0], "lowest_kept_at": lowest_kept[1],
            "confirmed_all_100": all(c == 1.0 for c in confirmed), "confirmed_min": min(confirmed)}


def round_label(x) -> str:
    if x is None:
        return "none"
    f, k, new = x
    return f"floor {f} round {k} ({new[0]}, {new[1]})"


# ---------------------------------------------------------------------------- B

def error_rates() -> dict:
    """Before: 100 - audit_direct_right per length and repeat; after: test-set error after the
    round that first taught the length."""
    before, after = {}, {}
    for r in REPEATS:
        evs = run_events("solid-agr", r)
        ch = checks_of(evs)
        for f, k, new in taught_rounds():
            for d in new:
                before[(d, r)] = 100 * (1 - ch[(f, k, d)]["audit_direct_right"])
                after[(d, r)] = 100 - after_of(evs, f, k, d)
    mean = {d: sum(before[(d, r)] for r in REPEATS) / len(REPEATS) for d in LENGTHS}
    return {"before": before, "after": after, "mean": mean, "after_min": min(after.values()),
            "after_max": max(after.values())}


ERRORS_CAPTION = ("Error rate of the model's own direct answer before the round that teaches each length, "
                  "agreement rule, solid floor 1 (starting model 2); bar: mean of three repeats, dots: repeats. "
                  "Before is measured on 2,000 fresh problems per "
                  "length; after on the test set. After the round that first taught it, the test-set error at "
                  "every length 16-39 in every repeat was between {lo:.1f}% and {hi:.1f}%. Read after the runs, "
                  "from their logs.")


def errors_figure(paths: list[Path]) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gainpred.writeup import COLOURS

    er = error_rates()
    # Drawn at the width of the paper's text block (5.5 in), so the PDF is included unscaled and 8 pt is 8 pt.
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                         "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    xs = list(LENGTHS)
    ys = [er["mean"][d] for d in xs]
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.bar(xs, ys, width=0.72, color=COLOURS[1], zorder=2)
    for d in xs:
        ax.scatter([d] * len(REPEATS), [er["before"][(d, r)] for r in REPEATS], s=9, color="#0b0b0b", zorder=3,
                   linewidths=0)
    top = max(max(er["before"].values()), 1.0)
    ax.set_ylim(0, top * 1.12)
    for f in CLIMB_FLOORS:
        lo, hi = FLOORS[f].start, FLOORS[f].stop - 1
        if f > 2:
            ax.axvline(lo - 0.5, color="#52514e", lw=0.8, ls=":", zorder=1)
        ax.text((lo + hi) / 2, top * 1.10, f"{lo}\u2013{hi} digits", ha="center", va="top", fontsize=8,
                color="#52514e")
    ax.set_xticks(xs)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Number of digits")
    ax.set_ylabel("Error before the round (%)")
    fig.tight_layout()
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, **({"dpi": 200} if p.suffix == ".png" else {}))
    plt.close(fig)
    return er


# ---------------------------------------------------------------------------- C

@dataclass
class Case:
    condition: str
    run: str
    repeat: int
    floor: int
    round: int
    digits: int
    kept_share: float
    novelty: float | None
    confirmed_share: float | None
    basics: bool | None
    verdict: str
    before: float
    after: float
    best: float | None = None
    outcome: str = ""

    @property
    def gain(self) -> float:
        return self.after - self.before


def best_runs() -> dict[int, list[list[dict]]]:
    """Every pilot run of the 24-25 November pilot, by repeat number (the control of amendment 27,
    the timing and the rehearsal runs are not among them)."""
    out: dict[int, list[list[dict]]] = {r: [] for r in REPEATS}
    for r in REPEATS:
        for prefix in ("solid-agr", "solid-nc", "corrupt-agr", "corrupt-nc", "solid-key"):
            evs = run_events(prefix, r)
            if evs:
                out[r].append(evs)
    return out


def best_after(runs: list[list[dict]], digits: int) -> float | None:
    """The highest test-set accuracy at ``digits`` after the round that taught it, over runs."""
    f = next(f for f in CLIMB_FLOORS if digits in FLOORS[f])
    k = next(k for k in range(1, 5) if digits in round_lengths(f, k))
    vals = [a for evs in runs if (a := after_of(evs, f, k, digits)) is not None]
    return max(vals) if vals else None


def outcome(before: float, after: float, best: float | None) -> str:
    if before > 50:
        if abs(after - before) <= 3:
            return "did nothing"
        if after < before - 3:
            return "did harm"
        return "unclassified"
    if after - before >= 20 and best is not None and best - after <= 10:
        return "paid off"
    if after - before < 10:
        return "did nothing"
    if after > before and best is not None and best - after >= 30:
        return "taught wrong answers"
    return "unclassified"


def rule_cases() -> list[Case]:
    best = best_runs()
    cases = []
    for cond, prefix in CONDITIONS.items():
        for r in REPEATS:
            evs = run_events(prefix, r)
            run = f"{prefix}-r{r}"
            for (f, k, d), e in sorted(checks_of(evs).items()):
                a = after_of(evs, f, k, d)
                if a is None:
                    continue
                confirmed = e["confirmed_share"] if isinstance(e["confirmed_share"], float) else None
                basics = easy_before_passes(evs, f, k)
                c = Case(cond, run + ("" if not (prefix == "solid-agr" and r == 1 and f == 4) else "-again"), r, f, k,
                         d, e["kept_share"], e["novelty"], confirmed, basics,
                         p9b.verdict(basics, e["kept_share"], e["novelty"], confirmed),
                         100 * e["audit_direct_right"], a, best_after(best[r], d))
                c.outcome = outcome(c.before, c.after, c.best)
                cases.append(c)
    return cases


def matches(c: Case) -> bool:
    return c.outcome in p9b.MATCH[c.verdict]


def rule_counts(cases: list[Case]) -> dict[str, dict]:
    out = {}
    for cond in list(CONDITIONS) + ["total"]:
        cs = [c for c in cases if cond == "total" or c.condition == cond]
        cl = [c for c in cs if c.outcome != "unclassified"]
        out[cond] = {"cases": len(cs), "classified": len(cl), "match": sum(matches(c) for c in cl),
                     "paid off": sum(c.outcome == "paid off" for c in cl),
                     "did nothing": sum(c.outcome == "did nothing" for c in cl),
                     "did harm": sum(c.outcome in ("did harm", "taught wrong answers") for c in cl),
                     "verdict pays off": sum(c.verdict == "pays off" for c in cs),
                     "verdict does nothing": sum(c.verdict == "does nothing" for c in cs),
                     "verdict does harm": sum(c.verdict == "does harm" for c in cs)}
    return out


# ---------------------------------------------------------------------------- the write-up

def pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}"


def num(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def lengths_text(ds: list[int]) -> str:
    return f"{len(ds)}" + (f" ({', '.join(map(str, ds))})" if ds else "")


def analysis_md() -> str:
    a = skip_table()
    out = ["# Pilot: analyses of the logs (amendment 27, part 3)", "",
           "Generated by `uv run python -m gainpred.floors_analysis` from `results/pilot/runs/`. Made after the "
           "pilot's runs, from its logs; these are not predictions. Repeat 1 of the solid agreement-rule runs: "
           "floors 2-3 from `solid-agr-r1`, floor 4 from `solid-agr-r1-again`.", "",
           "## A. What the checks would have built (agreement rule, solid floor 1)", "",
           "| Repeat | First round with novelty < 50% at both new lengths | First round with novelty < 3% at both "
           "| Lengths (of 24) with novelty >= 50% | Lengths (of 24) with novelty >= 3% |",
           "| --- | --- | --- | --- | --- |"]
    for row in a["rows"]:
        out.append(f"| {row['repeat']} | {round_label(row['first_below_50'])} | {round_label(row['first_below_3'])} | "
                   f"{lengths_text(row['at_least_50'])} | {lengths_text(row['at_least_3'])} |")
    r, f, k, d = a["lowest_kept_at"]
    out += ["", "Novelty per length (%):", "", "| Repeat | " + " | ".join(map(str, LENGTHS)) + " |",
            "|" + " --- |" * 25]
    out += [f"| {row['repeat']} | " + " | ".join(pct(row["novelty"][d_]) for d_ in LENGTHS) + " |" for row in a["rows"]]
    out += ["", f"* With the agreement-rule teacher the frozen rule (P9b) reduces to \"novelty of 50% or more\": "
            f"the confirmed share is 100% by construction (in these runs: "
            f"{'100% in every check' if a['confirmed_all_100'] else 'lowest ' + pct(a['confirmed_min'])}), and the "
            f"kept share is always above 5% (lowest in these runs: {pct(a['lowest_kept_share'])}%, repeat {r}, "
            f"floor {f} round {k}, {d} digits).",
            "* These numbers are read from runs that trained on every length, so they show what a novelty rule "
            "would have built, not what such a loop would have reached.", ""]

    er = error_rates()
    out += ["## B. Error rate before the round", "",
            "Figure: `results/pilot/figure_climb_errors.png` (and `.pdf`; also `paper/generated/figure_climb_errors.pdf`).",
            "", ERRORS_CAPTION.format(lo=er["after_min"], hi=er["after_max"]), "",
            "| Digits | Mean error before (%) | " + " | ".join(f"Repeat {r}" for r in REPEATS)
            + " | Test-set error after, lowest-highest (%) |", "|" + " --- |" * 6]
    for d_ in LENGTHS:
        aft = [er["after"][(d_, r)] for r in REPEATS]
        out.append(f"| {d_} | {er['mean'][d_]:.2f} | " + " | ".join(f"{er['before'][(d_, r)]:.2f}" for r in REPEATS)
                   + f" | {min(aft):.1f}-{max(aft):.1f} |")
    out.append("")

    cases = rule_cases()
    counts = rule_counts(cases)
    out += ["## C. The frozen rule scored on the pilot's rounds", "",
            "Scored after the runs. \"Before\" comes from 2,000 fresh problems (the model's own direct answer, "
            "`audit_direct_right`), not from the test set. With the agreement rule the confirmed share is 100% by "
            "construction.", "",
            "| Condition | Cases | Classified | Verdict matches outcome | Paid off | Did nothing | Did harm or taught "
            "wrong | Verdict pays off / does nothing / does harm |", "|" + " --- |" * 8]
    for cond, c in counts.items():
        out.append(f"| {cond} | {c['cases']} | {c['classified']} | {c['match']} of {c['classified']} | "
                   f"{c['paid off']} | {c['did nothing']} | {c['did harm']} | {c['verdict pays off']} / "
                   f"{c['verdict does nothing']} / {c['verdict does harm']} |")
    misses = [c for c in cases if c.outcome != "unclassified" and not matches(c)]
    out += ["", f"**Misses ({len(misses)}):**", ""]
    if misses:
        out += ["| Run | Floor | Round | Digits | Kept share | Novelty | Confirmed | Basics | Verdict | Before | After "
                "| Best | Outcome |", "|" + " --- |" * 13]
        out += [f"| `{c.run}` | {c.floor} | {c.round} | {c.digits} | {pct(c.kept_share)} | {pct(c.novelty)} | "
                f"{pct(c.confirmed_share)} | {c.basics} | {c.verdict} | {num(c.before)} | {num(c.after)} | "
                f"{num(c.best)} | {c.outcome} |" for c in misses]
    uncl = [c for c in cases if c.outcome == "unclassified"]
    out += ["", f"**Unclassified cases ({len(uncl)}): novelty beside the gain (after minus before):**", ""]
    if uncl:
        out += ["| Run | Floor | Round | Digits | Novelty (%) | Before | After | Gain (points) | Best | Verdict |",
                "|" + " --- |" * 10]
        out += [f"| `{c.run}` | {c.floor} | {c.round} | {c.digits} | {pct(c.novelty)} | {num(c.before)} | "
                f"{num(c.after)} | {num(c.gain)} | {num(c.best)} | {c.verdict} |" for c in uncl]
    out += ["", "Every case: `results/pilot/rule_cases.csv`.", ""]
    return "\n".join(out)


def write_all() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    errors_figure([OUT / "figure_climb_errors.png", OUT / "figure_climb_errors.pdf"])
    cases = rule_cases()
    with (OUT / "rule_cases.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[*asdict(cases[0]), "gain", "match"])
        w.writeheader()
        for c in cases:
            w.writerow({**asdict(c), "gain": c.gain, "match": matches(c) if c.outcome != "unclassified" else ""})
    (OUT / "analysis.md").write_text(analysis_md())


if __name__ == "__main__":
    write_all()
    print("wrote results/pilot/analysis.md, rule_cases.csv, figure_climb_errors.{png,pdf}")
