"""The "train briefly and look" baseline, scored like our checks (owner, 15 November 2026).

Usage::

    uv run python -m gainpred.baseline fit      # learning runs only (seeds 0-1): choose G, H, D; write the levels
    uv run python -m gainpred.baseline score    # confirming runs, once, with the recorded levels

The baseline's verdict, per cell (run, round, length)
=====================================================
From a probe (``gainpred.probe``): the round's training data rebuilt, 200
updates, then the same test groups. ``change`` = probe accuracy minus the
accuracy before the round at that length (points); ``easy change`` the same on
easy problems; a *new* length is one P9b scores as new (new, and at most 50%
before the round); ``gap`` = the best teacher's probe at that length (same
starting point and round) minus this probe.

Checked in this order (fixed before fitting; harm first, as our own rule puts
check 0 first, so that "up a lot but far below the best" counts as harm, the
baseline's version of "taught wrong answers"):

1. *does harm* if easy change < -H, or (learned length and change < -H), or
   (new length and gap > D);
2. *pays off* if new length and change >= G;
3. *does nothing* otherwise.

Choosing G, H and D (owner's rule)
==================================
On the learning runs (starting points 0-1) only: every whole-point value of
G, H and D in 0-100 is scored against the learning outcomes
(the same classified cells as P9b); the triples with the most matches are
kept. (The first fit searched H only up to 30 and its best H, 29-30, sat at that
edge; widened to 0-100 on the learning runs, before any confirming probe was
fetched: 155 rather than 151 of 176 matches.) Ties (there are many) are broken by a rule fixed before fitting: take
the median of each level over the best triples; if that triple is itself among
the best, use it, else the best triple nearest to it (sum of absolute
differences, then smallest G, H, D). The chosen levels are written to
``results/baseline_levels.json`` and recorded in the protocol and committed
before any confirming probe is fetched.

What the baseline uses that ours does not
=========================================
It reads the test problems' reference answers (accuracy after 200 updates).
Our checks never look at a right answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from gainpred.report import RESULTS_DIR

RUNS = Path("runs")
LEVELS = RESULTS_DIR / "baseline_levels.json"
LEVELS_FAVOUR = RESULTS_DIR / "baseline_favour_levels.json"
PROBES = RESULTS_DIR / "probes"
MATCH = {"does harm": {"did harm", "taught wrong answers"}, "does nothing": {"did nothing"}, "pays off": {"paid off"}}
LEARNING, CONFIRMING = (0, 1), (2, 3, 4)


def probe_name(run: str) -> str:
    return f"probe-{run}"


def probe_evals(run: str) -> dict[tuple[int, str], dict]:
    """The probe's accuracies by (round, set). A failed probe is replaced as a whole by its
    ``-r2`` rerun; a probe without a ``done`` event (stopped part-way) is never scored."""
    path = None
    for name in (f"{probe_name(run)}-r2", probe_name(run)):
        candidate = PROBES / f"{name}.jsonl"
        if candidate.exists() and '"event": "done"' in candidate.read_text():
            path = candidate
            break
    if path is None:
        return {}
    out = {}
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "eval":
            out[(e["round"], e["set"])] = e["accuracy"]
    return out


def original_easy_before(run: str, rnd: int) -> float | None:
    path = RESULTS_DIR / "runs" / run / "rounds.jsonl"
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "eval" and e["set"] == "easy" and e["round"] == rnd - 1:
            return 100 * e["accuracy"]["all"]
    return None


def cells(seeds: tuple[int, ...]) -> list[dict]:
    """P9b's classified cells for these seeds, with the probe's accuracy added (None if no probe)."""
    out = []
    with (RESULTS_DIR / "p9b_cells.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if int(r["seed"]) in seeds]
    cache: dict[str, dict] = {}
    for r in rows:
        if r["outcome"] in ("unclassified", "missing"):
            continue
        run, rnd, d = r["run"], int(r["round"]), int(r["digits"])
        ev = cache.setdefault(run, probe_evals(run))
        acc = ev.get((rnd, r["test_set"]))
        easy = ev.get((rnd, "easy"))
        probe = None if acc is None else 100 * (acc[f"{d}/carry"] + acc[f"{d}/no-carry"]) / 2
        before = 100 * float(r["before"])
        eb = original_easy_before(run, rnd)
        out.append({"run": run, "seed": int(r["seed"]), "round": rnd, "digits": d, "teacher": r["teacher"],
                    "new": r["new_length"] == "True" and float(r["before"]) <= 0.50, "before": before,
                    "probe": probe, "change": None if probe is None else probe - before,
                    "easy_change": None if easy is None or eb is None else 100 * easy["all"] - eb,
                    "outcome": r["outcome"], "p9b_verdict": r["verdict"], "row": r})
    best: dict[tuple, float] = {}
    for c in out:
        if c["probe"] is not None:
            key = (c["seed"], c["round"], c["digits"])
            best[key] = max(best.get(key, 0.0), c["probe"])
    for c in out:
        c["gap"] = None if c["probe"] is None else best[(c["seed"], c["round"], c["digits"])] - c["probe"]
    return out


def verdict(c: dict, g: float, h: float, d: float) -> str:
    if (c["easy_change"] is not None and c["easy_change"] < -h) or (not c["new"] and c["change"] < -h) \
            or (c["new"] and c["gap"] > d):
        return "does harm"
    if c["new"] and c["change"] >= g:
        return "pays off"
    return "does nothing"


def verdict_favour(c: dict, z: float, g: float, h: float, d: float) -> str:
    """Variant defined after the confirming runs, in the baseline's favour (amendment 21).

    The as-run rule tests the gap to the best teacher before asking whether anything
    changed, so a new length that did not move at all is called harm when the best
    teacher moved a lot (20 of the as-run baseline's 34 confirming misses, all sample
    and vote). Our own checks dropped that order at decision point 1. Here:
    1. harm if easy change < -H, or a learned length's change < -H;
    2. a learned length otherwise: nothing;
    3. a new length: nothing if change < Z; else harm if gap > D; else pays off if
       change >= G; else nothing.
    """
    if (c["easy_change"] is not None and c["easy_change"] < -h) or (not c["new"] and c["change"] < -h):
        return "does harm"
    if not c["new"]:
        return "does nothing"
    if c["change"] < z:
        return "does nothing"
    if c["gap"] > d:
        return "does harm"
    if c["change"] >= g:
        return "pays off"
    return "does nothing"


def fit_favour() -> dict:
    """Exhaustive search over whole-point Z, G, H, D in 0-100 on the learning runs, same tie-break.

    101^4 combinations are scored at once per H with numpy: for each cell the verdict is a
    simple function of the four levels, so matches are summed over a (Z, G, D) grid.
    """
    import numpy as np

    cs = cells(LEARNING)
    if any(c["probe"] is None for c in cs):
        raise SystemExit("learning cells without a probe")
    grid = np.arange(101)
    Z, G, D = np.meshgrid(grid, grid, grid, indexing="ij")  # axes: z, g, d
    best, top = -1, []
    for h in grid:
        total = np.zeros(Z.shape, dtype=np.int32)
        for c in cs:
            harm_first = (c["easy_change"] is not None and c["easy_change"] < -h) or (not c["new"] and c["change"] < -h)
            if harm_first or not c["new"]:
                v = "does harm" if harm_first else "does nothing"
                total += int(c["outcome"] in MATCH[v])
                continue
            nothing = (c["change"] < Z) | ((c["gap"] <= D) & (c["change"] < G))
            harm = (c["change"] >= Z) & (c["gap"] > D)
            pays = (c["change"] >= Z) & (c["gap"] <= D) & (c["change"] >= G)
            ok = {"does nothing": nothing, "does harm": harm, "pays off": pays}
            total += sum(ok[v].astype(np.int32) for v in MATCH if c["outcome"] in MATCH[v])
        m = int(total.max())
        idx = np.argwhere(total == m)
        quads = [(int(z), int(g), int(h), int(d)) for z, g, d in idx]
        if m > best:
            best, top = m, quads
        elif m == best:
            top += quads
    med = tuple(float(np.median([t[i] for t in top])) for i in range(4))
    chosen = med if med in top else min(top, key=lambda t: (sum(abs(a - b) for a, b in zip(t, med)), t))
    levels = {"Z": chosen[0], "G": chosen[1], "H": chosen[2], "D": chosen[3], "learning_matches": best,
              "learning_cells": len(cs), "best_quadruples": len(top), "median_of_best": med,
              "ranges_of_best": {k: [min(t[i] for t in top), max(t[i] for t in top)] for i, k in enumerate("ZGHD")},
              "defined": "after the confirming runs, in the baseline's favour (amendment 21)"}
    LEVELS_FAVOUR.write_text(json.dumps(levels, indent=2) + "\n")
    return levels


def matches(cs: list[dict], g: float, h: float, d: float) -> int:
    return sum(c["outcome"] in MATCH[verdict(c, g, h, d)] for c in cs)


def fit() -> None:
    cs = cells(LEARNING)
    missing = [c for c in cs if c["probe"] is None]
    if missing:
        raise SystemExit(f"{len(missing)} learning cells have no probe yet, e.g. {missing[0]['run']} r{missing[0]['round']}")
    best, top = -1, []
    for g in range(0, 101):
        for h in range(0, 101):
            for d in range(0, 101):
                m = matches(cs, g, h, d)
                if m > best:
                    best, top = m, [(g, h, d)]
                elif m == best:
                    top.append((g, h, d))
    med = tuple(statistics.median(t[i] for t in top) for i in range(3))
    chosen = med if med in top else min(top, key=lambda t: (sum(abs(a - b) for a, b in zip(t, med)), t))
    levels = {"G": chosen[0], "H": chosen[1], "D": chosen[2], "learning_matches": best, "learning_cells": len(cs),
              "best_triples": len(top), "median_of_best": med,
              "ranges_of_best": {k: [min(t[i] for t in top), max(t[i] for t in top)] for i, k in enumerate("GHD")}}
    LEVELS.write_text(json.dumps(levels, indent=2) + "\n")
    print(json.dumps(levels, indent=2))


def score() -> dict:
    """Score the confirming runs once with the recorded levels (and the learning runs for reference)."""
    lv = json.loads(LEVELS.read_text())
    lf = json.loads(LEVELS_FAVOUR.read_text()) if LEVELS_FAVOUR.exists() else None
    out = {}
    for name, seeds in (("learning", LEARNING), ("confirming", CONFIRMING)):
        cs = cells(seeds)
        have = [c for c in cs if c["probe"] is not None]
        for c in have:
            c["baseline_verdict"] = verdict(c, lv["G"], lv["H"], lv["D"])
            if lf:
                c["favour_verdict"] = verdict_favour(c, lf["Z"], lf["G"], lf["H"], lf["D"])
        out[name] = {"cells": cs, "scored": have, "missing": len(cs) - len(have)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["fit", "fit-favour", "score"])
    args = parser.parse_args()
    if args.command == "fit":
        fit()
    elif args.command == "fit-favour":
        print(json.dumps(fit_favour(), indent=2))
    else:
        res = score()
        for name, r in res.items():
            m = sum(c["outcome"] in MATCH[c["baseline_verdict"]] for c in r["scored"])
            print(f"{name}: {m}/{len(r['scored'])} cells match (missing probes for {r['missing']})")
            if r["scored"] and "favour_verdict" in r["scored"][0]:
                f = sum(c["outcome"] in MATCH[c["favour_verdict"]] for c in r["scored"])
                print(f"{name}, variant in the baseline's favour: {f}/{len(r['scored'])}")


if __name__ == "__main__":
    main()
