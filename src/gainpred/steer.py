"""Steered rounds (protocol amendment 25): fetch, identity check, measures, predictions.

Analysis only; the steering itself is in ``gainpred.rounds`` (``--steer``). Defined after all
experiments; nothing already measured changes and P9b is not rescored.

Usage::

    uv run python -m gainpred.steer --fetch     # download the steered runs, copy their logs, check identity
    uv run python -m gainpred.steer             # measures, predictions, results/steer.csv

What is compared
================
For each starting model 2-4, the *fixed* arm (``si-agreement-seedN-authors``: every length
built in every round) against the *steered* arms, which build only the lengths a rule teaches:

* (b) ``si-agreement-steer3-seedN`` -- teach a length if novelty (check 1) is at least 3%;
* (c) ``si-agreement-steerfrozen-seedN`` -- teach only if the frozen P9b verdict is "pays off";
* (d) ``si-agreement-steer3r-seedN`` -- as (b), re-using stored answers of skipped lengths
  (launched only if the trigger fires: in (b), a skipped length falls more than 3 points in a
  round in which it was skipped).

Accuracy at a length is P9b's: the mean of the no-carry and carry groups, on the study set at
8-12 digits and the five-cut study set at 13-15; easy lengths 3-7 from the easy set. "Final" is
after the last round the run made, repair rounds included.

Predictions (recorded before any steered run was launched)
==========================================================
S1: for each starting model, after the last round of (b), accuracy at every length 8-15 is at
most 3 points below the fixed arm's after round 4, and at every easy length 3-7 at most 3
points below. S2: summed over the three starting models, (b) keeps (builds) at most half as
many training answers as the fixed arm.
"""
from __future__ import annotations

import csv
import json
import statistics
import subprocess
from pathlib import Path

from gainpred.report import RESULTS_DIR, RoundsRun

SEEDS = (2, 3, 4)
ARMS = {"fixed": "si-agreement-seed{s}-authors", "steer3": "si-agreement-steer3-seed{s}",
        "frozen": "si-agreement-steerfrozen-seed{s}", "rehearse": "si-agreement-steer3r-seed{s}"}
ARM_LABELS = {"fixed": "Fixed schedule", "steer3": "Steered, novelty at least 3%",
              "frozen": "Steered, rule as frozen", "rehearse": "Steered, novelty at least 3%, with rehearsal"}
STEERED = ("steer3", "frozen", "rehearse")
HARD = range(8, 16)
EASY = range(3, 8)
SCHEDULED = range(1, 5)
LENGTH_ROUNDS = 20  # lengths per starting model over rounds 1-4: 2 + 4 + 6 + 8
FIVE_CUT = range(13, 16)
CSV = RESULTS_DIR / "steer.csv"
MARGIN = 3.0  # points (S1 and the trigger)


def run_name(arm: str, seed: int) -> str:
    return ARMS[arm].format(s=seed)


def load(arm: str, seed: int) -> RoundsRun | None:
    d = RESULTS_DIR / "runs" / run_name(arm, seed)
    return RoundsRun.load(d) if (d / "rounds.jsonl").exists() else None


def arms_present() -> list[str]:
    return [a for a in ARMS if all(load(a, s) is not None and load(a, s).of("done") for s in SEEDS)]


# ---------------------------------------------------------------- accuracy

def accuracy_by_round(run: RoundsRun) -> dict[int, dict[int, float]]:
    """round -> length -> accuracy (%), P9b's test group per length; easy lengths 3-7 too."""
    sets = {}
    for e in run.of("eval"):
        sets.setdefault(e["round"], {})[e["set"]] = e["accuracy"]
    out = {}
    for rnd, acc in sets.items():
        row = {}
        for d in HARD:
            src = acc.get("study5") if d in FIVE_CUT and "study5" in acc else acc["study"]
            row[d] = 100 * (src[f"{d}/no-carry"] + src[f"{d}/carry"]) / 2
        for d in EASY:
            row[d] = 100 * acc["easy"][str(d)]
        out[rnd] = row
    return out


def final(run: RoundsRun) -> dict[int, float]:
    acc = accuracy_by_round(run)
    return acc[max(acc)]


# ---------------------------------------------------------------- per (round, length) rows

def decisions(run: RoundsRun) -> dict[tuple[int, int], dict]:
    """(round, length) -> the steer event that decided it (closing checks: the repair round they
    belong to; the last closing check, which no round follows, is left out)."""
    rounds_made = {e["round"] for e in run.of("round_done")}
    return {(e["round"], e["digits"]): e for e in run.of("steer") if e["round"] in rounds_made}


def round_rows(arm: str, seed: int) -> list[dict]:
    run, fixed = load(arm, seed), load("fixed", seed)
    acc, acc_fixed = accuracy_by_round(run), accuracy_by_round(fixed)
    builds = {(e["round"], e["digits"]): e for e in run.of("build")}
    checks = {(e["round"], e["digits"]): e for e in run.of("checks") + run.of("closing_checks")}
    steer = decisions(run)
    rows = []
    for e in run.of("round_done"):
        rnd = e["round"]
        top = 15 if rnd > 4 else (9, 11, 13, 15)[rnd - 1]
        for d in range(8, top + 1):
            c, b, st = checks.get((rnd, d), {}), builds.get((rnd, d)), steer.get((rnd, d))
            rows.append({"arm": arm, "seed": seed, "round": rnd, "repair": rnd > 4, "digits": d,
                         "decision": st["decision"] if st else "teach", "verdict": (st or {}).get("verdict"),
                         "novelty": c.get("check_1"),
                         "kept_share": c["kept"] / c["problems"] if c.get("problems") else None,
                         "confirmed": c.get("check_2b"), "before": acc[rnd - 1][d], "after": acc[rnd][d],
                         "fixed_after": acc_fixed.get(rnd, {}).get(d), "kept": b["kept"] if b else 0,
                         "drawn": b["drawn"] if b else 0, "build_seconds": b["seconds"] if b else 0.0})
    return rows


def all_rows(arms=None) -> list[dict]:
    return [r for a in (arms or arms_present()) for s in SEEDS for r in round_rows(a, s)]


# ---------------------------------------------------------------- cost

def seconds(arm: str, seed: int) -> dict[str, float]:
    """Timed seconds: checks, build (answers for training), replay, train; closing checks included.

    The round's "build" timer covers building and replay; replay is that minus the per-length
    build events. Training comes from each round's training log. Evaluations are not timed (in
    either arm)."""
    run = load(arm, seed)
    out = {"checks": 0.0, "build": 0.0, "replay": 0.0, "train": 0.0}
    builds = run.of("build")
    for e in run.of("round_done"):
        b = sum(x["seconds"] for x in builds if x["round"] == e["round"])
        out["build"] += b
        out["replay"] += e["seconds"].get("build", 0.0) - b
        out["checks"] += e["seconds"].get("checks", 0.0)
        log = RESULTS_DIR / "runs" / run.name / f"round_{e['round']}" / "log.jsonl"
        if log.exists() and not e.get("skipped"):
            out["train"] += json.loads(log.read_text().splitlines()[-1]).get("seconds", {}).get("train", 0.0)
    for e in run.of("closing_done"):
        out["checks"] += e["seconds"].get("checks", 0.0)
    return out


# ---------------------------------------------------------------- identity check

IDENTITY_KEYS = ("drawn", "kept", "kept_share", "kept_right", "carries_by_place", "shortfall")


def identity(arm: str, seed: int) -> tuple[bool | None, list[str]]:
    """Round 1's build log against the fixed arm's, when both lengths were taught (amendment 25).

    Returns (passed, differences); passed is None when a length was skipped (nothing to compare).
    Differences in the round-1 check values are reported separately (``check_differences``).
    """
    run, fixed = load(arm, seed), load("fixed", seed)
    mine = {e["digits"]: e for e in run.of("build") if e["round"] == 1}
    if set(mine) != {8, 9}:
        return None, []
    theirs = {e["digits"]: e for e in fixed.of("build") if e["round"] == 1}
    diffs = [f"{run.name} round 1, {d} digits, {k}: {mine[d][k]} vs fixed {theirs[d][k]}"
             for d in (8, 9) for k in IDENTITY_KEYS if mine[d][k] != theirs[d][k]]
    return not diffs, diffs


CHECK_KEYS = ("problems", "kept", "check_1", "teachable", "check_2_kept", "check_2b")


def check_differences(arm: str, seed: int) -> list[str]:
    run, fixed = load(arm, seed), load("fixed", seed)
    mine = {e["digits"]: e for e in run.of("checks") if e["round"] == 1}
    theirs = {e["digits"]: e for e in fixed.of("checks") if e["round"] == 1}
    return [f"{run.name} round 1, {d} digits, {k}: {mine[d][k]} vs fixed {theirs[d][k]}"
            for d in (8, 9) for k in CHECK_KEYS if mine[d].get(k) != theirs[d].get(k)]


# ---------------------------------------------------------------- measures and predictions

def kept_total(arm: str) -> int:
    return sum(e["kept"] for s in SEEDS for e in load(arm, s).of("build"))


def taught(arm: str, seed: int, repairs: bool = False) -> int:
    rows = [r for r in round_rows(arm, seed) if r["repair"] == repairs]
    return sum(r["decision"] == "teach" for r in rows)


def repair_rounds(arm: str, seed: int) -> int:
    return sum(e["round"] > 4 for e in load(arm, seed).of("round_done"))


def gaps(arm: str, seed: int, lengths) -> dict[int, float]:
    """Shortfall against the fixed arm after round 4, points (positive = the steered arm is lower)."""
    mine, fixed = final(load(arm, seed)), accuracy_by_round(load("fixed", seed))[4]
    return {d: fixed[d] - mine[d] for d in lengths}


def falls(arm: str) -> list[dict]:
    """Every skipped (round, length) with its fall, before minus after (points)."""
    return [{**r, "fall": r["before"] - r["after"]} for s in SEEDS for r in round_rows(arm, s)
            if r["decision"] == "skip"]


def worst_fall(arm: str) -> dict | None:
    f = falls(arm)
    return max(f, key=lambda r: r["fall"]) if f else None


def s1(arm: str = "steer3") -> dict:
    out = {}
    for s in SEEDS:
        hard, easy = gaps(arm, s, HARD), gaps(arm, s, EASY)
        out[s] = {"held": max(hard.values()) <= MARGIN and max(easy.values()) <= MARGIN,
                  "worst_hard": max(hard.values()), "worst_hard_digits": max(hard, key=hard.get),
                  "worst_easy": max(easy.values()), "worst_easy_digits": max(easy, key=easy.get)}
    out["pooled"] = {"held": all(out[s]["held"] for s in SEEDS),
                     "worst_hard": max(out[s]["worst_hard"] for s in SEEDS),
                     "worst_easy": max(out[s]["worst_easy"] for s in SEEDS)}
    return out


def s2(arm: str = "steer3") -> dict:
    mine, fixed = kept_total(arm), kept_total("fixed")
    per = {s: (sum(e["kept"] for e in load(arm, s).of("build")),
               sum(e["kept"] for e in load("fixed", s).of("build"))) for s in SEEDS}
    return {"kept": mine, "fixed": fixed, "share": mine / fixed, "held": mine <= fixed / 2, "per_seed": per}


def trigger() -> list[dict]:
    """Arm (b)'s skipped lengths that fell more than 3 points in a round they were skipped."""
    return [r for r in falls("steer3") if r["fall"] > MARGIN]


def write_csv() -> None:
    rows = all_rows()
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{CSV}: {len(rows)} rows")


# ---------------------------------------------------------------- fetch

def fetch(arms=("steer3", "frozen")) -> None:
    """Download each finished steered run, copy its logs to results/runs/, check identity."""
    from gainpred.report import collect
    for arm in arms:
        for s in SEEDS:
            name = run_name(arm, s)
            subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "gainpred", f"runs/{name}", "runs/"],
                           capture_output=True, check=True)
    collect()
    bad = []
    for arm in arms:
        for s in SEEDS:
            ok, diffs = identity(arm, s)
            print(f"{run_name(arm, s)}: identity {'passed' if ok else 'not applicable' if ok is None else 'FAILED'}")
            bad += diffs
            for d in check_differences(arm, s):
                print("  round-1 check differs (reported, not a stop):", d)
    if bad:
        raise SystemExit("identity check failed; the analysis stops:\n" + "\n".join(bad))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--arms", default="steer3,frozen")
    a = ap.parse_args()
    if a.fetch:
        fetch(tuple(a.arms.split(",")))
    else:
        write_csv()
