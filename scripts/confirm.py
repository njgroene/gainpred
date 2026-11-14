"""Launch the confirming runs (starting points 2-4) as their prerequisites finish.

Usage (idempotent; call it again whenever runs finish)::

    uv run python scripts/confirm.py            # launch everything unblocked
    uv run python scripts/confirm.py --dry-run  # only show what would launch

Runs are launched only from the commit tagged ``stage1-freeze`` or a descendant
(runbook section 3); a name is never launched twice (``gainpred.launch`` refuses).

What a confirming starting point consists of (runbook B, protocol P9b)
=====================================================================
Addition
    start-seed{s}-authors (10,000 updates, authors' setup) ->
    si-{nocarry,agreement,noselect}-seed{s}-authors (four rounds, five cuts at 13-15),
    si-vote-seed{s}-authors (four rounds, 10-minute building limit per length) ->
    after the no-carry run: si-noselect-after-nocarry-seed{s}-authors (run f: round 3
    with no selection from the no-carry run's round 2).
Weak starting models (matched on target accuracy, decision point 1)
    start-seed{s}-authors-s{n} for a grid of step counts -> pick by rule (closest
    within 5 points of 60% and of 97.5%; more step counts if none) ->
    si-nocarry-weak-s{n}-seed{s} (one round). Picking needs the finished grid and is
    done by ``--pick-weak``.
Longest run
    runs-start-seed{s}-u30000 (30,000 updates) -> runs-{edges_differ,pick_the_longer,
    join_at_the_cut}-seed{s}-u30000 and runs-join_at_the_cut-withheld-seed{s}-u30000;
    runs-start-seed{s}-longruns -> runs-join_at_the_cut-withheld-seed{s}-longruns (control arm).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

SEEDS = (2, 3, 4)
WEAK_GRID = (1000, 1250, 1500, 1750, 2000, 2125, 2200, 2250, 2300, 2375, 2500, 2625,
             2750, 2875, 3000, 3250, 3400, 3450, 3500, 3600, 3750, 4000)
WEAK_TARGETS = (60.0, 97.5)


def plan(s: int) -> list[tuple[str, list[str], list[str]]]:
    """(run name, launch arguments, runs that must be done first) for starting point ``s``."""
    start = f"start-seed{s}-authors"
    runs_start = f"runs-start-seed{s}-u30000"
    long_start = f"runs-start-seed{s}-longruns"
    p = [
        (start, ["main", start, "--seed", str(s), "--norm-eps", "1e-6", "--decay-fraction", "0.1"], []),
        (runs_start, ["runs-main", runs_start, "--seed", str(s), "--steps", "30000"], []),
        (long_start, ["runs-main", long_start, "--seed", str(s), "--steps", "30000", "--long-runs"], []),
    ]
    for arm, teacher in (("nocarry", "no_carry"), ("agreement", "agreement"), ("noselect", "no_selection")):
        name = f"si-{arm}-seed{s}-authors"
        p.append((name, ["rounds", name, "--start", start, "--teacher", teacher], [start]))
    vote = f"si-vote-seed{s}-authors"
    p.append((vote, ["rounds", vote, "--start", start, "--teacher", "sample_and_vote", "--build-seconds", "600"], [start]))
    f = f"si-noselect-after-nocarry-seed{s}-authors"
    nocarry = f"si-nocarry-seed{s}-authors"
    p.append((f, ["rounds", f, "--start", nocarry, "--start-round", "2", "--rounds", "1", "--teacher", "no_selection"],
              [nocarry]))
    for n in WEAK_GRID:
        name = f"start-seed{s}-authors-s{n}"
        p.append((name, ["main", name, "--seed", str(s), "--steps", str(n), "--norm-eps", "1e-6",
                         "--decay-fraction", "0.1"], []))
    for t in ("edges_differ", "pick_the_longer", "join_at_the_cut"):
        name = f"runs-{t}-seed{s}-u30000"
        p.append((name, ["runs-rounds", name, "--start", runs_start, "--teacher", t], [runs_start]))
    w = f"runs-join_at_the_cut-withheld-seed{s}-u30000"
    p.append((w, ["runs-rounds", w, "--start", runs_start, "--teacher", "join_at_the_cut", "--withheld"], [runs_start]))
    c = f"runs-join_at_the_cut-withheld-seed{s}-longruns"
    p.append((c, ["runs-rounds", c, "--start", long_start, "--teacher", "join_at_the_cut", "--withheld", "--long-runs"],
              [long_start]))
    return p


def states() -> dict[str, str]:
    out = subprocess.run(["uv", "run", "python", "-m", "gainpred.launch", "status"],
                         capture_output=True, text=True, check=True).stdout
    return {line.split()[0]: line.split(None, 1)[1].strip() for line in out.splitlines() if line.strip()}


def frozen_ok() -> bool:
    """HEAD is the stage1-freeze tag or a descendant of it."""
    tag = subprocess.run(["git", "rev-parse", "-q", "--verify", "stage1-freeze^{commit}"],
                         capture_output=True, text=True).stdout.strip()
    if not tag:
        return False
    return subprocess.run(["git", "merge-base", "--is-ancestor", tag, "HEAD"]).returncode == 0


def easy_accuracy(run: str) -> float | None:
    log = Path("runs") / run / "log.jsonl"
    if not log.exists():
        log.parent.mkdir(parents=True, exist_ok=True)  # else the download writes a file named after the run
        subprocess.run(["uv", "run", "modal", "volume", "get", "gainpred", f"runs/{run}/log.jsonl",
                        f"runs/{run}/"], capture_output=True)
    if not log.exists():
        return None
    evals = [json.loads(line) for line in log.read_text().splitlines() if '"set": "easy"' in line]
    return 100 * evals[-1]["accuracy"]["all"] if evals else None


def pick_weak(s: int, st: dict[str, str], dry: bool) -> None:
    """Runbook rule: per target, the step count closest within 5 points (ties: fewer steps)."""
    names = [f"start-seed{s}-authors-s{n}" for n in WEAK_GRID]
    if not all(st.get(n) == "done" for n in names):
        print(f"seed {s}: weak grid not finished")
        return
    acc = {n: easy_accuracy(f"start-seed{s}-authors-s{n}") for n in WEAK_GRID}
    print(f"seed {s} weak grid: " + ", ".join(f"{n}:{a:.1f}" for n, a in acc.items() if a is not None))
    for target in WEAK_TARGETS:
        close = sorted((abs(a - target), n) for n, a in acc.items() if a is not None and abs(a - target) <= 5)
        if not close:
            print(f"  target {target}%: NONE within 5 points -> train more step counts (runbook rule)")
            continue
        n = close[0][1]
        name = f"si-nocarry-weak-s{n}-seed{s}"
        print(f"  target {target}%: {n} steps ({acc[n]:.1f}%) -> {name}")
        if not dry and name not in st:
            subprocess.run(["uv", "run", "python", "-m", "gainpred.launch", "rounds", name, "--start",
                            f"start-seed{s}-authors-s{n}", "--teacher", "no_carry", "--rounds", "1"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pick-weak", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not frozen_ok():
        raise SystemExit("HEAD is not stage1-freeze or a descendant: confirming runs need the freeze (runbook)")
    st = states()
    for s in SEEDS:
        for name, launch_args, needs in plan(s):
            if name in st:
                continue
            if all(st.get(n) == "done" for n in needs):
                print(f"launch {name}" + (" (dry run)" if args.dry_run else ""))
                if not args.dry_run:
                    subprocess.run(["uv", "run", "python", "-m", "gainpred.launch", *launch_args])
            else:
                print(f"wait   {name} (needs {', '.join(n for n in needs if st.get(n) != 'done')})")
        if args.pick_weak:
            pick_weak(s, st, args.dry_run)


if __name__ == "__main__":
    main()
