"""Check 3 on addition (amendment 22): plan, launch and fetch. Inference only, no training.

    uv run python scripts/check3_addition.py              # list the runs
    uv run python scripts/check3_addition.py --launch     # spawn one Modal call per run not yet launched
    uv run python scripts/check3_addition.py --fetch      # download, merge -> results/check3_addition.csv
    uv run python scripts/check3_addition.py --three-cuts --launch   # the 26 cells under the three-cut rule
    uv run python scripts/check3_addition.py --three-cuts --fetch    # -> results/check3_addition_3cut.csv

The 26 three-cut cells (owner, 18 November 2026): the cells at 13-15 digits of seed 0-1 runs made
before the five-cut change, which the first redraw (under five cuts) could not reproduce. They are
redrawn under the three-cut rule those runs ran under, into new folders ``runs/check3-<run>-3cut/``
and new files; the first redraw's files are kept as they are.

The runs are every addition run P9b scores (results/p9b_cells.csv): learning seeds 0-1 and
confirming seeds 2-4, all arms, run f and the weak starting models. Every logged check cell of
those runs is redrawn (all rounds, lengths 8 to the round's maximum); ``in_p9b`` marks the
cells P9b itself scores. Logged files are never changed: each call writes its own folder
``runs/check3-<run>/``.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

CELLS = Path("results/p9b_cells.csv")
OUT_DIR = Path("results/check3")
CSV = Path("results/check3_addition.csv")
THREE_CUT_DIR = Path("results/check3_3cut")
THREE_CUT_CSV = Path("results/check3_addition_3cut.csv")


def three_cut_plan() -> list[tuple[str, str, int, str, list[tuple[int, int]]]]:
    """(name, original run, seed, teacher, cells) for the cells the five-cut redraw did not reproduce."""
    rows = [r for r in csv.DictReader(CSV.open()) if r["teacher"] != "sample_and_vote" and r["reproduced"] != "True"]
    out: dict[str, tuple] = {}
    for r in rows:
        name = f"check3-{r['run']}-3cut"
        out.setdefault(name, (name, r["run"], int(r["seed"]), r["teacher"], []))[4].append(
            (int(r["round"]), int(r["digits"])))
    return list(out.values())


def plan() -> list[tuple[str, str, int, str]]:
    """(check-3 run, original run, seed, teacher), in the order of p9b_cells.csv."""
    seen = {}
    for r in csv.DictReader(CELLS.open()):
        seen.setdefault(r["run"], (int(r["seed"]), r["teacher"]))
    return [(f"check3-{run}", run, s, t) for run, (s, t) in seen.items()]


def p9b_cells() -> set[tuple[str, int, int]]:
    return {(r["run"], int(r["round"]), int(r["digits"])) for r in csv.DictReader(CELLS.open())}


def launch(three_cuts: bool = False) -> None:
    if three_cuts:
        for name, run, _, _, cells in three_cut_plan():
            subprocess.run(["uv", "run", "python", "-m", "gainpred.launch", "check3", name, "--of", run,
                            "--three-cuts", "--only", ",".join(f"{k}:{d}" for k, d in cells)])
        return
    for name, run, _, _ in plan():
        subprocess.run(["uv", "run", "python", "-m", "gainpred.launch", "check3", name, "--of", run])


def fetch(three_cuts: bool = False) -> None:
    out = subprocess.run(["uv", "run", "python", "-m", "gainpred.launch", "status"], capture_output=True,
                         text=True).stdout
    states = {line.split()[0]: line.split(None, 1)[1].strip() for line in out.splitlines() if line.strip()}
    out_dir, out_csv = (THREE_CUT_DIR, THREE_CUT_CSV) if three_cuts else (OUT_DIR, CSV)
    out_dir.mkdir(exist_ok=True)
    in_p9b = p9b_cells()
    rows, waiting = [], []
    todo = [p[:4] for p in three_cut_plan()] if three_cuts else plan()
    for name, run, seed, teacher in todo:
        state = states.get(name, "not launched")
        if state != "done":
            waiting.append(f"{name}: {state[:80]}")
            if state in ("running", "not launched"):
                continue
        dest = Path("runs") / name
        subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "gainpred", f"runs/{name}", "runs/"],
                       capture_output=True)
        merged = "".join(p.read_text() for p in sorted(dest.glob("attempt_*/check3.jsonl")))
        (out_dir / f"{name}.jsonl").write_text(merged)
        if state != "done":
            continue  # a failed call's partial output is kept, not used
        for line in merged.splitlines():
            e = json.loads(line)
            if e["event"] != "cell":
                continue
            red, log = e["redrawn"], e["logged"]
            rows.append({"run": run, "seed": seed, "teacher": teacher, "round": e["round"], "digits": e["digits"],
                         "kept": e["kept"], "check_3": e["check_3"], "matches": e["matches"],
                         "unreadable_pieces": e["unreadable_pieces"], "pieces": e["pieces"],
                         "in_p9b": (run, e["round"], e["digits"]) in in_p9b,
                         "logged_kept": log["kept"], "logged_check_1": log["check_1"],
                         "logged_check_2b": log["check_2b"], "redrawn_check_1": red["check_1"],
                         "redrawn_check_2b": red["check_2b"],
                         "reproduced": (red["kept"], red["check_1"], red["check_2b"])
                         == (log["kept"], log["check_1"], log["check_2b"]),
                         "seconds": round(e["seconds"], 2)})
    rows.sort(key=lambda r: (r["seed"], r["run"], r["round"], r["digits"]))
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{out_csv}: {len(rows)} cells from {len({r['run'] for r in rows})} runs")
    for w_ in waiting:
        print("  waiting/failed:", w_)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--three-cuts", action="store_true")
    a = ap.parse_args()
    if a.launch:
        launch(a.three_cuts)
    elif a.fetch:
        fetch(a.three_cuts)
    elif a.three_cuts:
        for p in three_cut_plan():
            print(*p)
    else:
        for name, run, seed, teacher in plan():
            print(name, run, seed, teacher)
