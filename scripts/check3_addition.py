"""Check 3 on addition (amendment 22): plan, launch and fetch. Inference only, no training.

    uv run python scripts/check3_addition.py              # list the runs
    uv run python scripts/check3_addition.py --launch     # spawn one Modal call per run not yet launched
    uv run python scripts/check3_addition.py --fetch      # download, merge -> results/check3_addition.csv

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


def plan() -> list[tuple[str, str, int, str]]:
    """(check-3 run, original run, seed, teacher), in the order of p9b_cells.csv."""
    seen = {}
    for r in csv.DictReader(CELLS.open()):
        seen.setdefault(r["run"], (int(r["seed"]), r["teacher"]))
    return [(f"check3-{run}", run, s, t) for run, (s, t) in seen.items()]


def p9b_cells() -> set[tuple[str, int, int]]:
    return {(r["run"], int(r["round"]), int(r["digits"])) for r in csv.DictReader(CELLS.open())}


def launch() -> None:
    for name, run, _, _ in plan():
        subprocess.run(["uv", "run", "python", "-m", "gainpred.launch", "check3", name, "--of", run])


def fetch() -> None:
    out = subprocess.run(["uv", "run", "python", "-m", "gainpred.launch", "status"], capture_output=True,
                         text=True).stdout
    states = {line.split()[0]: line.split(None, 1)[1].strip() for line in out.splitlines() if line.strip()}
    OUT_DIR.mkdir(exist_ok=True)
    in_p9b = p9b_cells()
    rows, waiting = [], []
    for name, run, seed, teacher in plan():
        state = states.get(name, "not launched")
        if state != "done":
            waiting.append(f"{name}: {state[:80]}")
            if state in ("running", "not launched"):
                continue
        dest = Path("runs") / name
        subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "gainpred", f"runs/{name}", "runs/"],
                       capture_output=True)
        merged = "".join(p.read_text() for p in sorted(dest.glob("attempt_*/check3.jsonl")))
        (OUT_DIR / f"{name}.jsonl").write_text(merged)
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
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{CSV}: {len(rows)} cells from {len({r['run'] for r in rows})} runs")
    for w_ in waiting:
        print("  waiting/failed:", w_)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.launch:
        launch()
    elif a.fetch:
        fetch()
    else:
        for name, run, seed, teacher in plan():
            print(name, run, seed, teacher)
