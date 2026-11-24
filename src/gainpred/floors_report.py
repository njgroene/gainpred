"""Pilot report: reads ``results/pilot/runs/*/rounds.jsonl`` and applies the spec's fixed definitions.

Definitions (spec, "Definitions fixed now"; owner's answers 1-3, 6, 12)
=======================================================================
* Accuracy at a length: mean of the carry and no-carry groups of the test set (8-15: the study's
  set, 13-15 from the five-cut set; 16-39: the pilot's set); easy lengths: per length.
* **Learned**: after the floor's last round, at least 90% at every length of the floor, in at
  least two of three repeats (the answer key has two repeats: both).
* **Failed**: below 70% at one or more lengths of the floor, in at least two of three repeats.
* **Partial**: neither.
* **Eroded** (floor 1): a floor-1 length more than 3 points below that repeat's accuracy just
  before the floor's first round, in at least two of three repeats. Test 1 also reports the
  largest drop at any length from 3 up to the top of the floor below.
* **Seen coming**: the check's value in every repeat of a failing condition lies beyond its value
  in every repeat of the solid condition (lower confirmed share; higher novelty at some floor-1
  length). The looser comparison (mean lower or higher) is reported beside it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from gainpred.floors import FLOORS, PILOT_RESULTS, round_lengths, seconds_of
from gainpred.floors_cost import dollars

RUNS = PILOT_RESULTS / "runs"


def events(run: str) -> list[dict]:
    path = RUNS / run / "rounds.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def accuracy(evs: list[dict], floor: int | None, round_: int) -> dict[int, float]:
    """Percent per length after ``round_`` of ``floor`` (None: an under-trained floor-1 run)."""
    out: dict[int, float] = {}
    sets = {e["set"]: e["accuracy"] for e in evs if e["event"] == "eval" and e["round"] == round_
            and e.get("floor") == floor}
    for name in ("easy", "study", "floors", "study5"):  # study5 last: it replaces 13-15
        groups = defaultdict(list)
        for k, v in sets.get(name, {}).items():
            if k != "all":
                groups[int(k.split("/")[0])].append(v)
        out |= {d: 100 * sum(v) / len(v) for d, v in groups.items()}
    return dict(sorted(out.items()))


def finished_rounds(evs: list[dict], floor: int) -> list[int]:
    return sorted(e["round"] for e in evs if e["event"] == "round_done" and e.get("floor") == floor)


def verdict(runs: list[str], floor: int) -> dict:
    """Learned / failed / partial over the repeats that finished the floor's four rounds."""
    finals, eroded, worst_drop, learned_each = {}, {}, {}, {}
    for run in runs:
        evs = events(run)
        if 4 not in finished_rounds(evs, floor):
            continue
        after, before = accuracy(evs, floor, 4), accuracy(evs, floor, 0)
        finals[run] = after
        learned_each[run] = all(after.get(d, 0) >= 90 for d in FLOORS[floor])
        eroded[run] = any(before[d] - after.get(d, 0) > 3 for d in FLOORS[1] if d in before)
        drops = [(before[d] - after.get(d, 0), d) for d in before if d < FLOORS[floor].start]
        worst_drop[run] = max(drops) if drops else None
    n = len(finals)
    need = 2 if len(runs) >= 3 else len(runs)
    learned = sum(learned_each.values()) >= need
    failed = sum(any(a.get(d, 0) < 70 for d in FLOORS[floor]) for a in finals.values()) >= need
    return {"repeats": n, "learned": learned, "failed": failed and not learned,
            "verdict": "learned" if learned else "failed" if failed else "partial" if n else "not run",
            "learned_each": learned_each, "floor1_eroded": sum(eroded.values()) >= need, "eroded_each": eroded,
            "worst_drop": worst_drop, "finals": finals}


def seen_coming(runs: list[str]) -> list[dict]:
    """The two numbers, before floor-2 training, per repeat (from the run that computed them)."""
    out = []
    for run in runs:
        e = next((e for e in events(run) if e["event"] == "seen_coming"), None)
        if e:
            out.append({"run": run, "confirmed": {d: v["confirmed_share"] for d, v in e["confirmed_share"].items()},
                        "novelty": {d: v["novelty"] for d, v in e["floor1_novelty"].items()}})
    return out


def compare_seen(cond: list[dict], solid: list[dict]) -> dict:
    """Strict ("beyond every solid repeat") and loose (mean) comparisons."""
    if not cond or not solid:
        return {}
    res = {}
    for d in ("16", "17"):
        c, s = [x["confirmed"][d] for x in cond], [x["confirmed"][d] for x in solid]
        res[f"confirmed {d}"] = {"strict": max(c) < min(s), "loose": sum(c) / len(c) < sum(s) / len(s)}
    strict = loose = False
    for d in map(str, FLOORS[1]):
        c, s = [x["novelty"][d] or 0 for x in cond], [x["novelty"][d] or 0 for x in solid]
        strict |= min(c) > max(s)
        loose |= sum(c) / len(c) > sum(s) / len(s)
    res["floor-1 novelty"] = {"strict": strict, "loose": loose}
    return res


def fmt(v, digits=1):
    return "–" if v is None else f"{v:.{digits}f}"


def acc_table(runs: list[str], floor: int | None, round_: int, lengths) -> list[str]:
    head = "| Run | " + " | ".join(map(str, lengths)) + " |"
    lines = [head, "|" + " --- |" * (len(lengths) + 1)]
    for run in runs:
        a = accuracy(events(run), floor, round_)
        if a:
            lines.append(f"| `{run}` | " + " | ".join(fmt(a.get(d)) for d in lengths) + " |")
    return lines


def rounds_table(runs: list[str]) -> list[str]:
    lines = ["| Run | Floor | Round | Examples | Updates | Kept per new length | Kept right (%) | Minutes |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run in runs:
        evs = events(run)
        for e in evs:
            if e["event"] != "round_done":
                continue
            builds = [b for b in evs if b["event"] == "build" and b["floor"] == e["floor"] and b["round"] == e["round"]]
            kept = ", ".join(f"{b['digits']}: {b['kept']:,} of {b['drawn']:,}" for b in builds)
            right = ", ".join(fmt(100 * b["kept_right"], 2) if b["kept_right"] is not None else "–" for b in builds)
            lines.append(f"| `{run}` | {e['floor']} | {e['round']} | {e['examples']:,} | {e['updates']:,} | {kept} | "
                         f"{right} | {sum(e['seconds'].values()) / 60:.1f} |")
    return lines


def stage_dollars(stages: dict[str, list[str]]) -> list[str]:
    lines = ["| Stage | Runs | Logged GPU-hours | Dollars |", "| --- | --- | --- | --- |"]
    total = 0.0
    for stage, runs in stages.items():
        s = sum(seconds_of(r) for r in runs)
        total += s
        lines.append(f"| {stage} | {len(runs)} | {s / 3600:.2f} | {dollars(s):.2f} |")
    lines.append(f"| **Total** | | {total / 3600:.2f} | **{dollars(total):.2f}** |")
    return lines
