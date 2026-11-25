"""Cost of pilot calls, estimated from measured timings (``results/pilot/runs/timing/timing.json``).

How a round's seconds are estimated
===================================
* **Generation** (building, checks, testing). The model re-reads the whole sequence for every
  character it writes (no cache), so answering a p-digit sum costs about
  ``sum over the p + 2 written characters of (prompt + written so far)`` token positions, with a
  2p + 14 character prompt (``work``). The measurement gives seconds per token position at 23, 31
  and 39 digits; between and below those, the nearest measured rate is used.
* **Building** draws ``kept / keep rate`` problems (keep rates with right pieces: agreement rule
  0.34, no-carry rule 0.39; ``weak`` multiplies the problems drawn for weak floors).
* **Training**: updates (owner's change D) x seconds per update at the round's widest example
  (3n + 16 characters for an n-digit length), interpolated between the measured widths 85, 109, 133,
  proportional to width below 85.
* **Overhead**: 3 minutes per call (start-up, loading, saving), as observed on earlier runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from gainpred.floors import FLOORS, SPLITS, round_lengths, updates_for

TIMING = Path("results/pilot/runs/timing/timing.json")
KEEP = {"agreement": 0.34, "no_carry": 0.39, "answer_key": 1.0, "direct": 0.95}
FLOOR1_NOVELTY_SECONDS = 64.0  # the study's no-carry checks at 8-15 (results/writeup/cost.md, round 4)
OVERHEAD_SECONDS = 180.0
DOLLARS_PER_HOUR = 1.10
EVAL_PROBLEMS = {d: 1_000 for d in range(3, 8)} | {d: 2_000 for d in range(8, 16)} | {d: 2_000 for d in range(13, 16)}


def work(p: int) -> int:
    return sum(2 * p + 14 + t for t in range(p + 2))


def split_work(n: int, splits: tuple[int, ...] = (0, 1, 2)) -> int:
    return sum(work(w) for s in splits for w in SPLITS[n][s])


class Model:
    def __init__(self, timing: dict, factor: float = 1.0):
        t = {k: {int(d): v for d, v in timing[k].items()} for k in ("train", "build", "direct")}
        self.train_at = {3 * d + 16: s for d, s in t["train"].items()}  # width -> seconds per update
        # seconds per token position, from building (six pieces) and direct answers
        # Only the 23-digit generation timing is usable: at 31 and 39 digits the solid model has never seen
        # pieces that long and stops early, so its answers were short and cheap. The 23-digit rate (1.4 us
        # per token position) matches the study's own building logs (1.40-1.49 us), and is used throughout.
        r23 = (t["build"][23] / split_work(23) + t["direct"][23] / work(23)) / 2
        self.rate = {23: r23}
        self.factor = factor

    def per_position(self, digits: int) -> float:
        nearest = min(self.rate, key=lambda d: abs(d - digits))
        return self.rate[nearest]

    def gen(self, problems: float, positions_each: float, digits: int) -> float:
        return problems * positions_each * self.per_position(max(digits, 16)) * self.factor

    def update_seconds(self, width: int) -> float:
        ws = sorted(self.train_at)
        if width <= ws[0]:
            return self.train_at[ws[0]] * width / ws[0] * self.factor
        for lo, hi in zip(ws, ws[1:]):
            if width <= hi:
                return (self.train_at[lo] + (self.train_at[hi] - self.train_at[lo]) * (width - lo) / (hi - lo)) * \
                    self.factor
        return self.train_at[ws[-1]] * width / ws[-1] * self.factor

    def test(self, max_digits: int) -> float:
        problems = dict(EVAL_PROBLEMS) | {d: 2_000 for d in range(16, max_digits + 1)}
        return sum(self.gen(n, work(d), d) for d, n in problems.items())

    def checks(self, digits: int, sample: int = 2_000) -> float:
        return self.gen(sample, split_work(digits) + work(digits), digits)

    def round(self, floor: int, round_: int, teacher: str, weak: float = 1.0) -> dict[str, float]:
        new = round_lengths(floor, round_)
        splits = {"agreement": (0, 1, 2), "no_carry": (0,), "answer_key": (), "direct": ()}[teacher]
        build = sum(self.gen(10_000 / KEEP[teacher] * (weak if teacher != "answer_key" else 1.0),
                             split_work(d, splits), d) for d in new) if splits else 0.0
        if teacher == "direct":  # amendment 27: one whole answer per problem, almost all readable
            build = sum(self.gen(10_000 / KEEP["direct"], work(d), d) for d in new)
        examples = 20_000 * round_ + 80_000 * (floor - 1) + 25_000  # new + stored + lower floors + replay
        updates = updates_for(examples)
        return {"checks": sum(self.checks(d) for d in new) + FLOOR1_NOVELTY_SECONDS * self.factor,
                "build": build + 7.0, "train": updates * self.update_seconds(3 * new[1] + 16),
                "test": self.test(new[1]), "examples": examples, "updates": updates}

    def call(self, floor: int, rounds: range, teacher: str, weak: float = 1.0, seen: bool = False) -> float:
        """Seconds of one pilot_floor call, overhead included."""
        total = OVERHEAD_SECONDS
        if rounds.start == 1:
            total += self.test(FLOORS[floor].start - 1)
            if seen:
                total += sum(self.checks(d) for d in (16, 17)) + FLOOR1_NOVELTY_SECONDS * self.factor
            if floor >= 3:
                total += sum(self.checks(d) for d in FLOORS[floor - 1])
        for k in rounds:
            r = self.round(floor, k, teacher, weak)
            total += r["checks"] + r["build"] + r["train"] + r["test"]
        return total


def undertrained_seconds(steps: int) -> float:
    """Floor 1 again with the no-carry rule: the study's rounds (results/writeup/cost.md: build
    342 s, checks 136 s, replay 28 s over four rounds, train 730 s per 3,000 updates a round)."""
    return OVERHEAD_SECONDS + 342 + 136 + 28 + 730 * steps / 3_000


def dollars(seconds: float) -> float:
    return seconds / 3600 * DOLLARS_PER_HOUR


def load(factor: float = 1.0) -> Model:
    return Model(json.loads(TIMING.read_text()), factor)


def stage_table(m: Model) -> tuple[list[list], float]:
    """Rows: stage, runs, rounds per run, minutes per round (checks, build, train, test), GPU-hours, dollars."""
    rows, total = [], 0.0

    def row(stage, runs, calls, rounds_per_run, mean_round):
        nonlocal total
        seconds = sum(calls)
        total += seconds
        rows.append([stage, runs, rounds_per_run, *(f"{mean_round[k] / 60:.1f}" for k in ("checks", "build", "train",
                                                                                          "test")),
                     f"{seconds / 3600:.1f}", f"{dollars(seconds):.1f}"])

    def mean(floor, rounds, teacher, weak=1.0):
        rs = [m.round(floor, k, teacher, weak) for k in rounds]
        return {k: sum(r[k] for r in rs) / len(rs) for k in ("checks", "build", "train", "test")}

    row("A (answer key, floor 2 rounds 1-2, repeats 0-1) + rehearsal set", 2,
        [m.call(2, range(1, 3), "answer_key")] * 2 + [OVERHEAD_SECONDS + 168], 2, mean(2, range(1, 3), "answer_key"))
    row("B (agreement, floor 2; answer key rounds 3-4 continued)", "3 + 2",
        [m.call(2, range(1, 5), "agreement", seen=True)] * 3 + [m.call(2, range(3, 5), "answer_key")] * 2, 4,
        mean(2, range(1, 5), "agreement"))
    row("B2 (agreement, floor 3)", 3, [m.call(3, range(1, 5), "agreement")] * 3, 4, mean(3, range(1, 5), "agreement"))
    row("B3 (agreement, floor 4)", 3, [m.call(4, range(1, 5), "agreement")] * 3, 4, mean(4, range(1, 5), "agreement"))
    c = ([m.call(2, range(1, 5), "no_carry")] * 3                                      # (i)
         + [m.call(2, range(1, 5), "agreement", 1.5, seen=True), m.call(2, range(1, 5), "no_carry")] * 3  # (ii)
         + [undertrained_seconds(s) for s in (500, 1_000, 1_500)]                      # (iii)
         + [m.call(2, range(1, 5), "agreement", 1.5, seen=True), m.call(2, range(1, 5), "no_carry", 1.2)] * 3  # (iv)
         + [m.call(2, range(1, 5), "answer_key")] * 3)                                 # (v)
    row("C (i-v)", "15 + 3 short", c, 4, mean(2, range(1, 5), "agreement", 1.5))
    return rows, total


def write_cost_md(m: Model, path: Path = Path("results/pilot/cost.md")) -> str:
    rows, total = stage_table(m)
    head = ["Stage", "Runs", "Rounds per run", "Checks (min/round)", "Building (min/round)", "Training (min/round)",
            "Testing (min/round)", "GPU-hours", "Dollars"]
    lines = ["# Pilot cost, from the timing measurement", "",
             "Generated by `gainpred.floors_cost` from `results/pilot/runs/timing/timing.json`; owner's changes A-D "
             "applied (updates per round = 7 x examples / 256, at least 3,000). Minutes per round are the mean over "
             "a run's rounds, for the stage's main teacher. A10 at $1.10 an hour; 3 minutes overhead per call.", "",
             "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    lines += [f"| **Total** | | | | | | | **{total / 3600:.1f}** | **{dollars(total):.1f}** |", "",
              "Seconds per update measured: " + ", ".join(f"width {w}: {s:.3f}" for w, s in sorted(m.train_at.items()))
              + ". Generation, seconds per million token positions: "
              + ", ".join(f"{d} digits: {r * 1e6:.1f}" for d, r in sorted(m.rate.items())) + ".", ""]
    text = "\n".join(lines)
    path.write_text(text)
    return text
