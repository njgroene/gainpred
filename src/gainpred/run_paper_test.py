"""The paper test for longest run (P9): every teacher run with a perfect solver in place of the model.

Usage::

    uv run python -m gainpred.run_paper_test     # 10,000 problems per size -> results/paper_test_longest_run.md

What it shows
=============
With a perfect solver every piece answer is right, so any wrong built answer
comes from the combining rule itself: a run that continues across the join.
Per teacher and size 12-20, and separately for the two kinds of hard problem
(random join, run across the join; P6):

* **kept** -- share of hard problems the teacher keeps an answer for;
* **right** -- share of kept answers that are right;
* **check 3** -- share of kept answers identical to one of the piece answers
  (P8; "hands back" if 100%).

Expected, if the code matches the protocol:

* pick the longer (= no selection): kept 100%; right about 94% on random joins
  (wrong only when an edge run merges into a longer or earlier run), and
  low on run-across problems; check 3 100% by construction;
* the edges-differ rule: kept about 90% of random joins (edges equal one time
  in ten) and 0% of run-across problems, all right; check 3 100%;
* join at the cut: kept 100%, right 100%; check 3 below 100%, because the run
  across the join is a new answer no piece gave.

Comparison with Shin et al.
===========================
Their Table 7 audits one round of max-run-length labels: unfiltered, length
at most 20, rule errors 5.9%; filtered, kept 90.0% and rule errors 0.0%.
Their parents are random concatenations, so the comparable rows here are the
*random join* ones. Their Section C.7 states that 66.2% of 11-83-symbol test
strings have no run crossing a block boundary when cut into ceil(L/10) blocks.
That setting (up to 9 blocks, lengths to 83) is not ours, but it does test
the uniform-digit assumption of ``gainpred.run_data``, so it is recomputed by
simulation in ``boundary_share``.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from gainpred.checks import check_3
from gainpred.formats import longest_run
from gainpred.run_data import HARD_LENGTHS, RANDOM_JOIN, RUN_ACROSS, draw_hard, draw_string
from gainpred.run_teachers import RUN_TEACHER_NAMES, RUN_TEACHERS, PerfectRunSolver

SEED = 3_101  # its own seed: these problems are never trained or tested on
OUT = Path("results/paper_test_longest_run.md")
KINDS = ("all", RANDOM_JOIN, RUN_ACROSS)

Table = dict[str, dict[str, dict[int, dict[str, float | None]]]]


def draw(length: int, n: int, seed: int = SEED) -> list:
    rng = random.Random(seed * 100 + length)
    return [draw_hard(rng, length) for _ in range(n)]


def paper_test(n: int = 10_000, seed: int = SEED) -> Table:
    """{teacher: {kind: {length: {"kept", "right", "check_3", "n"}}}}; kind is "all", "random_join" or "run_across"."""
    solver = PerfectRunSolver()
    table: Table = {name: {kind: {} for kind in KINDS} for name in RUN_TEACHERS}
    for length in HARD_LENGTHS:
        problems = draw(length, n, seed)
        for name, teacher in RUN_TEACHERS.items():
            decisions = teacher(problems, solver)
            for kind in KINDS:
                ds = [d for d in decisions if kind == "all" or d.problem.kind == kind]
                kept = [d for d in ds if d.kept is not None]
                right = sum(d.kept == longest_run(d.problem.string) for d in kept)
                table[name][kind][length] = {
                    "n": len(ds),
                    "kept": len(kept) / len(ds),
                    "right": right / len(kept) if kept else None,
                    "check_3": check_3((d.kept, d.pieces) for d in ds),
                }
    return table


def boundary_share(per_length: int = 2_000, seed: int = SEED, lengths: range = range(11, 84)) -> float:
    """Shin et al., Section C.7: share of strings with no run crossing a block boundary.

    Uniform-digit strings of each length 11-83, cut into ceil(L/10) blocks of
    near-equal length; a run crosses a boundary iff the symbols either side of
    it are equal. Mean over lengths (their test set has 100 per length).
    """
    rng = random.Random(seed)
    shares = []
    for length in lengths:
        blocks = math.ceil(length / 10)
        bounds = [round(i * length / blocks) for i in range(1, blocks)]
        clean = 0
        for _ in range(per_length):
            s = draw_string(rng, length)
            clean += all(s[b - 1] != s[b] for b in bounds)
        shares.append(clean / per_length)
    return sum(shares) / len(shares)


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _mean(table: Table, name: str, kind: str, key: str) -> float | None:
    """Mean over lengths 12-20, weighting each length by its count (right: by kept count)."""
    cells = table[name][kind]
    if key == "kept":
        return sum(c["kept"] * c["n"] for c in cells.values()) / sum(c["n"] for c in cells.values())
    weights = [(c[key], c["kept"] * c["n"]) for c in cells.values() if c[key] is not None]
    total = sum(w for _, w in weights)
    return sum(v * w for v, w in weights) / total if total else None


def to_markdown(table: Table, n: int, boundary: float) -> str:
    sizes = list(HARD_LENGTHS)
    lines = [
        "# Paper test, longest run: teachers with a perfect solver",
        "",
        f"Generated by `uv run python -m gainpred.run_paper_test` ({n:,} hard strings per length, seed {SEED};",
        "about half random joins and half runs across the join, P6). Every piece answer is right, so wrong built",
        "answers come only from a run continuing across the join. **kept** = share with a kept answer; **right** =",
        "share of kept answers that are right; **check 3** = share of kept answers identical to a piece answer (P8).",
        "",
    ]
    for kind, title in (("all", "All hard problems"), (RANDOM_JOIN, "Random joins"), (RUN_ACROSS, "Runs across the join")):
        lines += [f"## {title}", "",
                  "| Teacher | | " + " | ".join(str(s) for s in sizes) + " | 12–20 |",
                  "| --- | --- | " + " | ".join("---" for _ in sizes) + " | --- |"]
        for name in RUN_TEACHERS:
            cells = table[name][kind]
            label = RUN_TEACHER_NAMES[name].capitalize() + (" (= pick the longer)" if name == "no_selection" else "")
            for i, key in enumerate(("kept", "right", "check_3")):
                head = label if i == 0 else ""
                lines.append(f"| {head} | {key.replace('_', ' ')} | " + " | ".join(_pct(cells[s][key]) for s in sizes)
                             + f" | {_pct(_mean(table, name, kind, key))} |")
        lines.append("")
    rule_errors = 1 - _mean(table, "pick_the_longer", RANDOM_JOIN, "right")
    filtered_kept = _mean(table, "edges_differ", RANDOM_JOIN, "kept")
    filtered_errors = 1 - _mean(table, "edges_differ", RANDOM_JOIN, "right")
    lines += [
        "## Comparison with Shin et al. (2026)",
        "",
        "Their parents are random concatenations of two seed-length strings, so the comparable rows are our random joins.",
        "",
        "| | Shin et al. | Ours |",
        "| --- | --- | --- |",
        f"| Table 7, unfiltered, length ≤ 20: rule errors | 5.9 | {_pct(rule_errors)} (pick the longer, random joins, 12–20) |",
        f"| Table 7, filtered, all rounds: kept | 90.0 | {_pct(filtered_kept)} (the edges-differ rule, random joins) |",
        f"| Table 7, filtered: rule errors | 0.0 | {_pct(filtered_errors)} |",
        f"| Section C.7: 11–83-symbol strings with no run crossing a block boundary (⌈L/10⌉ blocks) | 66.2 | "
        f"{_pct(boundary)} (simulated, uniform digits; their setting, not ours) |",
        "",
        "Not comparable: their Table 7 row \"length ≤ 83\" (we stop at 20); and their wrong-label rates with the filter",
        "(0.2–1.2%), which come from wrong component answers that a perfect solver cannot show. Their kept 90.0%",
        "and the 66.2% both match independent uniform digits, the assumption of `gainpred.run_data`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", type=int, default=10_000, help="hard strings per length")
    args = parser.parse_args()
    markdown = to_markdown(paper_test(args.n), args.n, boundary_share())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(markdown)
    print(markdown)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
