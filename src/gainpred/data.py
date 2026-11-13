"""Addition datasets: the test sets of P7 and the starting-training set of P1.

Run ``uv run python -m gainpred.data`` to write them to ``data/addition/``.
The files are not in git (see .gitignore): anyone can regenerate them, byte for
byte, from the seeds below. ``manifest.json`` records each file's seed, size and
SHA-256 fingerprint, so you can check that a copy (e.g. on Modal) is the same.

What gets generated, in this order
==================================
1. **Test sets first, each from its own seed** (P7). Generating them before any
   training data, from separate seeds, means no change to the training
   recipe can ever change the test sets.

   ================  =========================================================
   test_study        lengths 8-15; 1,000 per length *with* a carry across
                     cut 1 and 1,000 *without* (the question of the study:
                     can the model learn the problems "the no-carry rule" rejects?)
   test_reproduction lengths 3-15, 100 each, for comparison with Shin et al.
   test_easy         lengths 3-7, 1,000 each, for check 0 (pass mark 98%)
   ================  =========================================================

2. **Starting training** (P1): 50,000 problems per length, 3-7 digits, with
   every test problem excluded (P7: "excluded from every training pool").

Hard problems for the self-improvement rounds are drawn during the rounds
themselves, with ``draw_hard``, and are not written here.

How a problem is drawn
======================
Both numbers in a problem have the same number of digits (P1: "both numbers
3-7 digits", 50,000 "per length"). Each digit is drawn uniformly, except at
positions where a 0 is forbidden:

* every number: the first digit (otherwise "0123" would be a 3-digit number);
* hard problems (8-15 digits): also the first digit of every piece, under all
  three cuts of P4 (P2: "no piece starts with 0"). A piece like "0712" would
  be read by the model as an unfamiliar 4-character number with a leading 0.

Within a set, problems are unique: drawing the same pair twice would quietly
give it double weight.

Reproducibility
===============
Python promises that ``random.Random(seed).random()`` gives the same numbers in
every Python version, but *not* that ``randint`` or ``choice`` will. So digits
are derived from ``random()`` only (see ``_digit``), and the tests regenerate
everything and compare fingerprints (P9: "regenerating gives identical files").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Callable, Iterable
from pathlib import Path

from gainpred.cuts import ADDITION_CUTS, EASY_DIGITS, HARD_DIGITS, carry_crosses, piece_starts
from gainpred.formats import Addition, solve

# ----------------------------------------------------------------------------
# Settings. Changing any of these changes the datasets: record it in P10.
# ----------------------------------------------------------------------------
SEEDS = {
    "test_study": 1_001,
    "test_reproduction": 1_002,
    "test_easy": 1_003,
    "train_start": 2_001,
}
"""One fixed seed per set. Test sets have their own seeds (P7)."""

TEST_STUDY_PER_GROUP = 1_000  # P7
TEST_REPRODUCTION_PER_LENGTH = 100  # P7
TEST_REPRODUCTION_DIGITS = range(3, 16)  # P7: lengths 3-15
TEST_EASY_PER_LENGTH = 1_000  # P7
TRAIN_START_PER_LENGTH = 50_000  # P1

DEFAULT_OUT_DIR = Path("data/addition")

Record = dict[str, int | bool]
"""One line of a dataset file, e.g. {"a": 6133, "b": 8894, "answer": 15027, "digits": 4}."""


# ----------------------------------------------------------------------------
# Drawing single problems
# ----------------------------------------------------------------------------


def _digit(rng: random.Random, nonzero: bool) -> int:
    """A uniform digit (1-9 if ``nonzero``, else 0-9), using only rng.random().

    ``int(random() * n)`` is uniform over 0..n-1 up to a bias of about 1e-16,
    far too small to matter, and is stable across Python versions.
    """
    low = 1 if nonzero else 0
    return low + int(rng.random() * (10 - low))


def draw_number(rng: random.Random, digits: int, nonzero_at: Iterable[int]) -> int:
    """A ``digits``-digit number with no 0 at the given 0-based positions."""
    forbidden = set(nonzero_at)
    value = 0
    for position in range(digits):
        value = value * 10 + _digit(rng, nonzero=position in forbidden)
    return value


def hard_nonzero_positions(digits: int) -> frozenset[int]:
    """Positions that start a piece under any of the three cuts of this size."""
    return frozenset().union(*(piece_starts(cut) for cut in ADDITION_CUTS[digits]))


def draw_easy(rng: random.Random, digits: int) -> Addition:
    """An easy (3-7 digit) or reproduction problem: only the leading digit is nonzero."""
    return Addition(draw_number(rng, digits, {0}), draw_number(rng, digits, {0}))


def draw_hard(rng: random.Random, digits: int) -> Addition:
    """A hard (8-15 digit) problem with no piece starting with 0 under any cut (P2)."""
    nonzero = hard_nonzero_positions(digits)
    return Addition(draw_number(rng, digits, nonzero), draw_number(rng, digits, nonzero))


# ----------------------------------------------------------------------------
# Drawing sets
# ----------------------------------------------------------------------------


def draw_unique(
    draw: Callable[[], Addition], n: int, exclude: set[Addition], max_tries_factor: int = 100
) -> list[Addition]:
    """Draw ``n`` distinct problems, none of them in ``exclude``.

    ``exclude`` grows as we go (so later sets also avoid this one). The try cap
    turns an impossible request -- more problems than exist -- into an error
    instead of an endless loop.
    """
    chosen: list[Addition] = []
    for _ in range(n * max_tries_factor):
        if len(chosen) == n:
            break
        problem = draw()
        if problem not in exclude:
            exclude.add(problem)
            chosen.append(problem)
    if len(chosen) < n:
        raise RuntimeError(f"could only draw {len(chosen)} of {n} distinct problems")
    return chosen


def record(problem: Addition, **extra: int | bool) -> Record:
    return {"a": problem.a, "b": problem.b, "answer": solve(problem), "digits": problem.n_digits, **extra}


def make_test_study(exclude: set[Addition]) -> list[Record]:
    """P7: per length 8-15, 1,000 problems with a carry across cut 1, 1,000 without.

    Problems are drawn one at a time and put in whichever group they belong
    to, until both groups are full. Roughly half of random problems have a
    carry across a 2-piece cut, so this wastes few draws.
    """
    rng = random.Random(SEEDS["test_study"])
    records = []
    for digits in HARD_DIGITS:
        cut_1 = ADDITION_CUTS[digits][0]
        groups: dict[bool, list[Addition]] = {True: [], False: []}
        while any(len(g) < TEST_STUDY_PER_GROUP for g in groups.values()):
            problem = draw_hard(rng, digits)
            group = groups[carry_crosses(problem, cut_1)]
            if problem not in exclude and len(group) < TEST_STUDY_PER_GROUP:
                exclude.add(problem)
                group.append(problem)
        for carry in (False, True):
            records += [record(p, carry_across_cut1=carry) for p in groups[carry]]
    return records


def make_per_length(
    seed_name: str, digit_range: range, per_length: int, draw, exclude: set[Addition]
) -> list[Record]:
    """``per_length`` distinct problems for each length in ``digit_range``."""
    rng = random.Random(SEEDS[seed_name])
    records = []
    for digits in digit_range:
        problems = draw_unique(lambda: draw(rng, digits), per_length, exclude)
        records += [record(p) for p in problems]
    return records


def make_all() -> dict[str, list[Record]]:
    """Every addition set, in protocol order: test sets first, then training.

    A single ``exclude`` set is threaded through, so each set avoids every set
    made before it: test sets are disjoint from each other, and training
    avoids all of them.
    """
    exclude: set[Addition] = set()
    sets = {}
    sets["test_study"] = make_test_study(exclude)
    sets["test_reproduction"] = make_per_length(
        "test_reproduction", TEST_REPRODUCTION_DIGITS, TEST_REPRODUCTION_PER_LENGTH, draw_easy, exclude
    )
    sets["test_easy"] = make_per_length("test_easy", EASY_DIGITS, TEST_EASY_PER_LENGTH, draw_easy, exclude)
    sets["train_start"] = make_per_length("train_start", EASY_DIGITS, TRAIN_START_PER_LENGTH, draw_easy, exclude)
    return sets


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------


def to_jsonl(records: list[Record]) -> bytes:
    """One JSON object per line. Fixed key order, so the bytes are reproducible."""
    return "".join(json.dumps(r) + "\n" for r in records).encode()


def write_all(out_dir: Path = DEFAULT_OUT_DIR) -> dict:
    """Generate every set, write ``<name>.jsonl`` files and ``manifest.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, records in make_all().items():
        content = to_jsonl(records)
        (out_dir / f"{name}.jsonl").write_bytes(content)
        manifest[name] = {
            "seed": SEEDS[name],
            "rows": len(records),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(path: Path) -> list[Record]:
    """Read a .jsonl dataset back into a list of records."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def problems_of(records: list[Record]) -> list[Addition]:
    return [Addition(r["a"], r["b"]) for r in records]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the addition datasets.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    manifest = write_all(args.out)
    for name, info in manifest.items():
        print(f"{name:<18} {info['rows']:>8,} rows   sha256 {info['sha256'][:16]}...")
    print(f"written to {args.out}/")


if __name__ == "__main__":
    main()
