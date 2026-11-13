"""How a model gets hard problems wrong, not just how often.

Usage::

    uv run python -m gainpred.errors start-seed1-stable              # final checkpoint, 8 and 9 digits
    uv run python -m gainpred.errors start-seed0-stable --step 4000 --digits 8

Writes ``results/errors/<run>_step<N>.json`` and prints a summary.

Why look at errors
==================
Accuracy says *whether* the answer is exactly right. Two models at 0% can be
nowhere near or one digit off, and which one matters: a model that is almost
right has a working method that needs fine-tuning, while one that is far off
has no method for that length at all. The profile below is computed on the
``test_study`` problems of each requested length (both carry groups).

What is measured, per length
============================
``right`` / ``unreadable``
    Exactly right answers; answers that cannot be read (P2), e.g. the model
    never wrote an end token.
``length_minus_true``
    For readable wrong answers: how many digits longer (+) or shorter (-) the
    answer is than the true one. A model that only ever saw answers up to 8
    characters long may systematically stop one digit early.
``leading_digits_right``
    For readable wrong answers: how many digits, from the left, match the
    true answer before the first mistake. Answers are written most
    significant digit first, so this is how far the model gets before it
    goes astray.
``wrong_place``
    For wrong answers of the *right* length: how often each place is wrong,
    counting from the right (0 = units, 1 = tens, ...). Shows whether
    mistakes are spread out or concentrated in one place.
``most_common_difference``
    For wrong answers of the right length: the most frequent values of
    (model answer - true answer). Small multiples of 10 mean "tens digit off".

First findings (13 November 2026, final checkpoints of starting training)
=========================================================================
Seed 1 answers *every* 8-digit problem with one digit too few, with only the
first 2-3 digits right: its sense of number length stops at the 7 digits it
trained on. Seed 0 writes the right length on all 2,000 problems, gets the
first 5-7 digits right and errs mostly by +-10 to +-30 (the tens digit): its
sense of length stretches one digit further. At 9 digits both write one digit
too few. See ``results/README.md``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from gainpred.data import load, problems_of
from gainpred.formats import Addition, Unreadable, solve
from gainpred.model import pick_device
from gainpred.report import RESULTS_DIR
from gainpred.train import DATA_DIR, answer_greedily, load_model

RUNS_DIR = Path("runs")


def error_profile(problems: list[Addition], answers: list) -> dict:
    """Summarise how ``answers`` (ints or Unreadable) differ from the true sums."""
    length_diff, leading, places, differences = Counter(), Counter(), Counter(), Counter()
    right = unreadable = 0
    for problem, answer in zip(problems, answers, strict=True):
        truth = solve(problem)
        if isinstance(answer, Unreadable):
            unreadable += 1
            continue
        if answer == truth:
            right += 1
            continue
        a, t = str(answer), str(truth)
        length_diff[len(a) - len(t)] += 1
        k = 0
        while k < min(len(a), len(t)) and a[k] == t[k]:
            k += 1
        leading[k] += 1
        if len(a) == len(t):
            differences[answer - truth] += 1
            for place, (x, y) in enumerate(zip(reversed(a), reversed(t))):
                if x != y:
                    places[place] += 1

    def ordered(c: Counter) -> dict[str, int]:
        return {str(k): v for k, v in sorted(c.items())}

    return {
        "problems": len(problems),
        "right": right,
        "unreadable": unreadable,
        "length_minus_true": ordered(length_diff),
        "leading_digits_right": ordered(leading),
        "wrong_place": ordered(places),
        "most_common_difference": {str(k): v for k, v in differences.most_common(8)},
    }


def profile_checkpoint(checkpoint: Path, digits: list[int], data_dir: Path = DATA_DIR) -> dict[str, dict]:
    device = pick_device()
    model = load_model(checkpoint, device).eval()
    records = load(data_dir / "test_study.jsonl")
    out = {}
    for d in digits:
        problems = problems_of([r for r in records if r["digits"] == d])
        out[str(d)] = error_profile(problems, answer_greedily(model, problems, 500, device))
    return out


def describe(digits: str, p: dict) -> str:
    n = p["problems"]
    return "\n".join([
        f"{digits} digits: {p['right']}/{n} right ({100 * p['right'] / n:.1f}%), {p['unreadable']} unreadable",
        f"  wrong answers, length minus true length: {p['length_minus_true']}",
        f"  wrong answers, leading digits right:     {p['leading_digits_right']}",
        f"  right-length wrong answers, wrong place (0 = units): {p['wrong_place']}",
        f"  most common (model - truth): {p['most_common_difference']}",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run")
    parser.add_argument("--step", type=int, help="checkpoint step (default: the last one)")
    parser.add_argument("--digits", type=int, nargs="+", default=[8, 9])
    args = parser.parse_args()
    checkpoints = sorted((RUNS_DIR / args.run / "checkpoints").glob("step_*.pt"))
    checkpoint = RUNS_DIR / args.run / "checkpoints" / f"step_{args.step:06d}.pt" if args.step else checkpoints[-1]
    step = int(checkpoint.stem.removeprefix("step_"))
    profiles = profile_checkpoint(checkpoint, args.digits)
    out = RESULTS_DIR / "errors" / f"{args.run}_step{step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": args.run, "step": step, "by_digits": profiles}, indent=2) + "\n")
    print(f"{args.run}, step {step}")
    for d, p in profiles.items():
        print(describe(d, p))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
