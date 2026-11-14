"""Self-improvement rounds for addition (P3): the model trains on answers it built itself.

Usage::

    uv run python -m gainpred.rounds --run si-nocarry-seed1 --start start-seed1-stable --teacher no_carry --rounds 1
    uv run python -m gainpred.rounds ... --kept 500 --replay 300 --steps 300 --eval-per-length 100   # quick check

On Modal: ``uv run modal run --detach modal_app.py::rounds --run ... --start ... --teacher ...``.

One round, step by step
=======================
Round r (1-4) has a maximum length of 9, 11, 13 or 15 digits (P3).

1. **Draw and build.** For every length from 8 to the maximum, draw fresh hard
   problems and let the teacher (``gainpred.teachers``) build and filter
   answers with the *current* model, until 10,000 answers per length are kept
   or 20 x 10,000 problems have been drawn (P3). Nobody looks at the right
   answers; we only *audit* them afterwards, for the report.
2. **Replay.** 5,000 new easy problems per easy length (3-7 digits), answered
   greedily by the current model, unchecked (Shin et al., Table 4: 25,000
   replay examples). They keep the model from forgetting easy addition.
3. **Checks** (P8) on 2,000 fresh problems at *every* length from 8 to the
   maximum, with the model before this round's training: is the teacher
   saying anything new (check 1), does the agreement rule keep an answer
   (check 2), does another cut confirm the kept answers (2b), and the
   teachable share.
4. **Train** 3,000 updates on kept answers + replay, continuing from the
   current weights. Fresh AdamW state each round; learning rate constant at
   the peak, then a cosine decay to 1% over the last 1,000 updates (P10, from
   Lee et al.'s code). Batch order seeded per starting point and round.
5. **Evaluate** on every test group: easy 3-7 digits, and 8-15 digits by
   carry across cut 1.

Before round 1 the starting model is evaluated the same way: P10, "measure
every run's own starting accuracy on all test groups before round 1".

Where things go
===============
``runs/<run>/``
    ``config.json``  -- settings, the starting run, dataset fingerprints
    ``rounds.jsonl`` -- one line per event: start eval, per-length build
                        statistics, checks, replay, round eval, timings
    ``round_<r>/``   -- the round's training run (``log.jsonl``,
                        ``config.json``, final checkpoint), as in starting training

Random numbers
==============
Every draw has its own seeded generator, derived from the starting point
(``seed``), the round and the length, so a run is reproducible and two
teachers given the same seed see the *same* candidate stream: they differ
only in what they keep. Rounds never reuse a test problem or a
starting-training problem.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from gainpred.checks import run_checks
from gainpred.cuts import EASY_DIGITS, HARD_DIGITS
from gainpred.data import DEFAULT_OUT_DIR as DATA_DIR
from gainpred.data import draw_easy, draw_hard, load, problems_of
from gainpred.formats import Addition, solve
from gainpred.model import Transformer, pick_device
from gainpred.teachers import TEACHERS, ModelSolver, Solver, Teacher
from gainpred.train import (
    RUNS_DIR,
    EvalSet,
    Stopwatch,
    TrainConfig,
    by_length,
    by_length_and_carry,
    device_name,
    evaluate,
    load_model,
    select,
    train,
)

MAX_DIGITS_BY_ROUND = (9, 11, 13, 15)  # P3


@dataclass(frozen=True)
class RoundConfig:
    teacher: str = "no_carry"
    rounds: int = 4
    seed: int = 0
    """Starting point (P9); taken from the starting run."""
    kept_per_length: int = 10_000  # P1, P3
    draw_cap_factor: int = 20  # P3: candidates drawn at most 20 x the quota
    replay_per_length: int = 5_000  # P1
    check_sample: int = 2_000  # P8
    steps: int = 3_000  # P1
    decay_steps: int = 1_000  # P10 (Lee et al. code)
    batch_size: int = 256
    eval_per_length: int | None = None
    solver_batch_size: int = 1_000
    build_seconds_per_length: float | None = None
    """Optional wall-clock limit on drawing for one length in one round. When it
    runs out, drawing stops and the shortfall is logged: a result, not a
    failure (used for sample and vote, whose keep rate can be tiny)."""
    max_chunk: int = 20_000
    """Most candidates handed to the teacher at once, so a time limit is checked
    often enough to matter."""

    def train_config(self, round_: int) -> TrainConfig:
        return TrainConfig(
            steps=self.steps,
            batch_size=self.batch_size,
            seed=self.seed * 1_000 + round_,
            eval_every=self.steps,
            checkpoint_every=self.steps,
            eval_per_length=self.eval_per_length,
            warmup_fraction=0.0,
            decay_fraction=min(1.0, self.decay_steps / self.steps),
        )


def rng_for(seed: int, round_: int, digits: int, purpose: str) -> random.Random:
    """A separate, reproducible random stream per (starting point, round, length, purpose)."""
    offset = {"build": 0, "checks": 1, "replay": 2}[purpose]
    return random.Random(((seed * 10 + round_) * 100 + digits) * 10 + offset + 7_000_000)


def draw_fresh(rng: random.Random, digits: int, n: int, exclude: set[Addition]) -> list[Addition]:
    """``n`` new problems of this length, none in ``exclude`` (which grows)."""
    draw = draw_hard if digits in HARD_DIGITS else draw_easy
    out = []
    while len(out) < n:
        p = draw(rng, digits)
        if p not in exclude:
            exclude.add(p)
            out.append(p)
    return out


def collect_kept(teacher: Teacher, solver: Solver, rng: random.Random, digits: int, quota: int,
                 cap: int, exclude: set[Addition], seconds: float | None = None,
                 max_chunk: int = 20_000, draw: Callable | None = None) -> tuple[list[tuple[Addition, int]], int, bool]:
    """Draw candidates in chunks until ``quota`` answers are kept, ``cap`` are drawn,
    or (if given) ``seconds`` have passed.

    Chunks are sized from the keep rate so far, so a teacher that keeps 7%
    is not called 140,000 times one by one, nor asked for far too many.
    Returns (kept examples, in draw order and at most ``quota``; number drawn;
    whether the time limit stopped it).

    ``draw(rng, size, n, exclude)`` draws the candidates; None means addition's
    ``draw_fresh``. Longest run (``gainpred.run_train``) passes its own pool.
    """
    draw = draw or draw_fresh
    kept: list[tuple[Addition, int]] = []
    drawn = 0
    rate = 0.3  # first guess; replaced by the observed rate after one chunk
    deadline = None if seconds is None else time.monotonic() + seconds
    while len(kept) < quota and drawn < cap:
        if deadline is not None and time.monotonic() > deadline:
            return kept, drawn, True
        need = quota - len(kept)
        chunk = min(cap - drawn, max_chunk, max(500, math.ceil(1.15 * need / max(rate, 0.02))))
        problems = draw(rng, digits, chunk, exclude)
        for d in teacher(problems, solver):
            drawn += 1
            if d.kept is not None:
                kept.append((d.problem, d.kept))
                if len(kept) == quota:
                    break
        rate = max(len(kept) / drawn, 0.001)
    return kept, drawn, False


def eval_sets(data_dir: Path, per_length: int | None) -> dict[str, EvalSet]:
    return {
        "easy": EvalSet(select(load(data_dir / "test_easy.jsonl"), by_length, per_length), by_length),
        "study": EvalSet(select(load(data_dir / "test_study.jsonl"), by_length_and_carry, per_length),
                         by_length_and_carry),
    }


def known_problems(data_dir: Path) -> set[Addition]:
    """Every test and starting-training problem: rounds must never draw these (P7)."""
    names = ("test_study", "test_reproduction", "test_easy", "train_start")
    return {p for name in names for p in problems_of(load(data_dir / f"{name}.jsonl"))}


def start_checkpoint_of(runs_dir: Path, start_run: str, start_round: int | None = None) -> Path:
    """The checkpoint to continue from.

    ``start_round`` None: the last checkpoint of a starting-training run.
    ``start_round`` k: the checkpoint after round k of a self-improvement run,
    e.g. run f of P9b (no selection after two rounds of the no-carry rule).
    """
    run_dir = runs_dir / start_run if start_round is None else runs_dir / start_run / f"round_{start_round}"
    return sorted((run_dir / "checkpoints").glob("step_*.pt"))[-1]


def start_seed(runs_dir: Path, start_run: str, start_round: int | None = None) -> int:
    """The starting point (P9) of the run we continue from."""
    config = json.loads((runs_dir / start_run / "config.json").read_text())
    return config["train"]["seed"] if start_round is None else config["rounds"]["seed"]


def self_improve(cfg: RoundConfig, run_name: str, start_run: str, data_dir: Path = DATA_DIR,
                 device: torch.device | None = None, runs_dir: Path = RUNS_DIR,
                 make_solver: Callable[[Transformer, int], Solver] | None = None,
                 start_round: int | None = None) -> Transformer:
    """Run ``cfg.rounds`` rounds. ``make_solver`` (model, round) -> Solver replaces the
    model as piece-adder; only tests use it (with a perfect calculator).

    With ``start_round`` k, continue from round k of the self-improvement run
    ``start_run`` and number the rounds from k + 1: round k + 1 then uses the
    same lengths, candidate stream and schedule as round k + 1 of that run
    (only the teacher may differ). The default (None) starts at round 1 from a
    starting-training run.
    """
    device = device or pick_device()
    run_dir = runs_dir / run_name
    if (run_dir / "rounds.jsonl").exists():
        raise FileExistsError(f"{run_dir} already holds a run; pick another --run name")
    run_dir.mkdir(parents=True)
    start_checkpoint = start_checkpoint_of(runs_dir, start_run, start_round)
    first_round = (start_round or 0) + 1
    if first_round + cfg.rounds - 1 > len(MAX_DIGITS_BY_ROUND):
        raise ValueError(f"rounds {first_round}..{first_round + cfg.rounds - 1}: the study has only "
                         f"{len(MAX_DIGITS_BY_ROUND)} rounds")
    model = load_model(start_checkpoint, device)
    teacher = TEACHERS[cfg.teacher]
    manifest = json.loads((data_dir / "manifest.json").read_text())
    (run_dir / "config.json").write_text(json.dumps({
        "rounds": asdict(cfg), "start_run": start_run, "start_round": start_round,
        "start_checkpoint": str(start_checkpoint.relative_to(runs_dir)),
        "device": device_name(device), "torch": torch.__version__, "task": "addition/self-improvement",
        "data": manifest}, indent=2) + "\n")
    log_file = open(run_dir / "rounds.jsonl", "a")

    def log(event: dict) -> None:
        log_file.write(json.dumps(event) + "\n")
        log_file.flush()

    tests = eval_sets(data_dir, cfg.eval_per_length)
    exclude = known_problems(data_dir)

    def run_evals(round_: int) -> None:
        model.eval()
        for name, eval_set in tests.items():
            accuracy = evaluate(model, eval_set.records, eval_set.group_key, 1_000, device)
            log({"event": "eval", "round": round_, "set": name, "accuracy": accuracy})
            print(f"  round {round_} eval {name}: all {accuracy['all']:.1%}  " +
                  "  ".join(f"{k}:{v:.0%}" for k, v in accuracy.items() if k != "all"))

    print(f"{run_name}: teacher {cfg.teacher}, starting from {start_run}/{start_checkpoint.name}, device {device}")
    run_evals(first_round - 1)  # the model as it starts: round 0, or round k of start_run
    for round_ in range(first_round, first_round + cfg.rounds):
        watch = Stopwatch(device)
        max_digits = MAX_DIGITS_BY_ROUND[round_ - 1]
        solver = (make_solver(model, round_) if make_solver
                  else ModelSolver(model, device, cfg.solver_batch_size, seed=cfg.seed * 1_000 + round_))
        examples: list[tuple[Addition, int]] = []

        # 1. Draw and build.
        for digits in range(min(HARD_DIGITS), max_digits + 1):
            tokens_before = getattr(solver, 'generated_tokens', 0)
            build_before = watch.totals["build"]
            with watch.time("build"):
                kept, drawn, timed_out = collect_kept(
                    teacher, solver, rng_for(cfg.seed, round_, digits, "build"), digits, cfg.kept_per_length,
                    cfg.draw_cap_factor * cfg.kept_per_length, exclude, cfg.build_seconds_per_length,
                    cfg.max_chunk)
            right = sum(answer == solve(p) for p, answer in kept)  # audit only
            log({"event": "build", "round": round_, "digits": digits, "drawn": drawn, "kept": len(kept),
                 "shortfall": cfg.kept_per_length - len(kept), "kept_share": len(kept) / drawn if drawn else 0.0,
                 "kept_right": right / len(kept) if kept else None, "stopped_by_time": timed_out,
                 "seconds": watch.totals["build"] - build_before,
                 "generated_tokens": getattr(solver, 'generated_tokens', 0) - tokens_before})
            print(f"  round {round_} {digits} digits: drew {drawn:,}, kept {len(kept):,} "
                  f"({len(kept) / max(drawn, 1):.1%}), right {right / max(len(kept), 1):.1%}"
                  f"{' (stopped by the time limit)' if timed_out else ''}")
            examples += kept

        # 2. Replay: easy problems answered by the current model, unchecked.
        with watch.time("build"):
            replay_problems = [p for d in EASY_DIGITS
                               for p in draw_fresh(rng_for(cfg.seed, round_, d, "replay"), d,
                                                   cfg.replay_per_length, exclude)]
            replay_answers = solver(replay_problems)
        replay = [(p, a) for p, a in zip(replay_problems, replay_answers) if isinstance(a, int)]
        log({"event": "replay", "round": round_, "problems": len(replay_problems), "readable": len(replay),
             "right": sum(a == solve(p) for p, a in replay) / len(replay_problems)})
        examples += replay

        # 3. Checks (P8) at every length from 8 to the maximum, before training:
        #    the verdict of experiment 3 (P9b) is given in advance.
        for digits in range(min(HARD_DIGITS), max_digits + 1):
            sample = draw_fresh(rng_for(cfg.seed, round_, digits, "checks"), digits, cfg.check_sample, exclude)
            with watch.time("checks"):
                report = run_checks(teacher(sample, solver), solver)
            log({"event": "checks", "round": round_, "digits": digits, **asdict(report)})
            print(f"  round {round_} checks {digits} digits: check 1 {report.check_1}, 2b {report.check_2b}, "
                  f"teachable {report.teachable:.1%}, agreement rule {report.check_2_kept:.1%}")

        # 4. Train (fresh optimiser; schedule and batch order per round).
        if len(examples) < cfg.batch_size:
            raise RuntimeError(f"round {round_}: only {len(examples)} training examples; the model "
                               "answers almost nothing readably, so there is nothing to train on")
        model = train(model, examples, cfg.train_config(round_), run_dir / f"round_{round_}", {}, device,
                      extra_config={"task": "addition/self-improvement-round", "round": round_,
                                    "max_digits": max_digits, "examples": len(examples)})
        # 5. Evaluate.
        run_evals(round_)
        log({"event": "round_done", "round": round_, "examples": len(examples),
             "seconds": dict(watch.totals), "generated_tokens": getattr(solver, 'generated_tokens', 0)})

    log({"event": "done"})
    log_file.close()
    return model


def main() -> None:
    d = RoundConfig()
    parser = argparse.ArgumentParser(description="Addition self-improvement rounds (P3).")
    parser.add_argument("--run", required=True)
    parser.add_argument("--start", required=True, help="run to continue from (starting training, by default)")
    parser.add_argument("--start-round", type=int, default=None,
                        help="continue from this round of a self-improvement run --start (e.g. P9b run f)")
    parser.add_argument("--build-seconds", type=float, default=None,
                        help="wall-clock limit on drawing per length and round; the shortfall is logged")
    parser.add_argument("--teacher", choices=sorted(TEACHERS), default=d.teacher)
    parser.add_argument("--rounds", type=int, default=d.rounds)
    parser.add_argument("--kept", type=int, default=d.kept_per_length)
    parser.add_argument("--replay", type=int, default=d.replay_per_length)
    parser.add_argument("--checks", type=int, default=d.check_sample)
    parser.add_argument("--steps", type=int, default=d.steps)
    parser.add_argument("--decay-steps", type=int, default=d.decay_steps)
    parser.add_argument("--eval-per-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    seed = start_seed(RUNS_DIR, args.start, args.start_round)
    cfg = RoundConfig(teacher=args.teacher, rounds=args.rounds, seed=seed, build_seconds_per_length=args.build_seconds,
                      kept_per_length=args.kept, replay_per_length=args.replay, check_sample=args.checks,
                      steps=args.steps, decay_steps=args.decay_steps, eval_per_length=args.eval_per_length)
    self_improve(cfg, args.run, args.start, device=torch.device(args.device) if args.device else None,
                 start_round=args.start_round)


if __name__ == "__main__":
    main()
