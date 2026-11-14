"""Longest-run training (experiment 2): starting training (P1) and one self-improvement round (P3).

Usage::

    uv run python -m gainpred.run_data                                   # once: write data/longest_run/
    uv run python -m gainpred.run_train start --run runs-start-seed0 --seed 0
    uv run python -m gainpred.run_train round --run runs-join-seed0 --start runs-start-seed0 --teacher join_at_the_cut

    # quick check on a laptop
    uv run python -m gainpred.run_train start --run quick --steps 300 --eval-every 150 --eval-per-group 100

On Modal: ``uv run python -m gainpred.launch runs-main ...`` and ``... runs-rounds ...``.

This module is the longest-run twin of ``gainpred.train`` (starting training)
and ``gainpred.rounds`` (rounds). It reuses their machinery -- the training
loop ``train.train``, checkpoints, the learning-rate schedule, the batched
candidate drawing ``rounds.collect_kept`` -- and plugs in the longest-run
pieces: questions and answers (``formats``), data (``run_data``), teachers
(``run_teachers``) and checks (``checks.run_checks_longest_run``).

Starting training (P1)
======================
50,000 strings per length 6-10, each asked all three questions (runlen,
leftrun, rightrun; P3's default), with their true answers: 750,000 examples.
15,000 updates of 256 (P1), about five passes over the set. Same recipe as
addition, in the setup the project adopted (runbook A4): RMSNorm epsilon
1e-6, warm-up 10%, cosine decay over the last 10% to 1% of the peak, clipping
at 1.0 (``START_NORM_EPS``, ``START_DECAY_FRACTION``).

Evaluated every 1,000 updates on

* ``easy``  -- check 0 (P8): each question type at each length 6-10, grouped
  ``runlen/6`` etc.; the same answers are also grouped by run length,
  ``runlen/run=3`` etc. (P7 "also reported by run length"), at no extra cost;
* ``study`` -- the hard study strings (12-20 symbols) asked ``runlen``
  directly, per group (``random_join``, ``joined_length_5``, ``left_part_3-4``
  ...): the baseline the round has to beat.

One round (P3)
==============
1. **Draw and build.** For each length 12-20, draw hard strings from the P6
   pool (half random join, half run across the join) and let the teacher
   build and filter answers with the current model, until 2,000 are kept or
   20 x 2,000 drawn (P3), or an optional time limit runs out. Each length
   has its own seeded stream (8,000,000 + ...), the same for every teacher
   with the same starting point. The pool never contains a test string or a
   starting-training string (P7).
2. **Replay.** The starting-training rows, with their true answers (P1:
   "plus the starting strings replayed with true answers"). P10 reading: all
   of them, all three questions -- 750,000 rows (``replay_questions``).
3. **Checks** (P8) at every length 12-20 on 2,000 fresh pool strings, with the
   model before training: check 1, the check-2 diagnostic (three cuts), 2b,
   check 3 and the teachable share.
4. **Train** 3,000 updates on kept answers + replay; fresh AdamW, no warm-up,
   cosine decay over the last 1,000 updates (as addition's rounds).
5. **Evaluate** on the easy and study test sets (also before the round).

Logs go to ``runs/<run>/rounds.jsonl`` with the same events as addition's
rounds (eval, build, replay, checks, round_done, done); sizes are under the
key ``length`` instead of ``digits``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from gainpred.checks import run_checks_longest_run
from gainpred.config import ModelConfig
from gainpred.data import Record, load
from gainpred.formats import RUN_QUESTIONS, Run, RunQuestion, longest_run, parse_run_answer, solve_run
from gainpred.model import Transformer, pick_device
from gainpred.rounds import collect_kept, start_checkpoint_of, start_seed
from gainpred.run_data import DEFAULT_OUT_DIR as DATA_DIR
from gainpred.run_data import HARD_LENGTHS, HardRun, draw_pool, withheld
from gainpred.run_teachers import RUN_TEACHERS, RunModelSolver, RunSolver
from gainpred.formats import max_run_answer_chars
from gainpred.tokenizer import EOS_ID, PAD_ID, VOCAB_SIZE, make_prompts, read_generated
from gainpred.train import (
    RUNS_DIR,
    EvalSet,
    Stopwatch,
    TrainConfig,
    autocast,
    device_name,
    load_model,
    select,
    train,
)

START_STEPS = 15_000  # P1
START_NORM_EPS = 1e-6  # the authors' setup, adopted by the project (runbook A4)
START_DECAY_FRACTION = 0.1  # ditto: the last 10% of updates decay
MAX_LENGTH_BY_ROUND = (20,)  # P3: one round, lengths 12-20
SEED_OFFSET = 8_000_000  # round streams; addition's rounds use 7,000,000+


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------


def question_of(record: Record) -> RunQuestion:
    """Easy rows name their question; study rows are always asked ``runlen``."""
    return RunQuestion(str(record.get("question", "runlen")), str(record["string"]))


def by_question_and_length(record: Record) -> str:
    return f"{record['question']}/{record['length']}"


def by_group(record: Record) -> str:
    return str(record["group"])


def by_run_length(record: Record) -> str | None:
    """The secondary grouping of easy rows (P7): question and the answer's run length."""
    return f"{record['question']}/run={record['run_length']}" if "question" in record else None


@torch.no_grad()
def answer_runs_greedily(model: Transformer, questions: list[RunQuestion], batch_size: int,
                         device: torch.device) -> list:
    """The model's greedy answer to each question: a Run or Unreadable (P2)."""
    answers = []
    for start in range(0, len(questions), batch_size):
        chunk = questions[start : start + batch_size]
        limit = max(max_run_answer_chars(q) for q in chunk) + 1  # room for EOS
        with autocast(device):
            outputs = model.generate(make_prompts(chunk), eos_id=EOS_ID, max_new_tokens=limit)
        answers += [read_generated(ids, parse=parse_run_answer) for ids in outputs]
    return answers


def evaluate_runs(model: Transformer, records: list[Record], group_key, batch_size: int,
                  device: torch.device) -> dict[str, float]:
    """Accuracy per group (and per run length for easy rows), plus "all".

    Same contract as ``train.evaluate``, so ``train.train`` can call it through
    ``EvalSet.evaluate``. Unreadable answers count as wrong (P2).
    """
    questions = [question_of(r) for r in records]
    answers = answer_runs_greedily(model, questions, batch_size, device)
    right: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for record, question, answer in zip(records, questions, answers):
        correct = answer == solve_run(question)
        for key in (group_key(record), by_run_length(record)):
            if key is not None:
                total[key] += 1
                right[key] += correct
    # Main groups first (in file order), then the run-length groups (sorted), then "all".
    keys = [k for k in total if "/run=" not in k] + sorted(k for k in total if "/run=" in k)
    accuracy = {key: right[key] / total[key] for key in keys}
    accuracy["all"] = sum(right[k] for k in total if "/run=" not in k) / len(records)
    return accuracy


def eval_sets(data_dir: Path, per_group: int | None) -> dict[str, EvalSet]:
    """The two test sets (P7), optionally cut to the first ``per_group`` rows of each group."""
    easy = select(load(data_dir / "test_easy.jsonl"), by_question_and_length, per_group)
    study = select(load(data_dir / "test_study.jsonl"), by_group, per_group)
    return {"easy": EvalSet(easy, by_question_and_length, evaluate_runs),
            "study": EvalSet(study, by_group, evaluate_runs)}


# ----------------------------------------------------------------------------
# Starting training (P1)
# ----------------------------------------------------------------------------


def start_examples(records: list[Record], questions: tuple[str, ...] | None = None) -> list[tuple[RunQuestion, Run]]:
    """(question, true answer) pairs from starting-training rows, optionally only some question types."""
    keep = set(questions or RUN_QUESTIONS)
    out = []
    for r in records:
        if r["question"] in keep:
            q = question_of(r)
            out.append((q, solve_run(q)))
    return out


def starting_training_runs(cfg: TrainConfig, run_name: str, data_dir: Path = DATA_DIR,
                           device: torch.device | None = None, runs_dir: Path = RUNS_DIR,
                           norm_eps: float | None = START_NORM_EPS) -> Transformer:
    """Train a fresh model on 6-10 symbol strings, all three questions, true answers (P1, P3).

    ``cfg`` should carry ``steps`` 15,000 and ``decay_fraction`` 0.1 for the
    adopted recipe (``main`` and Modal do this); ``norm_eps`` None means
    ModelConfig's default.
    """
    device = device or pick_device()
    examples = start_examples(load(data_dir / "train_start.jsonl"))
    tests = eval_sets(data_dir, cfg.eval_per_length)
    torch.manual_seed(cfg.seed)  # initial weights depend on the starting point (P9)
    extra = {} if norm_eps is None else {"norm_eps": norm_eps}
    model = Transformer(ModelConfig(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, **extra))
    manifest = json.loads((data_dir / "manifest.json").read_text())
    print(f"{run_name}: {model.num_params():,} parameters, {len(examples):,} examples, device {device}")
    return train(model, examples, cfg, runs_dir / run_name, tests, device,
                 extra_config={"task": "longest_run/starting-training", "data": manifest})


# ----------------------------------------------------------------------------
# One round (P3)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRoundConfig:
    teacher: str = "join_at_the_cut"
    rounds: int = 1  # P3
    seed: int = 0
    """Starting point (P9); taken from the starting run."""
    kept_per_length: int = 2_000  # P1, P3
    draw_cap_factor: int = 20  # P3
    replay_questions: tuple[str, ...] = RUN_QUESTIONS
    """Which starting-training rows are replayed with their true answers (P10 reading: all)."""
    check_sample: int = 2_000  # P8
    check_2_rule: str = "join_at_the_cut"  # P10 reading: how the three diagnostic cuts are combined
    steps: int = 3_000  # P1
    decay_steps: int = 1_000  # P10 (Lee et al. code), as addition
    batch_size: int = 256
    eval_per_length: int | None = None
    """Evaluate on only the first N rows of each test group (quick checks)."""
    solver_batch_size: int = 1_000
    build_seconds_per_length: float | None = None
    max_chunk: int = 20_000
    withheld: bool = False
    """The withheld version of experiment 2 (P6): every candidate and check problem is
    drawn from the same pool, then run-across-the-join problems with a left part of
    3 or 4 are discarded, so the model is never taught them. Tested on joined runs
    of 7-8 (the study groups joined_length_7 and joined_length_8)."""

    def train_config(self, round_: int) -> TrainConfig:
        return TrainConfig(steps=self.steps, batch_size=self.batch_size, seed=self.seed * 1_000 + round_,
                           eval_every=self.steps, checkpoint_every=self.steps, eval_per_length=self.eval_per_length,
                           warmup_fraction=0.0, decay_fraction=min(1.0, self.decay_steps / self.steps))


def rng_for(seed: int, round_: int, length: int, purpose: str) -> random.Random:
    """A separate random stream per (starting point, round, length, purpose), from 8,000,000 up."""
    offset = {"build": 0, "checks": 1}[purpose]
    return random.Random(((seed * 10 + round_) * 100 + length) * 10 + offset + SEED_OFFSET)


def draw_fresh(rng: random.Random, length: int, n: int, exclude: set[str]) -> list[HardRun]:
    """``n`` new pool strings of this length (P6), none in ``exclude`` (which grows)."""
    return draw_pool(rng, length, n, exclude)


def draw_withheld(rng: random.Random, length: int, n: int, exclude: set[str]) -> list[HardRun]:
    """The same pool draw, minus run-across problems with a of 3 or 4 (P6, withheld version).

    Returns fewer than ``n`` (about n x 0.75); the round counts only problems the
    teacher actually saw as drawn, so draws per kept answer stay comparable.
    """
    return withheld(draw_pool(rng, length, n, exclude))


def known_strings(data_dir: Path) -> set[str]:
    """Every test and starting-training string: rounds never draw these (P7)."""
    return {str(r["string"]) for name in ("test_study", "test_easy", "train_start")
            for r in load(data_dir / f"{name}.jsonl")}


def self_improve_runs(cfg: RunRoundConfig, run_name: str, start_run: str, data_dir: Path = DATA_DIR,
                      device: torch.device | None = None, runs_dir: Path = RUNS_DIR,
                      make_solver: Callable[[Transformer, int], RunSolver] | None = None) -> Transformer:
    """Run ``cfg.rounds`` longest-run rounds from the last checkpoint of ``start_run``.

    ``make_solver`` (model, round) -> solver replaces the model as piece
    answerer; only tests use it (with a perfect solver).
    """
    device = device or pick_device()
    run_dir = runs_dir / run_name
    if (run_dir / "rounds.jsonl").exists():
        raise FileExistsError(f"{run_dir} already holds a run; pick another --run name")
    if cfg.rounds > len(MAX_LENGTH_BY_ROUND):
        raise ValueError(f"the study has {len(MAX_LENGTH_BY_ROUND)} longest-run round(s) (P3)")
    run_dir.mkdir(parents=True)
    start_checkpoint = start_checkpoint_of(runs_dir, start_run)
    model = load_model(start_checkpoint, device)
    teacher = RUN_TEACHERS[cfg.teacher]
    manifest = json.loads((data_dir / "manifest.json").read_text())
    (run_dir / "config.json").write_text(json.dumps({
        "rounds": asdict(cfg), "start_run": start_run, "start_round": None,
        "start_checkpoint": str(start_checkpoint.relative_to(runs_dir)),
        "device": device_name(device), "torch": torch.__version__, "task": "longest_run/self-improvement",
        "data": manifest}, indent=2) + "\n")
    log_file = open(run_dir / "rounds.jsonl", "a")

    def log(event: dict) -> None:
        log_file.write(json.dumps(event) + "\n")
        log_file.flush()

    tests = eval_sets(data_dir, cfg.eval_per_length)
    replay = start_examples(load(data_dir / "train_start.jsonl"), cfg.replay_questions)
    exclude = known_strings(data_dir)

    def run_evals(round_: int) -> None:
        model.eval()
        for name, eval_set in tests.items():
            accuracy = evaluate_runs(model, eval_set.records, eval_set.group_key, 1_000, device)
            log({"event": "eval", "round": round_, "set": name, "accuracy": accuracy})
            print(f"  round {round_} eval {name}: all {accuracy['all']:.1%}  " +
                  "  ".join(f"{k}:{v:.0%}" for k, v in accuracy.items() if k != "all" and "/run=" not in k))

    print(f"{run_name}: teacher {cfg.teacher}, starting from {start_run}/{start_checkpoint.name}, device {device}")
    run_evals(0)
    for round_ in range(1, cfg.rounds + 1):
        watch = Stopwatch(device)
        max_length = MAX_LENGTH_BY_ROUND[round_ - 1]
        lengths = range(min(HARD_LENGTHS), max_length + 1)
        solver = (make_solver(model, round_) if make_solver
                  else RunModelSolver(model, device, cfg.solver_batch_size, seed=cfg.seed * 1_000 + round_))
        examples: list[tuple[RunQuestion, Run]] = []

        # 1. Draw and build.
        for length in lengths:
            tokens_before = getattr(solver, "generated_tokens", 0)
            build_before = watch.totals["build"]
            with watch.time("build"):
                kept, drawn, timed_out = collect_kept(
                    teacher, solver, rng_for(cfg.seed, round_, length, "build"), length, cfg.kept_per_length,
                    cfg.draw_cap_factor * cfg.kept_per_length, exclude, cfg.build_seconds_per_length,
                    cfg.max_chunk, draw=draw_withheld if cfg.withheld else draw_fresh)
            right = sum(answer == longest_run(p.string) for p, answer in kept)  # audit only
            kinds = {k: sum(p.kind == k for p, _ in kept) for k in ("random_join", "run_across")}
            log({"event": "build", "round": round_, "length": length, "drawn": drawn, "kept": len(kept),
                 "shortfall": cfg.kept_per_length - len(kept), "kept_share": len(kept) / drawn if drawn else 0.0,
                 "kept_right": right / len(kept) if kept else None, "kept_by_kind": kinds,
                 "stopped_by_time": timed_out, "seconds": watch.totals["build"] - build_before,
                 "generated_tokens": getattr(solver, "generated_tokens", 0) - tokens_before})
            print(f"  round {round_} length {length}: drew {drawn:,}, kept {len(kept):,} "
                  f"({len(kept) / max(drawn, 1):.1%}), right {right / max(len(kept), 1):.1%}"
                  f"{' (stopped by the time limit)' if timed_out else ''}")
            examples += [(p.question, answer) for p, answer in kept]

        # 2. Replay: the starting strings with their true answers (P1).
        log({"event": "replay", "round": round_, "problems": len(replay), "readable": len(replay), "right": 1.0,
             "true_answers": True, "questions": list(cfg.replay_questions)})
        examples += replay

        # 3. Checks (P8) at every length, before training.
        for length in lengths:
            draw = draw_withheld if cfg.withheld else draw_fresh
            sample = draw(rng_for(cfg.seed, round_, length, "checks"), length, cfg.check_sample, exclude)
            with watch.time("checks"):
                report = run_checks_longest_run(teacher(sample, solver), solver, cfg.check_2_rule)
            log({"event": "checks", "round": round_, "length": length, **asdict(report)})
            print(f"  round {round_} checks {length}: check 1 {report.check_1}, 2b {report.check_2b}, "
                  f"check 3 {report.check_3}, teachable {report.teachable:.1%}, "
                  f"check 2 keeps {report.check_2_kept:.1%}")

        # 4. Train.
        model = train(model, examples, cfg.train_config(round_), run_dir / f"round_{round_}", {}, device,
                      extra_config={"task": "longest_run/self-improvement-round", "round": round_,
                                    "max_length": max_length, "examples": len(examples)})
        # 5. Evaluate.
        run_evals(round_)
        log({"event": "round_done", "round": round_, "examples": len(examples),
             "seconds": dict(watch.totals), "generated_tokens": getattr(solver, "generated_tokens", 0)})

    log({"event": "done"})
    log_file.close()
    return model


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("start", help="starting training (P1)")
    s.add_argument("--run", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--steps", type=int, default=START_STEPS)
    s.add_argument("--eval-every", type=int, default=1_000)
    s.add_argument("--eval-per-group", type=int, default=None)
    s.add_argument("--checkpoint-every", type=int, default=3_000)
    s.add_argument("--norm-eps", type=float, default=START_NORM_EPS)
    s.add_argument("--decay-fraction", type=float, default=START_DECAY_FRACTION)
    s.add_argument("--device", default=None)
    r = sub.add_parser("round", help="one self-improvement round (P3)")
    r.add_argument("--run", required=True)
    r.add_argument("--start", required=True, help="starting-training run to continue from")
    r.add_argument("--teacher", choices=sorted(RUN_TEACHERS), default=RunRoundConfig.teacher)
    r.add_argument("--kept", type=int, default=RunRoundConfig.kept_per_length)
    r.add_argument("--checks", type=int, default=RunRoundConfig.check_sample)
    r.add_argument("--steps", type=int, default=RunRoundConfig.steps)
    r.add_argument("--decay-steps", type=int, default=RunRoundConfig.decay_steps)
    r.add_argument("--build-seconds", type=float, default=None)
    r.add_argument("--eval-per-group", type=int, default=None)
    r.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else None
    if args.command == "start":
        cfg = TrainConfig(steps=args.steps, seed=args.seed, eval_every=args.eval_every,
                          eval_per_length=args.eval_per_group, checkpoint_every=args.checkpoint_every,
                          decay_fraction=args.decay_fraction)
        starting_training_runs(cfg, args.run, device=device, norm_eps=args.norm_eps or None)
    else:
        cfg = RunRoundConfig(teacher=args.teacher, seed=start_seed(RUNS_DIR, args.start),
                             kept_per_length=args.kept, check_sample=args.checks, steps=args.steps,
                             decay_steps=args.decay_steps, build_seconds_per_length=args.build_seconds,
                             eval_per_length=args.eval_per_group)
        self_improve_runs(cfg, args.run, args.start, device=device)


if __name__ == "__main__":
    main()
