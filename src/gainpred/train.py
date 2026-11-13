"""Training loop and evaluation; entry point for starting training (P1, P9).

Run starting training with::

    uv run python -m gainpred.data                     # once: write data/addition/
    uv run python -m gainpred.train --run quick --steps 300 --eval-every 150 --eval-per-length 200
    uv run python -m gainpred.train --run start-seed0  # the full P1 recipe: 10,000 updates

Everything a run produces goes in ``runs/<name>/`` (not in git):

    config.json       model + training settings, and the dataset fingerprints
    log.jsonl         one JSON line per log/eval event (loss, accuracies, timings)
    checkpoints/      model + optimiser state, every ``checkpoint_every`` steps

What one training step does
===========================
1. Take a batch of 256 examples (P1), each ``prompt + answer + EOS``.
2. Forward pass: the model scores every possible next character at every
   position. Loss = average cross-entropy on the answer characters only (P1).
3. Backward pass: compute how much each of the 14M weights contributed to the
   loss (its gradient).
4. Optimiser step: AdamW nudges every weight against its gradient, scaled by
   the learning rate (constant 5e-4, no warm-up: P1, P10).

Repeat 10,000 times for starting training; 3,000 per self-improvement round.

Epochs and shuffling
====================
Batches are drawn by shuffling the whole training set and walking through it;
when it is used up (one "epoch"), it is reshuffled. Starting training sees
10,000 x 256 = 2.56M examples from a 250,000-example set: about 10 epochs. The
shuffle uses its own seeded generator, so a run's batch order is reproducible.

Evaluation
==========
Accuracy = share of problems the model answers exactly right by greedy
decoding, with unreadable answers counted as wrong (P2). It is reported per
group (per length, and for the study set also by carry across cut 1), since
an average over lengths would hide exactly the pattern we care about.

Two evaluations matter for starting training (P9):
* ``easy``  -- check 0 on 3-7 digit test problems; acceptance: at least 99%
  per length (the general check-0 pass mark is 98%);
* ``start`` -- 8 and 9 digits asked directly; expected close to 0%. This is
  the baseline self-improvement has to beat.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from gainpred.config import ModelConfig
from gainpred.data import DEFAULT_OUT_DIR as DATA_DIR
from gainpred.data import Record, load, problems_of
from gainpred.formats import Addition, max_answer_chars, solve
from gainpred.model import Transformer, pick_device
from gainpred.tokenizer import EOS_ID, PAD_ID, VOCAB_SIZE, make_prompts, make_training_batch, read_generated

RUNS_DIR = Path("runs")


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 10_000
    """P1: 10,000 updates for addition starting training (3,000 per round)."""

    batch_size: int = 256
    """P1."""

    seed: int = 0
    """P9 "starting point" 0-4. Sets the initial weights and the batch order."""

    log_every: int = 50
    """How often to log the training loss (averaged since the last log)."""

    eval_every: int = 1_000
    """How often to run the evaluations. They also always run at the end."""

    eval_per_length: int | None = None
    """Evaluate on only the first N problems of each group (quick checks).
    None = the full test sets, as the protocol requires for real runs."""

    eval_batch_size: int = 1_000
    """Prompts per generation batch. Only affects speed, not results."""

    checkpoint_every: int = 2_000
    """How often to save model + optimiser. Also saved at the end."""


# ----------------------------------------------------------------------------
# Devices and timing
# ----------------------------------------------------------------------------


def autocast(device: torch.device):
    """bfloat16 mixed precision (P1) on GPUs; plain float32 on CPU.

    Weights stay float32; only the matrix multiplications inside run in
    bfloat16 (see the precision note in model.py). CPU bfloat16 is slow on
    most machines, so the CPU path (used by the tests) stays in float32.
    """
    if device.type in ("cuda", "mps"):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def synchronize(device: torch.device) -> None:
    """Wait for the GPU to finish queued work.

    GPU calls return immediately and run in the background, so without this a
    timer would measure how fast work is *queued*, not how fast it is *done*.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class Stopwatch:
    """Accumulates wall-clock seconds per activity (P9 logs train/eval separately).

    Usage: ``with watch.time("train"): ...``
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.totals: dict[str, float] = defaultdict(float)

    @contextmanager
    def time(self, activity: str):
        synchronize(self.device)
        start = time.perf_counter()
        yield
        synchronize(self.device)
        self.totals[activity] += time.perf_counter() - start


# ----------------------------------------------------------------------------
# Batches
# ----------------------------------------------------------------------------


def batches(
    tokens: torch.Tensor, loss_mask: torch.Tensor, batch_size: int, generator: torch.Generator
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Endless stream of shuffled batches, reshuffling after every epoch.

    ``tokens``/``loss_mask`` hold the *whole* training set, already tokenised
    and padded to one width (tokenising once up front is much faster than
    per step). Each batch is trimmed to its own longest row, so a batch of
    3-digit problems is not padded out to 7-digit width.
    """
    n = tokens.shape[0]
    while True:
        order = torch.randperm(n, generator=generator)
        # Drop the last incomplete batch so every step has exactly batch_size rows.
        for start in range(0, n - batch_size + 1, batch_size):
            index = order[start : start + batch_size].to(tokens.device)
            batch_tokens, batch_mask = tokens[index], loss_mask[index]
            width = int((batch_tokens != PAD_ID).sum(dim=1).max())
            yield batch_tokens[:, :width], batch_mask[:, :width]


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------


GroupKey = Callable[[Record], str]


def by_length(record: Record) -> str:
    return str(record["digits"])


def by_length_and_carry(record: Record) -> str:
    return f"{record['digits']}{'/carry' if record['carry_across_cut1'] else '/no-carry'}"


def select(records: list[Record], group_key: GroupKey, per_group: int | None) -> list[Record]:
    """The first ``per_group`` records of each group (all if None), order kept."""
    if per_group is None:
        return records
    taken: dict[str, int] = defaultdict(int)
    chosen = []
    for r in records:
        key = group_key(r)
        if taken[key] < per_group:
            taken[key] += 1
            chosen.append(r)
    return chosen


@torch.no_grad()
def answer_greedily(
    model: Transformer, problems: list[Addition], batch_size: int, device: torch.device
) -> list:
    """The model's greedy answer to each problem: an int or Unreadable (P2).

    Problems are answered in batches of ``batch_size``. Batching problems of
    similar length wastes less padding, which is why the test files are
    stored grouped by length.
    """
    answers = []
    for start in range(0, len(problems), batch_size):
        chunk = problems[start : start + batch_size]
        # +1 leaves room for EOS after the longest possible correct answer.
        limit = max(max_answer_chars(p) for p in chunk) + 1
        with autocast(device):
            outputs = model.generate(make_prompts(chunk), eos_id=EOS_ID, max_new_tokens=limit)
        answers += [read_generated(ids) for ids in outputs]
    return answers


def evaluate(
    model: Transformer,
    records: list[Record],
    group_key: GroupKey,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Accuracy per group, plus "all". Unreadable answers count as wrong (P2)."""
    problems = problems_of(records)
    answers = answer_greedily(model, problems, batch_size, device)
    right: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for record, problem, answer in zip(records, problems, answers):
        key = group_key(record)
        total[key] += 1
        right[key] += answer == solve(problem)
    accuracy = {key: right[key] / total[key] for key in total}
    accuracy["all"] = sum(right.values()) / sum(total.values())
    return accuracy


# ----------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------


def save_checkpoint(path: Path, model: Transformer, optimizer, step: int, train_cfg: TrainConfig) -> None:
    """Everything needed to evaluate the model later or continue training it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model.cfg.to_dict(),
            "train_config": asdict(train_cfg),
        },
        path,
    )


def load_model(path: Path, device: torch.device | str = "cpu") -> Transformer:
    """Rebuild a model from a checkpoint (weights only, ready to evaluate)."""
    checkpoint = torch.load(path, map_location=device)
    model_cfg = checkpoint["model_config"]
    model_cfg["betas"] = tuple(model_cfg["betas"])
    model = Transformer(ModelConfig(**model_cfg)).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


# ----------------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------------


@dataclass
class EvalSet:
    """A named test set and how to group its accuracies."""

    records: list[Record]
    group_key: GroupKey


def train(
    model: Transformer,
    examples: list[tuple[Addition, int]],
    cfg: TrainConfig,
    run_dir: Path,
    eval_sets: dict[str, EvalSet],
    device: torch.device,
    extra_config: dict | None = None,
) -> Transformer:
    """Train ``model`` on ``examples`` for ``cfg.steps`` updates.

    ``examples`` are (problem, answer to train on) pairs. For starting training
    the answer is the true sum; in self-improvement rounds it will be whatever
    a teacher built (P5) -- this function does not care which.
    """
    if (run_dir / "log.jsonl").exists():
        # Refuse rather than append: two runs in one log would be impossible to untangle.
        raise FileExistsError(f"{run_dir} already holds a run; pick another --run name")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {"model": model.cfg.to_dict(), "train": asdict(cfg), "device": str(device), **(extra_config or {})},
            indent=2,
        )
        + "\n"
    )
    log_file = open(run_dir / "log.jsonl", "a")

    def log(event: dict) -> None:
        log_file.write(json.dumps(event) + "\n")
        log_file.flush()

    model.to(device).train()
    optimizer = model.configure_optimizer()
    tokens, loss_mask = make_training_batch(examples)
    stream = batches(
        tokens.to(device), loss_mask.to(device), cfg.batch_size, torch.Generator().manual_seed(cfg.seed)
    )
    watch = Stopwatch(device)

    def run_evals(step: int) -> None:
        model.eval()
        for name, eval_set in eval_sets.items():
            with watch.time("eval"):
                accuracy = evaluate(model, eval_set.records, eval_set.group_key, cfg.eval_batch_size, device)
            log({"event": "eval", "set": name, "step": step, "accuracy": accuracy})
            shown = "  ".join(f"{k}:{v:.1%}" for k, v in accuracy.items())
            print(f"  eval {name:<6} {shown}")
        model.train()

    # Summing the loss on the GPU and reading it only at log time avoids
    # forcing the CPU to wait for the GPU after every step.
    loss_sum, answer_tokens = torch.zeros((), device=device), 0
    for step in range(1, cfg.steps + 1):
        batch_tokens, batch_mask = next(stream)
        with watch.time("train"):
            with autocast(device):
                _, loss = model(batch_tokens, batch_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        loss_sum += loss.detach()
        answer_tokens += int(batch_mask.sum())

        if step % cfg.log_every == 0 or step == cfg.steps:
            steps_since = (step - 1) % cfg.log_every + 1
            mean_loss = (loss_sum / steps_since).item()
            loss_sum.zero_()
            rate = step / watch.totals["train"]
            log({"event": "train", "step": step, "loss": mean_loss, "answer_tokens": answer_tokens,
                 "seconds": dict(watch.totals)})
            print(f"step {step:>6}  loss {mean_loss:.4f}  {rate:.2f} steps/s")

        if step % cfg.eval_every == 0 or step == cfg.steps:
            run_evals(step)
        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            save_checkpoint(run_dir / "checkpoints" / f"step_{step:06d}.pt", model, optimizer, step, cfg)

    log({"event": "done", "step": cfg.steps, "seconds": dict(watch.totals)})
    log_file.close()
    return model


# ----------------------------------------------------------------------------
# Starting training (P1)
# ----------------------------------------------------------------------------


def starting_training(cfg: TrainConfig, run_name: str, data_dir: Path = DATA_DIR,
                      device: torch.device | None = None) -> Transformer:
    """Train a fresh model on 3-7 digit additions with their true answers."""
    device = device or pick_device()
    train_records = load(data_dir / "train_start.jsonl")
    examples = [(p, solve(p)) for p in problems_of(train_records)]

    study = [r for r in load(data_dir / "test_study.jsonl") if r["digits"] in (8, 9)]
    eval_sets = {
        "easy": EvalSet(select(load(data_dir / "test_easy.jsonl"), by_length, cfg.eval_per_length), by_length),
        "start": EvalSet(select(study, by_length_and_carry, cfg.eval_per_length), by_length_and_carry),
    }

    torch.manual_seed(cfg.seed)  # initial weights depend on the starting point (P9)
    model = Transformer(ModelConfig(vocab_size=VOCAB_SIZE, pad_id=PAD_ID))
    manifest = json.loads((data_dir / "manifest.json").read_text())
    print(f"{run_name}: {model.num_params():,} parameters, {len(examples):,} examples, device {device}")
    return train(model, examples, cfg, RUNS_DIR / run_name, eval_sets, device,
                 extra_config={"task": "addition/starting-training", "data": manifest})


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Addition starting training (P1).")
    parser.add_argument("--run", required=True, help="run name; output goes to runs/<run>/")
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    parser.add_argument("--eval-per-length", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    args = parser.parse_args()
    cfg = TrainConfig(
        steps=args.steps,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_per_length=args.eval_per_length,
        checkpoint_every=args.checkpoint_every,
    )
    starting_training(cfg, args.run, device=torch.device(args.device) if args.device else None)


if __name__ == "__main__":
    main()
