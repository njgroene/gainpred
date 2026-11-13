"""Collect run results into ``results/``, which *is* kept in git.

Usage::

    uv run python -m gainpred.report            # copy logs from runs/, rebuild the tables
    uv run python -m gainpred.report --no-copy  # only rebuild the tables

Why this exists
===============
``runs/<name>/`` holds everything a run produces, but it is not in git: the
checkpoints are hundreds of megabytes per run. The *measurements*, though,
are small (a run's ``log.jsonl`` is ~60 KB) and are the evidence behind every
decision in the protocol. So this module copies each run's ``config.json`` and
``log.jsonl`` into ``results/runs/<name>/`` and builds two files from them:

``results/summary.md``
    Generated tables, one section per run: settings, final accuracy per
    length, accuracy at each evaluation, and how unstable training was.
    Do not edit by hand -- rerun this module instead.
``results/evals.csv``
    Every evaluation of every run in "long" format (one row per run, step,
    test set and group), easy to load into pandas or a spreadsheet.

``results/README.md`` is written by hand: what each run was for and what we
concluded. The generated files hold the numbers; the README holds the story.

Reading the logs
================
``log.jsonl`` has one JSON object per line, told apart by ``"event"``:

* ``train`` -- every ``log_every`` steps: average loss, and (newer runs) the
  largest loss, gradient norm and learning rate in that window.
* ``eval`` -- accuracy on one test set (``easy`` or ``start``), per group.
* ``done`` -- the run finished; total seconds spent training and evaluating.

A run without a ``done`` line was stopped early; it is reported as such.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

RUNS_DIR = Path("runs")
RESULTS_DIR = Path("results")
KEPT_FILES = ("config.json", "log.jsonl")


@dataclass
class Run:
    """One run's settings and log, read back from disk."""

    name: str
    config: dict
    train: list[dict] = field(default_factory=list)
    evals: list[dict] = field(default_factory=list)
    done: dict | None = None

    @classmethod
    def load(cls, run_dir: Path) -> Run:
        run = cls(run_dir.name, json.loads((run_dir / "config.json").read_text()))
        for line in (run_dir / "log.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "train":
                run.train.append(event)
            elif event["event"] == "eval":
                run.evals.append(event)
            elif event["event"] == "done":
                run.done = event
        return run

    @property
    def last_step(self) -> int:
        steps = [e["step"] for e in self.train + self.evals]
        return max(steps, default=0)

    def evals_of(self, name: str) -> list[dict]:
        return [e for e in self.evals if e["set"] == name]


def collect(runs_dir: Path = RUNS_DIR, results_dir: Path = RESULTS_DIR) -> list[str]:
    """Copy config.json and log.jsonl of every run into results/runs/. Returns run names."""
    copied = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not all((run_dir / f).exists() for f in KEPT_FILES):
            continue
        target = results_dir / "runs" / run_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for f in KEPT_FILES:
            shutil.copy2(run_dir / f, target / f)
        copied.append(run_dir.name)
    return copied


def load_all(results_dir: Path = RESULTS_DIR) -> list[Run]:
    runs_dir = results_dir / "runs"
    return [Run.load(p) for p in sorted(runs_dir.iterdir()) if p.is_dir()] if runs_dir.exists() else []


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def recipe(run: Run) -> str:
    """Short description of the learning-rate schedule and clipping."""
    t = run.config["train"]
    clip = t.get("grad_clip")
    if "warmup_fraction" not in t:
        return "constant LR, no clipping"
    sched = (
        f"warmup {t['warmup_fraction']:.0%} / decay {t['decay_fraction']:.0%} to {t['min_lr_ratio']:.0%}"
        if t["warmup_fraction"] or t["decay_fraction"]
        else "constant LR"
    )
    return f"{sched}, clip {clip}" if clip else f"{sched}, no clipping"


def accuracy_table(evals: list[dict]) -> list[str]:
    """Rows = evaluation steps, columns = groups (``all`` last)."""
    groups = [g for g in evals[0]["accuracy"] if g != "all"] + ["all"]
    rows = [[f"{e['step']:,}"] + [pct(e["accuracy"].get(g)) for g in groups] for e in evals]
    return table(["step"] + groups, rows)


def stability(run: Run) -> list[str]:
    """Loss and gradient spikes, split into the phases of the schedule."""
    if not run.train or "grad_norm_max" not in run.train[0]:
        return ["Spike statistics not logged for this run (older log format)."]
    t = run.config["train"]
    steps = t["steps"]
    warm_end = int(steps * t.get("warmup_fraction", 0))
    decay_start = steps - int(steps * t.get("decay_fraction", 0))
    phases = [("warm-up", 0, warm_end), ("stable", warm_end, decay_start), ("decay", decay_start, steps)]
    rows = []
    for name, lo, hi in phases:
        window = [e for e in run.train if lo < e["step"] <= hi]
        if not window:
            continue
        rows.append([
            name,
            f"{lo:,}–{hi:,}",
            f"{max(e['loss_max'] for e in window):.3g}",
            f"{max(e['grad_norm_max'] for e in window):.3g}",
            f"{window[-1]['loss']:.2e}",
        ])
    return table(["phase", "steps", "largest batch loss", "largest grad norm", "loss at end"], rows)


def summarise_run(run: Run) -> list[str]:
    t = run.config["train"]
    status = "finished" if run.done else f"**stopped early** at step {run.last_step:,}"
    lines = [f"## {run.name}", ""]
    lines += [
        f"- Device: {run.config.get('device', 'not recorded')}, torch {run.config.get('torch', 'not recorded')}",
        f"- Steps: {t['steps']:,} planned, {status}; batch {t['batch_size']}, seed {t['seed']}",
        f"- Recipe: {recipe(run)}",
        f"- Test sets: {'full' if t.get('eval_per_length') is None else f'{t['eval_per_length']} per length'}",
    ]
    if run.done:
        s = run.done["seconds"]
        lines.append(f"- Time: {s['train'] / 60:.1f} min training, {s.get('eval', 0) / 60:.1f} min evaluating")
    lines.append("")

    easy = run.evals_of("easy")
    if easy:
        best = max(easy, key=lambda e: e["accuracy"]["all"])
        final = easy[-1]
        lines.append(
            f"Easy addition (3–7 digits, check 0; pass mark 99% for starting training): "
            f"final **{pct(final['accuracy']['all'])}%** at step {final['step']:,}; "
            f"best {pct(best['accuracy']['all'])}% at step {best['step']:,}."
        )
        lines += ["", *accuracy_table(easy), ""]
    start = run.evals_of("start")
    if start:
        lines.append("Starting point (8–9 digits asked directly, split by carry across cut 1; P9 expects ~0%):")
        lines += ["", *accuracy_table(start), ""]
    lines += ["Training stability:", "", *stability(run), ""]
    return lines


def overview(runs: list[Run]) -> list[str]:
    rows = []
    for run in runs:
        easy, start = run.evals_of("easy"), run.evals_of("start")
        final_easy = easy[-1]["accuracy"]["all"] if easy else None
        final_start = start[-1]["accuracy"] if start else {}
        rows.append([
            run.name,
            run.config.get("device", "?"),
            f"{run.last_step:,} / {run.config['train']['steps']:,}",
            recipe(run),
            pct(final_easy),
            pct(max((e["accuracy"]["all"] for e in easy), default=None)),
            pct(final_start.get("8/no-carry")),
            pct(final_start.get("8/carry")),
            pct(final_start.get("9/no-carry")),
            pct(final_start.get("9/carry")),
        ])
    header = ["run", "device", "steps", "recipe", "easy final", "easy best",
              "8 no-carry", "8 carry", "9 no-carry", "9 carry"]
    return table(header, rows)


def write_summary(runs: list[Run], results_dir: Path = RESULTS_DIR) -> None:
    lines = [
        "# Run summary",
        "",
        "Generated by `uv run python -m gainpred.report` from `results/runs/*/log.jsonl`; do not edit by hand.",
        "Accuracies are percentages of greedy answers that are exactly right, at the *last* evaluation",
        "unless marked otherwise. What each run was for is in `README.md`.",
        "",
        *overview(runs),
        "",
    ]
    for run in runs:
        lines += summarise_run(run)
    (results_dir / "summary.md").write_text("\n".join(lines))


def write_evals_csv(runs: list[Run], results_dir: Path = RESULTS_DIR) -> None:
    with (results_dir / "evals.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "step", "set", "group", "accuracy"])
        for run in runs:
            for e in run.evals:
                for group, acc in e["accuracy"].items():
                    writer.writerow([run.name, e["step"], e["set"], group, acc])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-copy", action="store_true", help="don't copy from runs/; only rebuild tables")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    if not args.no_copy:
        print("copied:", ", ".join(collect(args.runs_dir, args.results_dir)) or "nothing")
    runs = load_all(args.results_dir)
    write_summary(runs, args.results_dir)
    write_evals_csv(runs, args.results_dir)
    print(f"wrote {args.results_dir}/summary.md and evals.csv for {len(runs)} runs")


if __name__ == "__main__":
    main()
