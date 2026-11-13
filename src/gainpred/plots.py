"""Training curves for runs in ``results/runs/``, saved as PNGs in ``results/plots/``.

Usage::

    uv run python -m gainpred.plots start-seed0 start-seed0-stable
    uv run python -m gainpred.plots start-seed0-stable --out results/plots/stable.png

(matplotlib is a dev dependency: plotting happens on the laptop, never on Modal.)

What the four panels show
=========================
All share the x-axis (training step), so a spike can be followed downwards
from the loss into the accuracies.

1. **Training loss** (log scale). The solid line is the average loss over each
   logging window of 50 steps; the faint band above it reaches the *largest*
   single-batch loss in that window (newer runs only), so a spike shows even
   when the average smooths it. Loss is cross-entropy on answer characters:
   ~2.3 means guessing among 10 digits, ~0 means certain and right.
2. **Gradient norm before clipping** (log scale, largest in each window). The
   dashed line is the clip threshold 1.0: everything above it was scaled down.
   Only logged for runs with clipping.
3. **Easy test accuracy** (3–7 digits, check 0). Dotted line: the 99% pass mark.
4. **8-digit accuracy, asked directly** -- the starting point for
   self-improvement (P9 expects ~0%).

Grey shading marks the warm-up and decay phases of the learning-rate schedule
(for runs that have one; the first run used a constant rate).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files; no window
import matplotlib.pyplot as plt  # noqa: E402

from gainpred.report import RESULTS_DIR, Run  # noqa: E402

# Validated categorical slots 1-2 (blue, orange) of the dataviz reference palette.
SERIES = ["#2a78d6", "#eb6834"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def shade_schedule(ax: plt.Axes, run: Run) -> None:
    t = run.config["train"]
    steps = t["steps"]
    warm = steps * t.get("warmup_fraction", 0)
    decay = steps * t.get("decay_fraction", 0)
    if warm:
        ax.axvspan(0, warm, color=GRID, alpha=0.6, linewidth=0)
    if decay:
        ax.axvspan(steps - decay, steps, color=GRID, alpha=0.6, linewidth=0)


def plot(runs: list[Run], out: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    loss_ax, grad_ax, easy_ax, hard_ax = axes
    for ax in axes:
        style(ax)
    # Shade by the run that has a schedule (if any), so phases are visible.
    scheduled = [r for r in runs if r.config["train"].get("decay_fraction")]
    if scheduled:
        for ax in axes:
            shade_schedule(ax, scheduled[0])

    for run, color in zip(runs, SERIES):
        steps = [e["step"] for e in run.train]
        loss_ax.plot(steps, [e["loss"] for e in run.train], color=color, linewidth=1.6, label=run.name)
        if "loss_max" in run.train[0]:
            loss_ax.fill_between(steps, [e["loss"] for e in run.train], [e["loss_max"] for e in run.train],
                                 color=color, alpha=0.18, linewidth=0)
        if "grad_norm_max" in run.train[0]:
            grad_ax.plot(steps, [e["grad_norm_max"] for e in run.train], color=color, linewidth=1.4)
        easy = run.evals_of("easy")
        easy_ax.plot([e["step"] for e in easy], [100 * e["accuracy"]["all"] for e in easy],
                     color=color, linewidth=2, marker="o", markersize=5)
        start = run.evals_of("start")
        eight = [(e["accuracy"]["8/no-carry"] + e["accuracy"]["8/carry"]) / 2 for e in start]
        hard_ax.plot([e["step"] for e in start], [100 * a for a in eight],
                     color=color, linewidth=2, marker="o", markersize=5)

    loss_ax.set_yscale("log")
    loss_ax.set_ylabel("training loss\n(window mean; band to max)", color=INK, fontsize=10)
    grad_ax.set_yscale("log")
    grad_ax.axhline(1.0, color=INK_2, linewidth=1, linestyle="--")
    grad_ax.text(50, 1.15, "clip threshold 1.0", color=INK_2, fontsize=8, va="bottom")
    grad_ax.set_ylabel("gradient norm\n(max per window)", color=INK, fontsize=10)
    easy_ax.axhline(99, color=INK_2, linewidth=1, linestyle=":")
    easy_ax.text(50, 96, "99% pass mark", color=INK_2, fontsize=8, va="top")
    easy_ax.set_ylim(-3, 103)
    easy_ax.set_ylabel("easy test acc. %\n(3–7 digits)", color=INK, fontsize=10)
    hard_ax.set_ylim(-3, 103)
    hard_ax.set_ylabel("8-digit test acc. %\n(asked directly)", color=INK, fontsize=10)
    hard_ax.set_xlabel("training step", color=INK, fontsize=10)
    hard_ax.set_xlim(0, max(r.config["train"]["steps"] for r in runs))

    # One run: the title names it, so the legend only explains the band and shading.
    handles, labels = loss_ax.get_legend_handles_labels() if len(runs) > 1 else ([], [])
    if any("loss_max" in r.train[0] for r in runs):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=SERIES[0], alpha=0.18))
        labels.append("band: up to largest batch loss")
    if scheduled:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.6))
        labels.append("warm-up / decay phases" if len(runs) == 1 else f"warm-up / decay ({scheduled[0].name})")
    loss_ax.legend(handles, labels, frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    title = "Starting training: " + " vs ".join(r.name for r in runs)
    fig.suptitle(title, color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help=f"run names under {RESULTS_DIR}/runs/ (at most {len(SERIES)})")
    parser.add_argument("--out", type=Path, help="output PNG (default: results/plots/<runs>.png)")
    args = parser.parse_args()
    if len(args.runs) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} runs per plot")
    runs = [Run.load(RESULTS_DIR / "runs" / name) for name in args.runs]
    out = args.out or RESULTS_DIR / "plots" / ("__".join(args.runs) + ".png")
    plot(runs, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
