"""Check 3 on addition: does a teacher hand back a piece answer? (Analysis defined after the
confirming runs, amendment 22; it never enters the verdict, and P9b is not rescored.)

Usage::

    uv run python scripts/check3_addition.py              # list the runs
    uv run python scripts/check3_addition.py --launch     # one Modal call per run (inference only)
    uv run python scripts/check3_addition.py --fetch      # -> results/check3_addition.csv

What check 3 is (P8; the same rule as longest run's reading 17)
===============================================================
The share of a teacher's kept answers that are *identical* to one of the piece
answers the teacher asked the model for. Problems set aside are skipped; an
unreadable piece answer never matches (identity is ``==``).

Which answers count as "piece answers" on addition:

* the no-carry rule, the agreement rule, no selection -- the model's answers
  to the piece sums, for *every* cut the teacher built (not only the cut it
  kept): cut 1 for the no-carry rule and no selection, all its cuts for the
  agreement rule;
* sample and vote -- its five sampled direct answers.

Expected by construction: 0% for the three split-and-combine teachers (a
joined answer is written from all its piece sums side by side, so it is longer
than any one of them), 100% for sample and vote (a voted answer is one of the
samples). Longest run is where this matters: there "pick the longer" can only
return a piece's run.

Why this needs the models again
===============================
Addition's logged check reports have no check 3, and a teacher's decision does
not keep the piece sums. So, per run, round and length: load the model as it
was *before* that round's training, redraw the round's check sample (same
seeded stream, 2,000 problems), run the teacher while recording the piece
answers, and compute check 3. Inference only.

Reproducing the logged checks
=============================
Checks 1 and 2b are recomputed on the redrawn sample and compared with the
logged values. For the three greedy teachers they should match exactly: the
sample comes from the same stream, and greedy answers do not depend on any
random state. (The original draw also excluded that round's training-data
draws; for 8-15 digit problems a collision is about one in ten million, and
the comparison would show one.) Sample and vote cannot match exactly: its
answers are sampled, and the sampling generator had been advanced by the
whole build phase before the original checks ran. Replaying the build would
cost most of a round; check 3 for sample and vote does not depend on which
samples were drawn, so its checks 1 and 2b are compared as close, not equal.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, fields
from functools import partial
from pathlib import Path

import torch

from gainpred.checks import check_3, run_checks
from gainpred.formats import Addition, Answer, Unreadable
from gainpred.teachers import (Decision, Solver, TEACHERS, agreement_from, build_all, cuts_of, readable,
                               teacher_key)

GREEDY = ("no_carry", "agreement", "no_selection")


def teacher_with_pieces(name: str, problems: list[Addition], solver: Solver) -> list[tuple[Decision, tuple[Answer, ...]]]:
    """Run teacher ``name`` and return each decision with the piece answers it asked for.

    The decisions are made exactly as ``teachers.TEACHERS[name]`` makes them, from the
    same solver calls in the same order (so a greedy model gives the same answers);
    ``tests/test_check3.py`` checks they agree.
    """
    if name == "sample_and_vote":
        return [(d, d.candidates) for d in TEACHERS[name](problems, solver)]
    if name not in GREEDY:
        raise ValueError(f"unknown addition teacher {name!r}")
    cuts = [cuts_of(p) if name == "agreement" else cuts_of(p)[:1] for p in problems]
    built = build_all(problems, cuts, solver)
    if name == "agreement":
        decisions = agreement_from(problems, built)
    elif name == "no_carry":
        decisions = [Decision(p, readable(b.answer) if b.fits else None, (b.answer,), 0)
                     for p, (b,) in zip(problems, built)]
    else:
        decisions = [Decision(p, readable(b.answer), (b.answer,), 0) for p, (b,) in zip(problems, built)]
    return [(d, tuple(a for b in bs for a in b.pieces)) for d, bs in zip(decisions, built)]


def check_3_addition(pairs: list[tuple[Decision, tuple[Answer, ...]]]) -> float | None:
    """Check 3 over (decision, piece answers) pairs; set-aside problems are skipped."""
    return check_3((d.kept, pieces) for d, pieces in pairs)


def unreadable_pieces(pairs: list[tuple[Decision, tuple[Answer, ...]]]) -> tuple[int, int]:
    """(unreadable piece answers, all piece answers), over every problem."""
    allp = [a for _, pieces in pairs for a in pieces]
    return sum(isinstance(a, Unreadable) for a in allp), len(allp)


@contextmanager
def three_cut_rule():
    """Draw and cut as runs made before the five-cut change did (P4 as frozen, 12 November 2026).

    Those runs (seeds 0-1, rounds 3-4 of ``-authors``, and run f of seeds 0-1) drew hard
    problems so that no piece of any of *three* cuts starts with 0, and built the agreement
    rule and checks 2 and 2b from three cuts. The five cuts now in use add starting-digit
    constraints at 13-15 digits, so the same seeded stream draws a different sample there.
    Inside this block, ``rounds.draw_hard`` and ``teachers.cuts_of`` use ``THREE_CUTS``
    again (8-12 digits are unchanged either way). Owner's approval, 18 November 2026.
    """
    from gainpred import data, rounds, teachers
    from gainpred.cuts import THREE_CUTS

    saved = rounds.draw_hard, teachers.ADDITION_CUTS
    rounds.draw_hard, teachers.ADDITION_CUTS = partial(data.draw_hard, cuts=THREE_CUTS), THREE_CUTS
    try:
        yield
    finally:
        rounds.draw_hard, teachers.ADDITION_CUTS = saved


def redraw(original: str, out_run: str, runs_dir: Path, data_dir: Path, device: torch.device,
           three_cuts: bool = False, only: list[tuple[int, int]] | None = None) -> None:
    """Every logged check cell of ``original``: redraw, run the teacher, compute check 3.

    ``three_cuts``: draw and cut under the three-cut rule (``three_cut_rule``). ``only``:
    compute only these (round, digits) cells; every sample is still drawn in order, so the
    stream and the excluded problems are the same as in the full redraw.

    Writes ``runs/<out_run>/attempt_<n>/check3.jsonl`` (one ``cell`` event per round and
    length, then ``done``). A retried call starts a new attempt folder and skips rounds an
    earlier attempt finished.
    """
    from gainpred.probe import attempt_dir, events
    from gainpred.rounds import RoundConfig

    src = runs_dir / original
    config = json.loads((src / "config.json").read_text())
    known = {f.name for f in fields(RoundConfig)}
    cfg = RoundConfig(**{k: v for k, v in config["rounds"].items() if k in known})
    teacher = teacher_key(cfg.teacher)
    first_round = (config.get("start_round") or 0) + 1
    logged = {(e["round"], e["digits"]): e for e in events(src) if e["event"] == "checks"}
    out, finished = attempt_dir(runs_dir, out_run)
    log_file = open(out / "check3.jsonl", "a")

    def log(e: dict) -> None:
        log_file.write(json.dumps(e) + "\n")
        log_file.flush()

    log({"event": "config", "of": original, "teacher": teacher, "seed": cfg.seed, "first_round": first_round,
         "three_cuts": three_cuts, "only": only})
    with (three_cut_rule() if three_cuts else _nothing()):
        _redraw_rounds(cfg, config, src, runs_dir, data_dir, device, teacher, first_round, logged, finished,
                       {tuple(c) for c in only} if only else None, log)
    log({"event": "done"})
    log_file.close()


@contextmanager
def _nothing():
    yield


def redrawn_rounds(cfg, config, src, runs_dir, data_dir, device, first_round, logged, finished, only):
    """Per round of the original run: (round, start checkpoint, solver, [(digits, check sample), ...]).

    The model is the one *before* the round's training. Every logged length's sample is drawn
    in order (so the seeded stream and the excluded problems are those of the original run),
    then narrowed to ``only`` (a set of (round, digits), or None for all). Rounds in
    ``finished`` or with nothing left to compute are skipped. Shared with
    ``gainpred.teacher_accuracy`` (amendment 24), which redraws the same samples.
    """
    from gainpred.cuts import HARD_DIGITS
    from gainpred.rounds import MAX_DIGITS_BY_ROUND, ModelSolver, draw_fresh, known_problems, rng_for, \
        start_checkpoint_of
    from gainpred.train import load_model

    exclude = known_problems(data_dir)
    for k in range(first_round, first_round + cfg.rounds):
        lengths = [d for d in range(min(HARD_DIGITS), MAX_DIGITS_BY_ROUND[k - 1] + 1) if (k, d) in logged]
        if not lengths:
            continue
        samples = {d: draw_fresh(rng_for(cfg.seed, k, d, "checks"), d, cfg.check_sample, exclude) for d in lengths}
        if only is not None:
            lengths = [d for d in lengths if (k, d) in only]
        if k in finished or not lengths:
            continue
        start = (start_checkpoint_of(runs_dir, config["start_run"], config.get("start_round")) if k == first_round
                 else sorted((src / f"round_{k - 1}" / "checkpoints").glob("step_*.pt"))[-1])
        model = load_model(start, device)
        solver = ModelSolver(model, device, cfg.solver_batch_size, seed=cfg.seed * 1_000 + k)
        yield k, start, solver, [(d, samples[d]) for d in lengths]
        del model, solver


def _redraw_rounds(cfg, config, src, runs_dir, data_dir, device, teacher, first_round, logged, finished, only, log):
    for k, start, solver, cells in redrawn_rounds(cfg, config, src, runs_dir, data_dir, device, first_round,
                                                  logged, finished, only):
        for d, sample in cells:
            t0 = time.monotonic()
            pairs = teacher_with_pieces(teacher, sample, solver)
            report = run_checks([dec for dec, _ in pairs], solver)
            if device.type == "cuda":
                torch.cuda.synchronize()
            bad, total = unreadable_pieces(pairs)
            kept = [(dec, pieces) for dec, pieces in pairs if dec.kept is not None]
            orig = logged[(k, d)]
            log({"event": "cell", "round": k, "digits": d, "check_3": check_3_addition(pairs),
                 "kept": len(kept), "matches": sum(any(dec.kept == a for a in pieces) for dec, pieces in kept),
                 "unreadable_pieces": bad, "pieces": total, "redrawn": asdict(report),
                 "logged": {key: orig.get(key) for key in ("problems", "kept", "check_1", "check_2b",
                                                           "check_2_kept", "teachable")},
                 "seconds": time.monotonic() - t0, "start_checkpoint": str(start.relative_to(runs_dir))})
        log({"event": "round_done", "round": k})
