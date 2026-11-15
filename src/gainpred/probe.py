"""The "train briefly and look" baseline: rebuild a round's training data, train 200 updates, measure.

Usage (on Modal, through the launcher; one call per original run)::

    uv run python -m gainpred.launch probe probe-si-nocarry-seed2-authors --of si-nocarry-seed2-authors
    uv run python -m gainpred.launch probe probe-runs-edges_differ-seed2-u30000 --of runs-edges_differ-seed2-u30000

Writes ``runs/<probe run>/probe.jsonl`` and one training folder per round.

What the baseline is (owner, after decision point 2, 15 November 2026)
======================================================================
The plan promised a cost table per round that includes "the lazy alternative of
training briefly (200 updates) and looking at the result". Our checks predict a
round's effect *before* any training. The lazy alternative instead *does* a
little of the training and measures accuracy on the test groups, which needs
the reference answers of those test problems. That is the difference to keep in
mind: the baseline looks at right answers; our checks never do.

What a probe does, for each round ``k`` of an original run
==========================================================
1. Load the model the original round ``k`` started from (the run's starting
   model for its first round, else its checkpoint after round ``k - 1``).
2. **Rebuild the teacher's training data** exactly as the original round did:
   the same teacher, the same candidate streams (``rng_for``), the same
   solver seed, the same chunking. Problems the original run had drawn in its
   earlier rounds are excluded again where that can change the stream (the easy
   replay problems; the hard lengths' earlier draws cannot collide in practice
   and are not re-drawn). A build the original stopped by its time limit
   (sample and vote, 10 minutes per length) stops instead at the original's
   number of drawn candidates, so the same candidates are seen.
3. **Compare** the rebuilt kept answers with the original's. The original kept
   problems were not stored, so the comparison uses what the original logged
   per length: candidates drawn, answers kept, the share right, the carries
   per place (addition) or kinds kept (longest run), and the tokens generated.
   ``differ_at_least`` is a lower bound on how many kept problems differ:
   swapping one kept problem for another changes each logged count by at most
   one, so the largest count difference is a lower bound.
4. Replay as the original round did, then **train 200 updates** with the
   round's own schedule (3,000-update schedule, stopped at 200: at that point
   the learning rate is still at its peak), batch order unchanged.
5. **Evaluate** on the same test groups as the round (easy, study, study5).
   The checks are skipped. Every part is timed.
"""

from __future__ import annotations

import json
import time
from dataclasses import fields, replace
from pathlib import Path

import torch

from gainpred.cuts import EASY_DIGITS, HARD_DIGITS
from gainpred.formats import solve
from gainpred.train import evaluate, load_model, train

PROBE_STEPS = 200


def events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "rounds.jsonl").read_text().splitlines()]


def compare(original: dict, rebuilt: dict, count_keys: list[str]) -> dict:
    """Logged fingerprints of one length's build, original against rebuilt, and the lower bound."""
    def right_count(b):
        return round((b.get("kept_right") or 0.0) * b["kept"])
    diffs = [abs(original["kept"] - rebuilt["kept"]), abs(right_count(original) - right_count(rebuilt))]
    for key in count_keys:
        a, b = original.get(key), rebuilt.get(key)
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            diffs += [abs(x - y) for x, y in zip(a, b)]
        elif isinstance(a, dict) and isinstance(b, dict):
            diffs += [abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)]
    same = all(original.get(k) == rebuilt.get(k) for k in ["drawn", "kept", "generated_tokens", *count_keys]
               if k in original)
    return {"original_drawn": original["drawn"], "original_kept": original["kept"],
            "original_tokens": original.get("generated_tokens"),
            "fingerprint_identical": same, "differ_at_least": max(diffs),
            "differ_at_least_share": max(diffs) / max(original["kept"], 1)}


def probe_addition(original: str, probe: str, runs_dir: Path, data_dir: Path, device: torch.device,
                   only_rounds: list[int] | None = None) -> None:
    """Probe every round of ``original``, or only ``only_rounds`` (e.g. rounds 1-2 of a learning run
    whose rounds 3-4 were superseded by its five-cut rerun)."""
    from gainpred.rounds import (MAX_DIGITS_BY_ROUND, ModelSolver, RoundConfig, TEACHERS, carry_positions,
                                 collect_kept, draw_fresh, eval_sets, known_problems, rng_for, start_checkpoint_of)

    src = runs_dir / original
    config = json.loads((src / "config.json").read_text())
    known = {f.name for f in fields(RoundConfig)}
    cfg = RoundConfig(**{k: v for k, v in config["rounds"].items() if k in known})
    first_round = (config.get("start_round") or 0) + 1
    log_original = events(src)
    out = runs_dir / probe
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps({"probe_of": original, "steps": PROBE_STEPS, "rounds": config["rounds"],
                                                 "only_rounds": only_rounds, "task": "addition/probe"}, indent=2) + "\n")
    log_file = open(out / "probe.jsonl", "a")

    def log(e: dict) -> None:
        log_file.write(json.dumps(e) + "\n")
        log_file.flush()

    tests = eval_sets(data_dir, cfg.eval_per_length)
    for k in range(first_round, first_round + cfg.rounds):
        if only_rounds and k not in only_rounds:
            continue
        start = (start_checkpoint_of(runs_dir, config["start_run"], config.get("start_round")) if k == first_round
                 else sorted((src / f"round_{k - 1}" / "checkpoints").glob("step_*.pt"))[-1])
        model = load_model(start, device)
        exclude = known_problems(data_dir)
        for r in range(first_round, k):  # earlier rounds' replay draws (easy problems can collide)
            for d in EASY_DIGITS:
                draw_fresh(rng_for(cfg.seed, r, d, "replay"), d, cfg.replay_per_length, exclude)
        solver = ModelSolver(model, device, cfg.solver_batch_size, seed=cfg.seed * 1_000 + k)
        teacher = TEACHERS[cfg.teacher]
        seconds = {"build": 0.0, "replay": 0.0, "train": 0.0, "eval": 0.0}
        examples = []
        for digits in range(min(HARD_DIGITS), MAX_DIGITS_BY_ROUND[k - 1] + 1):
            orig = next(e for e in log_original if e["event"] == "build" and e["round"] == k
                        and e["digits"] == digits)
            cap = orig["drawn"] if orig.get("stopped_by_time") else cfg.draw_cap_factor * cfg.kept_per_length
            tokens_before, t0 = solver.generated_tokens, time.monotonic()
            kept, drawn, _ = collect_kept(teacher, solver, rng_for(cfg.seed, k, digits, "build"), digits,
                                          cfg.kept_per_length, cap, exclude, None, cfg.max_chunk)
            if device.type == "cuda":
                torch.cuda.synchronize()  # time finished GPU work, not queued work
            took = time.monotonic() - t0
            seconds["build"] += took
            rebuilt = {"drawn": drawn, "kept": len(kept),
                       "kept_right": sum(a == solve(p) for p, a in kept) / len(kept) if kept else None,
                       "carries_by_place": carry_positions([p for p, _ in kept]),
                       "generated_tokens": solver.generated_tokens - tokens_before}
            log({"event": "probe_build", "round": k, "digits": digits, **rebuilt, "seconds": took,
                 **compare(orig, rebuilt, ["carries_by_place"])})
            examples += kept
        t0 = time.monotonic()
        replay_problems = [p for d in EASY_DIGITS
                           for p in draw_fresh(rng_for(cfg.seed, k, d, "replay"), d, cfg.replay_per_length, exclude)]
        replay_answers = solver(replay_problems)
        examples += [(p, a) for p, a in zip(replay_problems, replay_answers) if isinstance(a, int)]
        seconds["replay"] = time.monotonic() - t0
        tcfg = replace(cfg.train_config(k), stop_after=PROBE_STEPS, save_checkpoints=False,
                       eval_every=10 ** 9)
        t0 = time.monotonic()
        model = train(model, examples, tcfg, out / f"round_{k}", {}, device,
                      extra_config={"task": "addition/probe", "probe_of": original, "round": k})
        seconds["train"] = time.monotonic() - t0
        t0 = time.monotonic()
        model.eval()
        for name, eval_set in tests.items():
            log({"event": "eval", "round": k, "set": name,
                 "accuracy": evaluate(model, eval_set.records, eval_set.group_key, 1_000, device)})
        seconds["eval"] = time.monotonic() - t0
        log({"event": "probe_done", "round": k, "examples": len(examples), "steps": PROBE_STEPS,
             "seconds": seconds, "generated_tokens": solver.generated_tokens})
        print(f"{probe} round {k}: {seconds}")
    log({"event": "done"})
    log_file.close()


def probe_runs(original: str, probe: str, runs_dir: Path, data_dir: Path, device: torch.device) -> None:
    """The same for a longest-run round (one round; replay uses true answers, so nothing to rebuild there)."""
    from gainpred.rounds import collect_kept
    from gainpred.run_train import (RUN_TEACHERS, RunModelSolver, RunRoundConfig, draw_fresh, draw_withheld,
                                    eval_sets, evaluate_runs, known_strings, rng_for, start_examples, start_rows)
    from gainpred.formats import longest_run
    from gainpred.rounds import start_checkpoint_of
    from gainpred.run_data import HARD_LENGTHS

    src = runs_dir / original
    config = json.loads((src / "config.json").read_text())
    known = {f.name for f in fields(RunRoundConfig)}
    raw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in config["rounds"].items() if k in known}
    cfg = RunRoundConfig(**raw)
    log_original = events(src)
    out = runs_dir / probe
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps({"probe_of": original, "steps": PROBE_STEPS, "rounds": config["rounds"],
                                                 "task": "longest_run/probe"}, indent=2) + "\n")
    log_file = open(out / "probe.jsonl", "a")

    def log(e: dict) -> None:
        log_file.write(json.dumps(e) + "\n")
        log_file.flush()

    k = 1
    model = load_model(start_checkpoint_of(runs_dir, config["start_run"]), device)
    exclude = known_strings(data_dir)
    solver = RunModelSolver(model, device, cfg.solver_batch_size, seed=cfg.seed * 1_000 + k)
    teacher = RUN_TEACHERS[cfg.teacher]
    seconds = {"build": 0.0, "replay": 0.0, "train": 0.0, "eval": 0.0}
    examples = []
    for length in HARD_LENGTHS:
        orig = next(e for e in log_original if e["event"] == "build" and e["round"] == k and e["length"] == length)
        cap = orig["drawn"] if orig.get("stopped_by_time") else cfg.draw_cap_factor * cfg.kept_per_length
        tokens_before, t0 = solver.generated_tokens, time.monotonic()
        kept, drawn, _ = collect_kept(teacher, solver, rng_for(cfg.seed, k, length, "build"), length,
                                      cfg.kept_per_length, cap, exclude, None, cfg.max_chunk,
                                      draw=draw_withheld if cfg.withheld else draw_fresh)
        took = time.monotonic() - t0
        seconds["build"] += took
        rebuilt = {"drawn": drawn, "kept": len(kept),
                   "kept_right": sum(a == longest_run(p.string) for p, a in kept) / len(kept) if kept else None,
                   "kept_by_kind": {kind: sum(p.kind == kind for p, _ in kept) for kind in ("random_join", "run_across")},
                   "generated_tokens": solver.generated_tokens - tokens_before}
        log({"event": "probe_build", "round": k, "length": length, **rebuilt, "seconds": took,
             **compare(orig, rebuilt, ["kept_by_kind"])})
        examples += [(p.question, a) for p, a in kept]
    examples += start_examples(start_rows(data_dir, cfg.long_runs), cfg.replay_questions)
    tcfg = replace(cfg.train_config(k), stop_after=PROBE_STEPS, save_checkpoints=False, eval_every=10 ** 9)
    t0 = time.monotonic()
    model = train(model, examples, tcfg, out / f"round_{k}", {}, device,
                  extra_config={"task": "longest_run/probe", "probe_of": original, "round": k})
    seconds["train"] = time.monotonic() - t0
    t0 = time.monotonic()
    model.eval()
    for name, eval_set in eval_sets(data_dir, cfg.eval_per_length).items():
        log({"event": "eval", "round": k, "set": name,
             "accuracy": evaluate_runs(model, eval_set.records, eval_set.group_key, 1_000, device)})
    seconds["eval"] = time.monotonic() - t0
    log({"event": "probe_done", "round": k, "examples": len(examples), "steps": PROBE_STEPS, "seconds": seconds,
         "generated_tokens": solver.generated_tokens})
    log({"event": "done"})
    log_file.close()
