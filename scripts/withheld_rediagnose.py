"""Rerun the withheld runs' check-2 diagnostic with the current cuts (owner's request, 14 Nov 2026).

The two withheld runs were launched minutes before the diagnostic cuts changed
from join +- 3 to the nearest alternative 6-10 splits. Their checks are
recomputed here on the same check samples (same seeded streams; a collision with
the round's own candidates, which the live run also excluded, is practically
impossible at 12-20 symbols) with the same pre-training model, and compared with
the logged values. Only checks 2 and 2b depend on the cuts.

    uv run python scripts/withheld_rediagnose.py
"""
import json
from pathlib import Path

from gainpred.checks import run_checks_longest_run
from gainpred.model import pick_device
from gainpred.run_data import HARD_LENGTHS
from gainpred.run_teachers import RunModelSolver, join_at_the_cut
from gainpred.run_train import draw_withheld, known_strings, rng_for
from gainpred.train import load_model

device = pick_device()
known = known_strings(Path("data/longest_run"))
out = {}
for s in (0, 1):
    model = load_model(Path(f"runs/runs-start-seed{s}-u30000/checkpoints/step_030000.pt"), device)
    solver = RunModelSolver(model, device, seed=s * 1_000 + 1)
    logged = {e["digits"] if "digits" in e else e["length"]: e for e in map(json.loads, open(
        f"results/runs/runs-join_at_the_cut-withheld-seed{s}-u30000/rounds.jsonl")) if e["event"] == "checks"}
    for length in HARD_LENGTHS:
        sample = draw_withheld(rng_for(s, 1, length, "checks"), length, 2_000, set(known))
        r = run_checks_longest_run(join_at_the_cut(sample, solver), solver)
        old = logged.get(length, {})
        out[f"seed{s} {length}"] = {"check_2_kept_new": r.check_2_kept, "check_2b_new": r.check_2b,
                                    "check_2_kept_old": old.get("check_2_kept"), "check_2b_old": old.get("check_2b"),
                                    "check_1_new": r.check_1, "check_1_old": old.get("check_1")}
        print(f"seed {s} length {length}: check 2 kept {old.get('check_2_kept', float('nan')):.3f} -> {r.check_2_kept:.3f}; "
              f"2b {old.get('check_2b', float('nan')):.3f} -> {r.check_2b:.3f}; check 1 {old.get('check_1', float('nan')):.3f} / {r.check_1:.3f}")
Path("results/withheld_rediagnose.json").write_text(json.dumps(out, indent=1) + "\n")
