"""Withheld version of experiment 2: joined runs of 4-6 split by left part (owner's request, 14 Nov 2026).

The withheld pool drops run-across-the-join problems whose left part a is 3 or 4,
so a combination (a = 3-4 on the left of the join) is never taught, although
joined runs of the same length with a = 1-2 are. Scores each model's direct
runlen answer on every run-across study problem with a + b in 4-6, split by
a in {1, 2} versus {3, 4}.

    uv run python scripts/withheld_split.py            # learning seeds 0-1
    uv run python scripts/withheld_split.py 2 3 4      # confirming seeds (15 Nov 2026)

Results for the seeds given are merged into results/withheld_split.json; entries
for other seeds are kept as they are.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from gainpred.formats import RunQuestion, parse_run_answer
from gainpred.model import pick_device
from gainpred.run_teachers import RunModelSolver
from gainpred.train import load_model

device = pick_device()
rows = [json.loads(l) for l in open("data/longest_run/test_study.jsonl")]
rows = [r for r in rows if r["kind"] == "run_across" and r["a"] + r["b"] in (4, 5, 6)]
questions = [RunQuestion("runlen", r["string"]) for r in rows]
models = {
    "before": "runs/runs-start-seed{s}-u30000/checkpoints/step_030000.pt",
    "withheld": "runs/runs-join_at_the_cut-withheld-seed{s}-u30000/round_1/checkpoints/step_003000.pt",
    "full pool": "runs/runs-join_at_the_cut-seed{s}-u30000/round_1/checkpoints/step_003000.pt",
}
seeds = [int(a) for a in sys.argv[1:]] or [0, 1]
out_path = Path("results/withheld_split.json")
out = json.loads(out_path.read_text()) if out_path.exists() else {}
for s in seeds:
    for name, path in models.items():
        answers = RunModelSolver(load_model(Path(path.format(s=s)), device), device)(questions)
        right, total = defaultdict(int), defaultdict(int)
        for r, a in zip(rows, answers):
            key = (r["a"] + r["b"], "1-2" if r["a"] <= 2 else "3-4")
            total[key] += 1
            right[key] += a == parse_run_answer(r["answer"])
        out[f"seed{s} {name}"] = {f"{k[0]}|{k[1]}": (right[k] / total[k], total[k]) for k in sorted(total)}
        print(f"seed {s} {name:10s} " + "  ".join(f"J{k[0]} a{k[1]}: {100*right[k]/total[k]:5.1f}% (n={total[k]})" for k in sorted(total)))
out_path.write_text(json.dumps(out, indent=1) + "\n")
