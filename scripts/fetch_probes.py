"""Fetch finished probes, in the order the owner set (amendment 20).

    uv run python scripts/fetch_probes.py learning     # seeds 0-1 only
    uv run python scripts/fetch_probes.py confirming   # refused until G, H, D are recorded and committed

Downloads each probe's folder into runs/<probe>/ and writes all its attempts' logs, merged, to
results/probes/<probe>.jsonl. Failed probes' partial output is fetched too (it is checked, never scored). Then checks the rebuild against the stop rule: any round whose
lower bound on differing kept problems passes 1% of kept problems is listed and the script
exits non-zero.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from probes import plan  # noqa: E402

group = sys.argv[1]
assert group in ("learning", "confirming")


def seed_of(name: str) -> int:
    return int(re.search(r"seed(\d)", name).group(1))


if group == "confirming":
    levels = Path("results/baseline_levels.json")
    committed = subprocess.run(["git", "ls-files", "--error-unmatch", str(levels)], capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(levels)]).returncode == 0
    if not (levels.exists() and committed and clean):
        sys.exit("G, H, D are not recorded and committed yet: confirming probes are not fetched (amendment 20)")

states = {}
out = subprocess.run(["uv", "run", "python", "-m", "gainpred.launch", "status"], capture_output=True, text=True).stdout
for line in out.splitlines():
    if line.strip():
        states[line.split()[0]] = line.split(None, 1)[1].strip()

names = [p for p, _, _ in plan()] + [n for n in states if n.startswith("probe-") and n.endswith("-r2")]
wanted = sorted({p for p in names if (seed_of(p) in (0, 1)) == (group == "learning")})
bad, waiting = [], []
Path("results/probes").mkdir(exist_ok=True)
for p in wanted:
    state = states.get(p, "not launched")
    if state == "running" or state == "not launched":
        waiting.append(f"{p}: {state}")
        continue
    if state != "done":
        # A failed probe: its partial output is kept for the record and checked against the
        # stop rule, but never scored (a failed probe is replaced as a whole by its -r2 rerun).
        waiting.append(f"{p}: {state[:60]} (partial output kept, not scored)")
    dest = Path("runs") / p
    if not dest.exists():
        subprocess.run(["uv", "run", "modal", "volume", "get", "gainpred", f"runs/{p}", "runs/"], capture_output=True)
    logs = [dest / "probe.jsonl", *sorted(dest.glob("attempt_*/probe.jsonl"))]
    merged = "".join(x.read_text() for x in logs if x.exists())
    (Path("results/probes") / f"{p}.jsonl").write_text(merged)
    for line in merged.splitlines():
        e = json.loads(line)
        if e["event"] == "probe_build" and e["differ_at_least_share"] > 0.01:
            bad.append(f"{p} round {e['round']} length {e.get('digits', e.get('length'))}: "
                       f"at least {e['differ_at_least']} of {e['original_kept']} differ")
print(f"{group}: {len(wanted) - len(waiting)} of {len(wanted)} probes fetched")
for w in waiting:
    print("  waiting:", w)
if bad:
    print("STOP: rebuilt training data differs by more than 1% of kept problems:")
    for b in bad:
        print("  ", b)
    sys.exit(2)
