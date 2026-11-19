"""The status section at the top of results/README.md (runbook section 6).

Usage::

    uv run python -m gainpred.status          # rebuild the section (after gainpred.report)

What it shows
=============
* **Every run** the project has ever started, from three sources: the
  launch record (``results/launches.jsonl``), the copied logs
  (``results/runs/``) and hand-written notes for runs from before the launcher
  (``results/run_notes.json``: died, stopped, superseded). Nothing is dropped:
  failed and parked runs stay listed (runbook, hard lines).
* **Spending**, estimated from the seconds each run logged, at an A10 price.
  An estimate, not Modal's bill: it leaves out container start-up and the
  time between log lines of a run that died.
* **Tripwires** (runbook section 4), checked automatically where the data
  allows; each is marked ok, TRIPPED, or not yet checkable.
* **Decisions waiting** for the owner, from ``results/decisions_waiting.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

from gainpred.report import RESULTS_DIR, RoundsRun, Run, load_all, load_rounds

A10_DOLLARS_PER_HOUR = 1.10  # Modal's A10 list price, November 2026
BUDGET, WARN_AT = 80.0, None  # runbook section 1 (raised to $80 on 15 Nov 2026; no notice level)
START, END = "<!-- status:start -->", "<!-- status:end -->"


def gpu_seconds_start(run: Run) -> float:
    if run.done:
        return sum(run.done["seconds"].values())
    return run.train[-1]["seconds"].get("train", 0.0) if run.train else 0.0


def gpu_seconds_rounds(run: RoundsRun, results_dir: Path) -> float:
    total = 0.0
    for e in run.of("round_done"):
        total += sum(e["seconds"].values())
    for log in sorted((results_dir / "runs" / run.name).glob("round_*/log.jsonl")):
        lines = log.read_text().splitlines()
        if lines:
            total += json.loads(lines[-1]).get("seconds", {}).get("train", 0.0)
    return total + 60 * (len(run.of("eval")) // 2)  # evaluations are not timed; ~1 minute each


def gpu_seconds_probe(path: Path) -> float:
    """A "train briefly and look" probe: finished rounds' timed parts, plus the builds of a round
    that was cut off (a stopped container); its start-up and untimed tail are not counted."""
    total, done_rounds, partial = 0.0, set(), {}
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e["event"] == "probe_done":
            total += sum(e["seconds"].values())
            done_rounds.add(e["round"])
        elif e["event"] == "probe_build":
            partial[e["round"]] = partial.get(e["round"], 0.0) + e["seconds"]
    return total + sum(v for k, v in partial.items() if k not in done_rounds)


def gpu_seconds_check3(path: Path) -> float:
    """Check 3 on addition (amendment 22, inference only): the timed cells; loading checkpoints and
    the container's start-up are not timed, as for probes."""
    return sum(json.loads(line).get("seconds", 0.0) for line in path.read_text().splitlines()
               if '"event": "cell"' in line)


def gpu_seconds_reproduction(path: Path) -> float:
    """Reproduction scoring (inference only): its log has no timings, so ``gainpred.reproduction.fetch``
    adds a ``wall`` event, spawn time to the last write on the volume (start-up included)."""
    return sum(json.loads(line)["seconds"] for line in path.read_text().splitlines() if '"event": "wall"' in line)


def tripwires(starts: list[Run], rounds: list[RoundsRun]) -> list[tuple[str, str]]:
    out = []
    # 1. The no-carry rule below 90% at any length in a finished four-round run.
    finished = [r for r in rounds if r.config["rounds"]["teacher"] in ("no_carry", "their_rule")
                and len(r.of("round_done")) == 4]
    low = []
    for r in finished:
        final = [e for e in r.of("eval") if e["set"] == "study"][-1]["accuracy"]
        for d in range(8, 16):
            mean = (final[f"{d}/no-carry"] + final[f"{d}/carry"]) / 2
            if mean < 0.90:
                low.append(f"{r.name} {d} digits {100 * mean:.1f}%")
    out.append(("No-carry rule below 90% at a length (four-round runs)",
                ("TRIPPED: " + "; ".join(low)) if low else f"ok ({len(finished)} runs checked)"))
    # 2. A standard starting model fails check 0.
    standard = [r for r in starts if r.name in ("start-seed0-authors", "start-seed1-authors")
                or (r.name.startswith("start-seed") and r.name.endswith("-authors") and r.config["train"]["steps"] == 10_000)]
    failing = [f"{r.name} {100 * r.evals_of('easy')[-1]['accuracy']['all']:.1f}%" for r in standard
               if r.evals_of("easy") and r.evals_of("easy")[-1]["accuracy"]["all"] < 0.99]
    out.append(("A standard starting model under 99% on easy problems",
                ("TRIPPED: " + "; ".join(failing)) if failing else f"ok ({len(standard)} models checked)"))
    # 3. Overlap between test and training problems: guaranteed by construction (P7);
    #    the generators' exclusion is tested in tests/test_data.py and tests/test_rounds.py.
    out.append(("Overlap between test and training problems",
                "ok by construction (exclusion tested); kept round problems are not stored, so not re-checked"))
    # 4. A rerun differs from its original by more than 3 points on any group.
    pairs = [("si-their-seed1-r1", "si-nocarry-seed1"), ("si-nocarry-seed1", "si-nocarry-seed1-v2")]
    by_name = {r.name: r for r in rounds}
    diffs = []
    for a, b in pairs:
        if a in by_name and b in by_name:
            ea = {e["round"]: e["accuracy"] for e in by_name[a].of("eval") if e["set"] == "study"}
            eb = {e["round"]: e["accuracy"] for e in by_name[b].of("eval") if e["set"] == "study"}
            for rnd in sorted(set(ea) & set(eb) - {0}):
                worst = max(abs(ea[rnd][g] - eb[rnd][g]) for g in ea[rnd] if g != "all")
                diffs.append((a, b, rnd, worst))
    over = [f"{a} vs {b} round {r}: {100 * w:.1f} points" for a, b, r, w in diffs if w > 0.03]
    out.append(("A rerun differs from its original by over 3 points",
                ("TRIPPED: " + "; ".join(over)) if over else
                "ok (" + ", ".join(f"{a} vs {b} round {r}: max {100 * w:.1f}" for a, b, r, w in diffs) + ")"))
    # 5. Checks computed after training: in rounds.jsonl, each round's checks must come before its eval.
    late = []
    for r in rounds:
        order = [(e["event"], e.get("round")) for e in r.events]
        for rnd in {e["round"] for e in r.of("checks")}:
            first_eval = next((i for i, (k, x) in enumerate(order) if k == "eval" and x == rnd), None)
            last_check = max(i for i, (k, x) in enumerate(order) if k == "checks" and x == rnd)
            if first_eval is not None and last_check > first_eval:
                late.append(f"{r.name} round {rnd}")
    out.append(("Checks computed after training", ("TRIPPED: " + "; ".join(late)) if late else "ok (log order checked)"))
    return out


def collect(results_dir: Path = RESULTS_DIR):
    """Every run's state, GPU seconds, hand-written note and launch record.

    Shared by the status section and the write-up's complete run list
    (``gainpred.writeup``), so both list exactly the same runs.
    Returns ``(starts, rounds, states, seconds, notes, launches)``.
    """
    starts, rounds = load_all(results_dir), load_rounds(results_dir)
    notes = json.loads((results_dir / "run_notes.json").read_text()) if (results_dir / "run_notes.json").exists() else {}
    launches: dict[str, dict] = {}
    if (results_dir / "launches.jsonl").exists():
        for line in (results_dir / "launches.jsonl").read_text().splitlines():
            e = json.loads(line)
            launches.setdefault(e["run"], {}).update(e)
    seconds = {r.name: gpu_seconds_start(r) for r in starts}
    seconds |= {r.name: gpu_seconds_rounds(r, results_dir) for r in rounds}
    seconds |= {p.stem: gpu_seconds_probe(p) for p in sorted((results_dir / "probes").glob("*.jsonl"))}
    seconds |= {p.stem: gpu_seconds_check3(p) for folder in ("check3", "check3_3cut")
                for p in sorted((results_dir / folder).glob("*.jsonl"))}
    seconds |= {p.stem: gpu_seconds_reproduction(p) for p in sorted((results_dir / "reproduction").glob("*.jsonl"))}
    states = {}
    for r in starts:
        states[r.name] = "done" if r.done else "incomplete"
    for r in rounds:
        states[r.name] = "done" if r.of("done") else f"incomplete ({len(r.of('round_done'))} rounds)"
    # Probes and check-3 calls log to results/probes and results/check3; a fetched log with a
    # "done" event is a finished call (before this, finished ones were listed as running).
    for folder in ("probes", "check3", "check3_3cut", "reproduction"):
        for p in sorted((results_dir / folder).glob("*.jsonl")):
            if '"event": "done"' in p.read_text():
                states[p.stem] = "done"
    for name, e in launches.items():
        if name not in states or states[name] != "done":
            states[name] = e.get("outcome", "?") if e.get("outcome") != "spawned" else states.get(name, "running")
    for name, n in notes.items():
        states[name] = n["state"]
    return starts, rounds, states, seconds, notes, launches


def build(results_dir: Path = RESULTS_DIR) -> str:
    starts, rounds, states, seconds, notes, launches = collect(results_dir)
    spent = sum(seconds.values()) / 3600 * A10_DOLLARS_PER_HOUR

    lines = [START, "## Status", "",
             "Generated by `uv run python -m gainpred.status`; do not edit by hand. Rules: `docs/runbook.md`.", "",
             f"**Spending (estimate):** ${spent:.2f} of the ${BUDGET:.0f} budget"
             + (f" — **over the ${WARN_AT:.0f} notice level**" if WARN_AT and spent >= WARN_AT else "") + ".", ""]
    lines += ["**Tripwires**", "", "| Tripwire | State |", "| --- | --- |"]
    lines += [f"| {name} | {state} |" for name, state in tripwires(starts, rounds)]
    waiting = results_dir / "decisions_waiting.md"
    lines += ["", "**Decisions waiting for the owner**", "",
              waiting.read_text().strip() if waiting.exists() else "None.", ""]
    lines += ["**Every run**", "", "| Run | State | GPU minutes | Note |", "| --- | --- | --- | --- |"]
    for name in sorted(states):
        lines.append(f"| `{name}` | {states[name]} | {seconds.get(name, 0) / 60:.0f} | {notes.get(name, {}).get('note', '')} |")
    lines += ["", END]
    return "\n".join(lines)


def main() -> None:
    readme = RESULTS_DIR / "README.md"
    text = readme.read_text()
    section = build()
    if START in text:
        text = text[: text.index(START)] + section + text[text.index(END) + len(END):]
    else:
        title, _, rest = text.partition("\n")
        text = f"{title}\n\n{section}\n{rest}"
    readme.write_text(text)
    print(section)


if __name__ == "__main__":
    main()
