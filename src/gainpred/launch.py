"""Launch Modal runs that survive this laptop: deployed, spawned, checked and recorded (runbook section 6).

Usage::

    uv run python -m gainpred.launch main start-seed1-s9000 --seed 1 --steps 9000 --norm-eps 1e-6 --decay-fraction 0.1
    uv run python -m gainpred.launch rounds si-agreement-seed1 --start start-seed1-authors --teacher agreement
    uv run python -m gainpred.launch runs-main runs-start-seed0 --seed 0            # longest run: starting training
    uv run python -m gainpred.launch runs-rounds runs-join-seed0 --start runs-start-seed0 --teacher join_at_the_cut
    uv run python -m gainpred.launch status            # every spawned run: running, done or failed

Why this exists
===============
Two lessons from 13 November:

1. ``modal run --detach`` launched in parallel can die on Modal's app-creation
   limit, unnoticed (three runs never started).
2. Runs started with ``modal run --detach`` **did not survive the laptop
   being switched off**: four runs stopped mid-way overnight.

So runs are now *spawned* on a *deployed* app. ``modal deploy`` uploads the
code once and keeps the app on Modal's side; ``Function.spawn`` starts a call
that runs there independently of this machine, which can then go to sleep.

Safety checks, before every launch
==================================
* **Never overwrite** (runbook, hard lines): refuses a run name that already
  exists on the volume or was spawned before.
* **Know what code ran**: refuses uncommitted changes to ``src/`` or
  ``modal_app.py``, and deploys again whenever the commit has changed, so the
  deployed code is always the recorded commit.
* **Same data**: the cloud datasets' fingerprints must equal the local ones.
* **Record everything**: one line per launch in ``results/launches.jsonl``
  (run, function, arguments, commit, call id, time), used by ``status``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

APP = "gainpred"
RECORD = Path("results/launches.jsonl")
DEPLOYED = Path("launch-logs/deployed_commit")  # local, not in git


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def code_is_committed() -> bool:
    return git("status", "--porcelain", "src", "modal_app.py") == ""


def runs_on_volume() -> set[str]:
    out = subprocess.run(["uv", "run", "modal", "volume", "ls", APP, "runs"],
                         capture_output=True, text=True, check=True).stdout
    return {line.strip().removeprefix("runs/") for line in out.splitlines() if line.strip().startswith("runs/")}


def records() -> list[dict]:
    return [json.loads(line) for line in RECORD.read_text().splitlines()] if RECORD.exists() else []


def record(entry: dict) -> None:
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    with RECORD.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def ensure_deployed(commit: str) -> None:
    """Deploy the app if the deployed code is not this commit."""
    if DEPLOYED.exists() and DEPLOYED.read_text().strip() == commit:
        return
    subprocess.run(["uv", "run", "modal", "deploy", "modal_app.py"], check=True, capture_output=True)
    DEPLOYED.parent.mkdir(exist_ok=True)
    DEPLOYED.write_text(commit + "\n")


DATA = {"addition": ("make_data", Path("data/addition/manifest.json")),
        "longest_run": ("make_run_data", Path("data/longest_run/manifest.json"))}
"""Per task: the Modal function that makes (or reads) the cloud data, and the local manifest to compare."""


def spawn(function: str, run: str, kwargs: dict, task: str = "addition") -> None:
    import modal

    if not code_is_committed():
        sys.exit("uncommitted changes in src/ or modal_app.py: commit first, so the run's code is known")
    if run in runs_on_volume() or any(r.get("run") == run and r.get("call_id") for r in records()):
        sys.exit(f"run {run} already exists; runs are never overwritten (pick a new name)")
    commit = git("rev-parse", "HEAD")
    ensure_deployed(commit)
    data_function, local_manifest = DATA[task]
    if not local_manifest.exists():
        sys.exit(f"no {local_manifest}: generate the {task} data locally first, so there is a manifest to compare")
    remote = modal.Function.from_name(APP, data_function).remote()
    if remote != json.loads(local_manifest.read_text()):
        sys.exit("cloud datasets differ from local ones (manifest fingerprints); not launching")
    call = modal.Function.from_name(APP, function).spawn(run=run, **kwargs)
    record({"run": run, "function": function, "args": kwargs, "commit": commit, "call_id": call.object_id,
            "outcome": "spawned", "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **({} if task == "addition" else {"task": task})})
    print(f"{run}: spawned ({call.object_id}) from commit {commit[:7]}")


def status() -> dict[str, str]:
    """State of every spawned run: running, done, or failed (with the error)."""
    import modal

    states = {}
    for r in records():
        if not r.get("call_id"):
            continue
        call = modal.FunctionCall.from_id(r["call_id"])
        try:
            call.get(timeout=0)
            states[r["run"]] = "done"
        except TimeoutError:
            states[r["run"]] = "running"
        except Exception as e:  # noqa: BLE001 -- any remote failure is reported, not raised
            states[r["run"]] = f"failed: {type(e).__name__}: {str(e)[:120]}"
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("main", help="starting training")
    m.add_argument("run")
    m.add_argument("--seed", type=int, required=True)
    m.add_argument("--steps", type=int, default=10_000)
    m.add_argument("--eval-every", type=int, default=1_000)
    m.add_argument("--checkpoint-every", type=int, default=2_000)
    m.add_argument("--norm-eps", type=float, default=0.0)
    m.add_argument("--decay-fraction", type=float, default=0.2)
    r = sub.add_parser("rounds", help="self-improvement rounds")
    r.add_argument("run")
    r.add_argument("--start", required=True)
    r.add_argument("--teacher", required=True)
    r.add_argument("--rounds", type=int, default=4)
    r.add_argument("--start-round", type=int, default=0)
    r.add_argument("--build-seconds", type=float, default=0.0)
    rm = sub.add_parser("runs-main", help="longest-run starting training")
    rm.add_argument("run")
    rm.add_argument("--seed", type=int, required=True)
    rm.add_argument("--steps", type=int, default=15_000)
    rm.add_argument("--eval-every", type=int, default=1_000)
    rm.add_argument("--checkpoint-every", type=int, default=3_000)
    rm.add_argument("--norm-eps", type=float, default=1e-6)
    rm.add_argument("--decay-fraction", type=float, default=0.1)
    rm.add_argument("--long-runs", action="store_true", help="control arm: add strings with a run of 7-8 (P6)")
    rr = sub.add_parser("runs-rounds", help="one longest-run self-improvement round")
    rr.add_argument("run")
    rr.add_argument("--start", required=True)
    rr.add_argument("--teacher", required=True,
                    choices=["edges_differ", "pick_the_longer", "join_at_the_cut", "no_selection"])
    rr.add_argument("--build-seconds", type=float, default=0.0)
    rr.add_argument("--withheld", action="store_true", help="P6 withheld version (no a = 3-4 run-across problems)")
    rr.add_argument("--long-runs", action="store_true", help="control arm: replay includes the long-run strings")
    sub.add_parser("status", help="state of every spawned run")
    args = parser.parse_args()

    if args.command == "status":
        for run, state in status().items():
            print(f"{run:45s} {state}")
    elif args.command == "main":
        spawn("train_start", args.run, {
            "seed": args.seed, "steps": args.steps, "eval_every": args.eval_every, "eval_per_length": 0,
            "checkpoint_every": args.checkpoint_every, "norm_eps": args.norm_eps,
            "decay_fraction": args.decay_fraction})
    elif args.command == "runs-main":
        spawn("train_start_runs", args.run, {
            "seed": args.seed, "steps": args.steps, "eval_every": args.eval_every,
            "checkpoint_every": args.checkpoint_every, "norm_eps": args.norm_eps,
            "decay_fraction": args.decay_fraction, "long_runs": args.long_runs}, task="longest_run")
    elif args.command == "runs-rounds":
        spawn("self_improve_runs", args.run, {
            "start": args.start, "teacher": args.teacher, "build_seconds": args.build_seconds,
            "withheld": args.withheld, "long_runs": args.long_runs}, task="longest_run")
    else:
        spawn("self_improve", args.run, {
            "start": args.start, "teacher": args.teacher, "rounds": args.rounds,
            "start_round": args.start_round, "build_seconds": args.build_seconds})


if __name__ == "__main__":
    main()
