"""Launch Modal runs safely: staggered, retried, confirmed, and recorded (runbook section 6).

Usage::

    uv run python -m gainpred.launch rounds si-noselect-seed1 --start start-seed1-stable --teacher no_selection
    uv run python -m gainpred.launch main start-seed1-s9000 --seed 1 --steps 9000

The first argument is the Modal entry point (``main`` = starting training,
``rounds`` = self-improvement), the second the run name; the rest is passed
to the entry point unchanged.

Why this exists
===============
On 13 November three runs "launched" in parallel never started: Modal limits
how fast apps can be created, the retries ran out, and the error was easy to
miss in the log. This module makes that impossible to miss:

* **Never overwrite** (runbook, hard lines): refuses a run name that already
  exists on the Modal volume.
* **Know what code ran**: refuses to launch with uncommitted changes to the
  code (``src/``, ``modal_app.py``), and records the commit.
* **Confirm the start**: waits until the run has passed the data check and
  been handed to its GPU function (the point after which a detached run
  survives this laptop being switched off). Retries on the app-creation
  limit, waiting longer each time.
* **Record everything**: one line per attempt in ``results/launches.jsonl``
  (name, arguments, commit, time, outcome), so failed launches appear in the
  report too.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path("launch-logs")  # local, not in git (see .gitignore)
RECORD = Path("results/launches.jsonl")
STARTED = "datasets match local fingerprints"
RATE_LIMIT = "rate limit exceeded"


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def code_is_committed() -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "src", "modal_app.py"],
                         capture_output=True, text=True, check=True).stdout
    return out.strip() == ""


def runs_on_volume() -> set[str]:
    out = subprocess.run(["uv", "run", "modal", "volume", "ls", "gainpred", "runs"],
                         capture_output=True, text=True, check=True).stdout
    return {line.strip().removeprefix("runs/") for line in out.splitlines() if line.strip().startswith("runs/")}


def record(entry: dict) -> None:
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    with RECORD.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def launch(entrypoint: str, run: str, extra: list[str], attempts: int = 5, wait_seconds: int = 300) -> bool:
    """Launch one run; True once it has reached its GPU function."""
    if not code_is_committed():
        sys.exit("uncommitted changes in src/ or modal_app.py: commit first, so the run's code is known")
    if run in runs_on_volume():
        sys.exit(f"run {run} already exists on the volume; runs are never overwritten (pick a new name)")
    commit = git_commit()
    LOG_DIR.mkdir(exist_ok=True)
    for attempt in range(1, attempts + 1):
        log_path = LOG_DIR / f"{run}.attempt{attempt}.log"
        cmd = ["uv", "run", "modal", "run", "--detach", f"modal_app.py::{entrypoint}", "--run", run, *extra]
        with log_path.open("w") as log:
            # start_new_session: the launch keeps streaming logs after we return, and
            # is not killed with this process.
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + wait_seconds
        outcome = "timeout"
        while time.monotonic() < deadline:
            time.sleep(5)
            text = log_path.read_text(errors="replace")
            if STARTED in text:
                outcome = "started"
                break
            if proc.poll() is not None:  # the launcher exited before the run started
                outcome = "rate-limited" if RATE_LIMIT in text.lower() else "failed"
                break
        record({"run": run, "entrypoint": entrypoint, "args": extra, "commit": commit, "attempt": attempt,
                "outcome": outcome, "log": str(log_path),
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        print(f"{run}: attempt {attempt}: {outcome}")
        if outcome == "started":
            return True
        if outcome != "rate-limited":
            return False
        time.sleep(60 * attempt)  # back off: 1, 2, 3 ... minutes
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("entrypoint", choices=["main", "rounds"])
    parser.add_argument("run")
    args, extra = parser.parse_known_args()
    ok = launch(args.entrypoint, args.run, extra)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
