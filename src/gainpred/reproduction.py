"""Reproduction table (appendix): score evaluation-run checkpoints on the reproduction test set.

What and why
============
Shin et al. test addition on 100 problems per length, 3 to 23 digits. P7 set
aside our own version of that set, ``test_reproduction`` (100 per length,
3 to 15 digits, the lengths our four rounds reach), so our numbers can be set
beside theirs. No run scored it during training; this module does it
afterwards, on saved checkpoints. It is inference only: nothing is trained and
no logged file is touched.

Which checkpoints
=================
The evaluation runs (seeds 2-4) of the two teachers closest to Shin et al.'s
setup:

* the no-carry rule (their *filtered* composition), and
* no selection (their *unfiltered* composition).

For each run: the starting model (round 0) and the model after each of the four
rounds. The two teachers of a seed share their starting model, which is scored
once: 3 + 6 x 4 = 27 checkpoints.

Identity check
==============
Each checkpoint is also scored on ``test_easy``, which every run logged after
every round. Locally, ``compare_easy`` checks that the two agree; if they did
not, the checkpoint would not be the model the run logged.

Output: ``runs/<name>/attempt_<n>/reproduction.jsonl``, one ``checkpoint``
event per model and a final ``done``. A retried Modal call starts a new attempt
folder and skips checkpoints an earlier attempt finished.
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = [f"si-{teacher}-seed{seed}-authors" for teacher in ("nocarry", "noselect") for seed in (2, 3, 4)]


def checkpoints(runs_dir: Path, runs: list[str] = RUNS) -> list[tuple[str, str, int, Path]]:
    """(scored-as run, starting model, round, checkpoint path) for every model to score.

    Round 0 is the starting model, listed once per starting model under the
    first run that uses it. Round k is the last checkpoint saved in round k,
    the model the run evaluated after that round.
    """
    from gainpred.rounds import start_checkpoint_of

    out, seen = [], set()
    for run in runs:
        config = json.loads((runs_dir / run / "config.json").read_text())
        start = config["start_run"]
        if start not in seen:
            seen.add(start)
            out.append((run, start, 0, start_checkpoint_of(runs_dir, start)))
        for k in range(1, config["rounds"]["rounds"] + 1):
            out.append((run, start, k, start_checkpoint_of(runs_dir, run, k)))
    return out


def attempt_folder(runs_dir: Path, name: str) -> tuple[Path, set[str]]:
    """A fresh attempt folder, and the checkpoints earlier attempts already scored."""
    base = runs_dir / name
    base.mkdir(parents=True, exist_ok=True)
    earlier = sorted(base.glob("attempt_*/reproduction.jsonl"))
    done = {e["checkpoint"] for log in earlier for e in map(json.loads, log.read_text().splitlines())
            if e["event"] == "checkpoint"}
    out = base / f"attempt_{len(earlier) + 1}"
    out.mkdir()
    return out, done


def score(name: str, runs_dir: Path, data_dir: Path, device, runs: list[str] = RUNS) -> None:
    """Score every checkpoint on test_reproduction (and test_easy), greedy, as in training."""
    from gainpred.data import load
    from gainpred.train import by_length, evaluate, load_model

    repro = load(data_dir / "test_reproduction.jsonl")
    easy = load(data_dir / "test_easy.jsonl")
    out, done = attempt_folder(runs_dir, name)
    with open(out / "reproduction.jsonl", "a") as f:
        for run, start, round_, path in checkpoints(runs_dir, runs):
            ckpt = str(path.relative_to(runs_dir))
            if ckpt in done:
                continue
            model = load_model(path, device)
            model.eval()
            event = {"event": "checkpoint", "run": run, "start": start, "round": round_, "checkpoint": ckpt,
                     "reproduction": evaluate(model, repro, by_length, 1_000, device),
                     "easy": evaluate(model, easy, by_length, 1_000, device)}
            f.write(json.dumps(event) + "\n")
            f.flush()
            print(f"{run} round {round_}: reproduction all {event['reproduction']['all']:.1%}")
        f.write(json.dumps({"event": "done"}) + "\n")


def compare_easy(events: list[dict], runs_dir: Path) -> list[str]:
    """Checkpoints whose easy-set accuracy differs from the one their run logged (should be none)."""
    bad = []
    for e in events:
        log = [json.loads(line) for line in (runs_dir / e["run"] / "rounds.jsonl").read_text().splitlines()]
        logged = next(x["accuracy"] for x in log
                      if x["event"] == "eval" and x["set"] == "easy" and x["round"] == e["round"])
        if any(abs(logged[k] - e["easy"][k]) > 1e-9 for k in logged):
            bad.append(f"{e['run']} round {e['round']}: logged {logged['all']:.4f}, now {e['easy']['all']:.4f}")
    return bad


RESULTS = Path("results/reproduction.jsonl")


def wall(name: str) -> dict:
    """Seconds from the spawn (results/launches.jsonl) to the last write on the volume."""
    import subprocess
    from datetime import datetime

    spawned = next(json.loads(line)["time"] for line in Path("results/launches.jsonl").read_text().splitlines()
                   if json.loads(line)["run"] == name)
    listing = json.loads(subprocess.run(["uv", "run", "modal", "volume", "ls", "--json", "gainpred",
                                         f"runs/{name}/attempt_1"], capture_output=True, text=True,
                                        check=True).stdout)
    last = max(datetime.fromisoformat(f["created_modified"]) for f in listing)
    return {"event": "wall", "seconds": (last - datetime.fromisoformat(spawned)).total_seconds(),
            "source": "spawn time to the last write on the volume, start-up included"}


def fetch(name: str = "reproduction-eval") -> list[dict]:
    """Download the call's folder, merge its attempts into ``results/reproduction.jsonl``.

    Refuses (raises) unless all 27 checkpoints were scored and the easy-set identity check
    passes, so a partial or mismatched result never reaches the paper.
    """
    import subprocess

    subprocess.run(["uv", "run", "modal", "volume", "get", "--force", "gainpred", f"runs/{name}", "runs/"],
                   capture_output=True, check=True)
    lines = [json.loads(line) for log in sorted((Path("runs") / name).glob("attempt_*/reproduction.jsonl"))
             for line in log.read_text().splitlines()]
    events = [e for e in lines if e["event"] == "checkpoint"]
    want = {str(p.relative_to(Path("runs"))) for *_, p in checkpoints(Path("runs"))}
    if {e["checkpoint"] for e in events} != want or len(events) != len(want):
        raise ValueError(f"{len(events)} checkpoints scored, {len(want)} wanted")
    bad = compare_easy(events, Path("runs"))
    if bad:
        raise ValueError("easy-set accuracy differs from the logged one:\n" + "\n".join(bad))
    RESULTS.write_text("".join(json.dumps(e) + "\n" for e in events))
    # The raw log, for the run list and the spending estimate (gainpred.status), plus its wall time.
    raw = Path("results/reproduction") / f"{name}.jsonl"
    raw.parent.mkdir(exist_ok=True)
    raw.write_text("".join(json.dumps(e) + "\n" for e in lines) + json.dumps(wall(name)) + "\n")
    print(f"{RESULTS}: {len(events)} checkpoints; easy-set identity check passed")
    return events


if __name__ == "__main__":
    fetch()
