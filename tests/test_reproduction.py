"""The reproduction scoring: which checkpoints are scored, and the identity check."""
import json
from pathlib import Path

import torch

from gainpred.reproduction import checkpoints, compare_easy, score


def fake_runs(tmp: Path) -> Path:
    """Two runs on one starting model, two rounds each; checkpoints are empty files."""
    for start in ("start-a",):
        (tmp / start / "checkpoints").mkdir(parents=True)
        for step in (1000, 2000):
            (tmp / start / "checkpoints" / f"step_{step:06d}.pt").touch()
    for run in ("si-x", "si-y"):
        (tmp / run).mkdir()
        (tmp / run / "config.json").write_text(json.dumps({"start_run": "start-a", "rounds": {"rounds": 2}}))
        for k in (1, 2):
            (tmp / run / f"round_{k}" / "checkpoints").mkdir(parents=True)
            for step in (500, 3000):
                (tmp / run / f"round_{k}" / "checkpoints" / f"step_{step:06d}.pt").touch()
    return tmp


def test_starting_model_once_and_last_checkpoint_of_each_round(tmp_path):
    runs = fake_runs(tmp_path)
    got = [(r, k, p.relative_to(runs).as_posix()) for r, _, k, p in checkpoints(runs, ["si-x", "si-y"])]
    assert got == [("si-x", 0, "start-a/checkpoints/step_002000.pt"),
                   ("si-x", 1, "si-x/round_1/checkpoints/step_003000.pt"),
                   ("si-x", 2, "si-x/round_2/checkpoints/step_003000.pt"),
                   ("si-y", 1, "si-y/round_1/checkpoints/step_003000.pt"),
                   ("si-y", 2, "si-y/round_2/checkpoints/step_003000.pt")]


def test_the_real_plan_is_27_checkpoints():
    if not Path("runs/si-nocarry-seed2-authors/config.json").exists():
        return  # runs/ is local only
    plan = checkpoints(Path("runs"))
    assert len(plan) == 27 and sum(k == 0 for _, _, k, _ in plan) == 3


def test_compare_easy_flags_a_mismatch(tmp_path):
    (tmp_path / "si-x").mkdir()
    (tmp_path / "si-x" / "rounds.jsonl").write_text(
        json.dumps({"event": "eval", "round": 1, "set": "easy", "accuracy": {"3": 1.0, "all": 1.0}}) + "\n")
    same = {"run": "si-x", "round": 1, "easy": {"3": 1.0, "all": 1.0}}
    assert compare_easy([same], tmp_path) == []
    assert len(compare_easy([{**same, "easy": {"3": 0.99, "all": 0.99}}], tmp_path)) == 1


def test_a_retry_skips_what_an_earlier_attempt_scored(tmp_path, monkeypatch):
    runs = fake_runs(tmp_path / "runs")
    data = tmp_path / "data"
    data.mkdir()
    for name in ("test_reproduction", "test_easy"):
        (data / f"{name}.jsonl").write_text(json.dumps({"a": 1, "b": 2, "answer": 3, "digits": 1}) + "\n")
    import gainpred.train as train
    monkeypatch.setattr(train, "load_model", lambda path, device: torch.nn.Identity())
    monkeypatch.setattr(train, "evaluate", lambda *a: {"all": 1.0})
    score("repro", runs, data, "cpu", ["si-x"])
    first = (runs / "repro" / "attempt_1" / "reproduction.jsonl").read_text().splitlines()
    assert len(first) == 4  # 3 checkpoints + done
    score("repro", runs, data, "cpu", ["si-x"])
    assert (runs / "repro" / "attempt_2" / "reproduction.jsonl").read_text().splitlines() == ['{"event": "done"}']


def test_table_rows_average_per_length_and_need_three_runs():
    import pytest

    from gainpred.writeup import reproduction_rows

    def ev(run, k, acc):
        return {"run": run, "round": k, "reproduction": {str(d): acc(d) for d in range(3, 16)}}

    events = [ev(f"si-nocarry-seed{s}-authors", k, lambda d: 1.0 if d < 15 else 0.5) for s in (2, 3, 4)
              for k in range(5)]
    events += [ev(f"si-noselect-seed{s}-authors", k, lambda d: 0.0) for s in (2, 3, 4) for k in range(1, 5)]
    events = [e for e in events if not (e["round"] == 0 and "noselect" in e["run"])]
    rows = reproduction_rows(events)
    assert len(rows) == 9
    assert rows[1] == ["No-carry rule, round 1", "93.8 (93.8–93.8)", "50.0 (50.0–50.0)"]  # (7 + 0.5) / 8
    with pytest.raises(AssertionError):
        reproduction_rows(events[1:])
