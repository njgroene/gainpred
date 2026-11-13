"""Tests for gainpred.report: collecting run logs and building the summary tables."""

import csv
import json

from gainpred.report import collect, load_all, write_evals_csv, write_summary
from gainpred.train import TrainConfig


def fake_run(runs_dir, name, finished=True):
    run_dir = runs_dir / name
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "step_000010.pt").write_bytes(b"big")
    cfg = {"train": TrainConfig(steps=20, warmup_fraction=0.0, decay_fraction=0.5).__dict__, "device": "CPU"}
    (run_dir / "config.json").write_text(json.dumps(cfg))
    events = [
        {"event": "train", "step": 10, "loss": 1.0, "loss_max": 2.0, "grad_norm": 3.0, "grad_norm_max": 4.0, "lr": 1e-3},
        {"event": "eval", "set": "easy", "step": 10, "accuracy": {"3": 0.5, "all": 0.5}},
    ]
    if finished:
        events += [
            {"event": "train", "step": 20, "loss": 0.1, "loss_max": 0.2, "grad_norm": 9.0, "grad_norm_max": 9.5, "lr": 1e-5},
            {"event": "eval", "set": "easy", "step": 20, "accuracy": {"3": 1.0, "all": 1.0}},
            {"event": "done", "step": 20, "seconds": {"train": 60.0, "eval": 6.0}},
        ]
    (run_dir / "log.jsonl").write_text("\n".join(json.dumps(e) for e in events))


def test_collect_copies_logs_but_not_checkpoints(tmp_path):
    fake_run(tmp_path / "runs", "a")
    (tmp_path / "runs" / "empty").mkdir()  # no log: skipped
    assert collect(tmp_path / "runs", tmp_path / "results") == ["a"]
    kept = sorted(p.name for p in (tmp_path / "results" / "runs" / "a").iterdir())
    assert kept == ["config.json", "log.jsonl"]


def test_summary_and_csv(tmp_path):
    fake_run(tmp_path / "runs", "done")
    fake_run(tmp_path / "runs", "stopped", finished=False)
    results = tmp_path / "results"
    collect(tmp_path / "runs", results)
    runs = load_all(results)
    write_summary(runs, results)
    write_evals_csv(runs, results)

    summary = (results / "summary.md").read_text()
    assert "final **100.0%** at step 20" in summary
    assert "**stopped early** at step 10" in summary
    assert "| decay | 10–20 | 0.2 | 9.5 | 1.00e-01 |" in summary

    rows = list(csv.DictReader((results / "evals.csv").open()))
    assert len(rows) == 6  # done: 2 evals x 2 groups; stopped: 1 eval x 2 groups
    assert rows[0] == {"run": "done", "step": "10", "set": "easy", "group": "3", "accuracy": "0.5"}
