"""Tests for longest-run training: evaluation, starting training and one round (P1, P3)."""

import json
import random

import pytest
import torch

from gainpred.config import ModelConfig
from gainpred.formats import Run, RunQuestion, longest_run
from gainpred.model import Transformer
from gainpred.rounds import collect_kept
from gainpred.run_data import RANDOM_JOIN, RUN_ACROSS
from gainpred.run_teachers import PerfectRunSolver, edges_differ
from gainpred.run_train import (
    RunRoundConfig,
    by_question_and_length,
    draw_fresh,
    evaluate_runs,
    rng_for,
    self_improve_runs,
    start_examples,
    starting_training_runs,
)
from gainpred.tokenizer import PAD_ID, VOCAB_SIZE, make_training_batch
from gainpred.train import TrainConfig


def tiny_model() -> Transformer:
    torch.manual_seed(0)
    return Transformer(ModelConfig(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, n_layers=1, d_model=32, n_heads=2, d_ff=64))


def test_training_batches_take_run_examples():
    tokens, mask = make_training_batch([(RunQuestion("runlen", "887476"), Run("8", 2))])
    assert tokens.shape == mask.shape and int(mask.sum()) == len("8|2") + 1


def test_start_examples_have_true_answers_and_can_filter_questions():
    rows = [{"question": "runlen", "string": "887476"}, {"question": "rightrun", "string": "887476"}]
    assert start_examples(rows) == [(RunQuestion("runlen", "887476"), Run("8", 2)),
                                    (RunQuestion("rightrun", "887476"), Run("6", 1))]
    assert len(start_examples(rows, ("runlen",))) == 1


def test_evaluate_groups_by_question_length_and_run_length():
    rows = [{"question": "runlen", "string": "887476", "length": 6, "run_length": 2},
            {"question": "leftrun", "string": "1234567", "length": 7, "run_length": 1}]
    model = tiny_model()  # untrained: answers are wrong or unreadable
    accuracy = evaluate_runs(model, rows, by_question_and_length, 10, torch.device("cpu"))
    assert set(accuracy) == {"runlen/6", "leftrun/7", "runlen/run=2", "leftrun/run=1", "all"}


def test_collect_kept_with_the_run_pool():
    exclude: set[str] = set()
    kept, drawn, timed_out = collect_kept(edges_differ, PerfectRunSolver(), random.Random(0), 16, quota=300,
                                          cap=6000, exclude=exclude, draw=draw_fresh)
    assert not timed_out and len(kept) == 300
    assert all(a == longest_run(p.string) and p.string[p.join - 1] != p.string[p.join] for p, a in kept)
    assert 300 / drawn == pytest.approx(0.45, abs=0.08)  # paper test: ~45% of the half-and-half pool
    assert len(exclude) >= drawn


def test_round_streams_are_shared_by_teachers_and_never_hit_addition_seeds():
    a = draw_fresh(rng_for(1, 1, 14, "build"), 14, 30, set())
    assert a == draw_fresh(rng_for(1, 1, 14, "build"), 14, 30, set())
    assert a != draw_fresh(rng_for(2, 1, 14, "build"), 14, 30, set())
    assert {p.kind for p in a} == {RANDOM_JOIN, RUN_ACROSS}


def test_round_schedule():
    t = RunRoundConfig().train_config(1)
    assert (t.steps, t.warmup_fraction, t.decay_fraction) == (3000, 0.0, pytest.approx(1 / 3))
    assert RunRoundConfig().kept_per_length == 2000 and RunRoundConfig().rounds == 1


@pytest.fixture(scope="module")
def small_data(tmp_path_factory):
    """The real test sets, and a small starting-training file (the full one takes seconds to load each time)."""
    from gainpred.run_data import make_test_easy, make_test_study, make_train_start, SEEDS
    from gainpred.data import to_jsonl

    d = tmp_path_factory.mktemp("data")
    exclude: set[str] = set()
    sets = {"test_study": make_test_study(exclude), "test_easy": make_test_easy(exclude),
            "train_start": make_train_start(exclude, per_length=40)}
    for name, records in sets.items():
        (d / f"{name}.jsonl").write_bytes(to_jsonl(records))
    (d / "manifest.json").write_text(json.dumps({n: {"seed": SEEDS[n], "rows": len(r)} for n, r in sets.items()}))
    return d


@pytest.mark.slow
def test_starting_training_and_one_round_run_end_to_end(tmp_path, small_data, monkeypatch):
    """Wiring only: a tiny model, a few steps, CPU, and a perfect solver for the round."""
    runs = tmp_path / "runs"
    cfg = TrainConfig(steps=4, batch_size=8, eval_every=4, checkpoint_every=4, eval_per_length=3,
                      decay_fraction=0.1)
    # Starting training with the real model size would be slow on CPU; swap in the tiny one.
    monkeypatch.setattr("gainpred.run_train.Transformer", lambda cfg_: tiny_model())
    starting_training_runs(cfg, "start", data_dir=small_data, device=torch.device("cpu"), runs_dir=runs)
    log = [json.loads(line) for line in (runs / "start" / "log.jsonl").read_text().splitlines()]
    evals = {e["set"]: e["accuracy"] for e in log if e["event"] == "eval"}
    assert "runlen/6" in evals["easy"] and "rightrun/run=1" in evals["easy"]
    assert {"random_join", "joined_length_8", "left_part_3-4"} <= set(evals["study"])
    assert json.loads((runs / "start" / "config.json").read_text())["task"] == "longest_run/starting-training"

    round_cfg = RunRoundConfig(teacher="join_at_the_cut", kept_per_length=10, check_sample=10, steps=4,
                               decay_steps=2, batch_size=8, eval_per_length=3, solver_batch_size=64)
    self_improve_runs(round_cfg, "si", "start", data_dir=small_data, device=torch.device("cpu"), runs_dir=runs,
                      make_solver=lambda model, round_: PerfectRunSolver())
    events = [json.loads(line) for line in (runs / "si" / "rounds.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "eval" and kinds[-1] == "done" and "round_done" in kinds
    builds = [e for e in events if e["event"] == "build"]
    assert [e["length"] for e in builds] == list(range(12, 21))
    assert all(e["kept"] == 10 and e["kept_right"] == 1.0 and not e["stopped_by_time"] for e in builds)
    checks = [e for e in events if e["event"] == "checks"]
    assert [e["length"] for e in checks] == list(range(12, 21))
    assert all(e["check_1"] is not None and "check_3" in e and e["check_2b"] == 1.0 for e in checks)
    replay = next(e for e in events if e["event"] == "replay")
    assert replay["problems"] == 40 * 5 * 3  # every starting row, true answers
    # Checks come before the round's evaluation (status tripwire 5).
    assert max(i for i, k in enumerate(kinds) if k == "checks") < max(i for i, k in enumerate(kinds) if k == "eval")
    assert (runs / "si" / "round_1" / "checkpoints" / "step_000004.pt").exists()
    with pytest.raises(FileExistsError):
        self_improve_runs(round_cfg, "si", "start", data_dir=small_data, device=torch.device("cpu"), runs_dir=runs)

    # The report collects both runs and summarises them as longest run.
    from gainpred import report

    results = tmp_path / "results"
    assert set(report.collect(runs, results)) == {"start", "si"}
    (start_run,) = report.load_all(results)
    assert "Easy longest-run questions" in "\n".join(report.summarise_run(start_run))
    (rounds_run,) = report.load_rounds(results)
    text = "\n".join(report.summarise_rounds_run(rounds_run))
    assert "join at the cut" in text and "check 3" in text


def test_withheld_draw_never_returns_a_3_or_4_run_across():
    import random

    from gainpred.run_data import RUN_ACROSS
    from gainpred.run_train import draw_withheld

    drawn = draw_withheld(random.Random(0), 16, 2_000, set())
    assert 1_400 < len(drawn) < 1_600  # half run-across, half of those dropped
    assert not any(p.kind == RUN_ACROSS and p.a in (3, 4) for p in drawn)
    assert any(p.kind == RUN_ACROSS and p.a in (1, 2) for p in drawn)
