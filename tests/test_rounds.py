"""Tests for the self-improvement rounds (P3), with a perfect calculator or a tiny model."""

import json
import random
import time

import pytest
import torch

from gainpred.config import ModelConfig
from gainpred.data import write_all
from gainpred.formats import solve
from gainpred.model import Transformer
from gainpred.rounds import RoundConfig, collect_kept, draw_fresh, rng_for, self_improve
from gainpred.teachers import PerfectCalculator, no_carry, agreement
from gainpred.tokenizer import PAD_ID, VOCAB_SIZE
from gainpred.train import TrainConfig, save_checkpoint


def test_collect_kept_meets_the_quota_with_right_answers():
    exclude = set()
    kept, drawn, timed_out = collect_kept(no_carry, PerfectCalculator(), random.Random(0), 8, quota=300, cap=6000,
                                          exclude=exclude)
    assert not timed_out
    assert len(kept) == 300 and all(a == solve(p) for p, a in kept)
    assert 300 / drawn == pytest.approx(0.38, abs=0.08)  # paper test keep rate
    assert len({p for p, _ in kept}) == 300 and len(exclude) >= drawn


def test_collect_kept_stops_at_the_cap_and_reports_a_shortfall():
    kept, drawn, _ = collect_kept(agreement, PerfectCalculator(), random.Random(0), 15, quota=1000, cap=2000,
                                  exclude=set())
    assert drawn == 2000 and len(kept) < 1000  # ~7% of 2,000


def test_collect_kept_stops_at_the_time_limit_and_reports_it():
    kept, drawn, timed_out = collect_kept(agreement, PerfectCalculator(), random.Random(0), 15, quota=10_000,
                                          cap=200_000, exclude=set(), seconds=0.0, max_chunk=1_000)
    assert timed_out and drawn == 0 and kept == []

    class Slow(PerfectCalculator):
        def __call__(self, problems, temperature=0.0):
            time.sleep(0.2)
            return super().__call__(problems, temperature)

    kept, drawn, timed_out = collect_kept(agreement, Slow(), random.Random(0), 15, quota=10_000,
                                          cap=200_000, exclude=set(), seconds=0.1, max_chunk=1_000)
    assert timed_out and drawn == 1_000  # one chunk, then the limit is noticed


def test_draw_fresh_avoids_excluded_problems():
    first = draw_fresh(random.Random(1), 9, 50, set())
    again = draw_fresh(random.Random(1), 9, 50, set(first))
    assert not set(first) & set(again)


def test_same_seed_same_candidates_for_every_teacher():
    a = draw_fresh(rng_for(1, 1, 8, "build"), 8, 20, set())
    b = draw_fresh(rng_for(1, 1, 8, "build"), 8, 20, set())
    assert a == b and a != draw_fresh(rng_for(2, 1, 8, "build"), 8, 20, set())


def test_round_schedule_is_constant_then_decays():
    t = RoundConfig().train_config(1)
    assert (t.steps, t.warmup_fraction, t.decay_fraction) == (3000, 0.0, pytest.approx(1 / 3))


@pytest.mark.slow
def test_one_tiny_round_runs_end_to_end(tmp_path):
    """Wiring only: a tiny untrained model, small quotas, CPU. Takes ~1 minute (writes all datasets)."""
    data_dir = tmp_path / "data"
    write_all(data_dir)
    torch.manual_seed(0)
    model = Transformer(ModelConfig(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, n_layers=1, d_model=32, n_heads=2, d_ff=64))
    runs = tmp_path / "runs"
    save_checkpoint(runs / "start" / "checkpoints" / "step_000001.pt", model, model.configure_optimizer(), 1,
                    TrainConfig())
    cfg = RoundConfig(teacher="no_selection", rounds=1, kept_per_length=20, replay_per_length=20, check_sample=10,
                      steps=4, decay_steps=2, batch_size=8, eval_per_length=5, solver_batch_size=64)
    # A perfect calculator adds the pieces, so there are answers to train on.
    self_improve(cfg, "si", "start", data_dir=data_dir, device=torch.device("cpu"), runs_dir=runs,
                 make_solver=lambda model, round_: PerfectCalculator())
    events = [json.loads(line) for line in (runs / "si" / "rounds.jsonl").read_text().splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "eval" and kinds[-1] == "done"
    assert {e["digits"] for e in events if e["event"] == "build"} == {8, 9}
    assert {e["digits"] for e in events if e["event"] == "checks"} == {8, 9}
    assert {e["round"] for e in events if e["event"] == "eval"} == {0, 1}
    built = [e for e in events if e["event"] == "build"]
    assert all(e["kept"] == 20 for e in built)  # no selection keeps everything
    assert (runs / "si" / "round_1" / "checkpoints" / "step_000004.pt").exists()

    # P9b run f: continue from round 1 of that run with another teacher; round 2 follows.
    cfg_f = RoundConfig(**{**cfg.__dict__, "teacher": "no_carry"})
    self_improve(cfg_f, "f", "si", data_dir=data_dir, device=torch.device("cpu"), runs_dir=runs,
                 make_solver=lambda model, round_: PerfectCalculator(), start_round=1)
    events_f = [json.loads(line) for line in (runs / "f" / "rounds.jsonl").read_text().splitlines()]
    assert {e["round"] for e in events_f if e["event"] == "eval"} == {1, 2}  # before (= round 1) and after
    assert {e["digits"] for e in events_f if e["event"] == "build"} == set(range(8, 12))  # round 2: 8-11
    assert json.loads((runs / "f" / "config.json").read_text())["start_round"] == 1

