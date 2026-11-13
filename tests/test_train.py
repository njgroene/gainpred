"""Tests for the training loop, using a tiny model on CPU so they run in seconds.

These check the *wiring* (batches, logging, checkpoints, evaluation), not that
the real model learns addition -- that is what the quick check and the
starting-training run on Modal are for.
"""

import json
import random

import pytest
import torch

from gainpred.config import ModelConfig
from gainpred.data import draw_easy, record
from gainpred.formats import Addition, solve
from gainpred.model import Transformer
from gainpred.tokenizer import PAD_ID, VOCAB_SIZE, make_training_batch
from gainpred.train import EvalSet, TrainConfig, batches, by_length, evaluate, load_model, lr_factor, select, train

CPU = torch.device("cpu")


def tiny_model(seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(ModelConfig(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, n_layers=2, d_model=64, n_heads=4, d_ff=256))


def examples(n: int, seed: int = 0) -> list[tuple[Addition, int]]:
    rng = random.Random(seed)
    problems = [draw_easy(rng, rng.choice([3, 4])) for _ in range(n)]
    return [(p, solve(p)) for p in problems]


def test_batches_cover_every_example_once_per_epoch():
    """One epoch = every example exactly once, in a shuffled order."""
    data = examples(40)
    tokens, mask = make_training_batch(data)
    stream = batches(tokens, mask, batch_size=10, generator=torch.Generator().manual_seed(0))
    seen = [tuple(row[row != PAD_ID].tolist()) for _ in range(4) for row in next(stream)[0]]
    expected = [tuple(row[row != PAD_ID].tolist()) for row in tokens]
    assert sorted(seen) == sorted(expected)
    assert seen != expected  # shuffled


def test_batches_are_trimmed_to_their_longest_row():
    data = [(Addition(100, 100), 200)] * 4  # every row is short
    tokens, mask = make_training_batch(data + [(Addition(1234567, 1234567), 2469134)])
    stream = batches(tokens[:4], mask[:4], batch_size=4, generator=torch.Generator().manual_seed(0))
    batch_tokens, _ = next(stream)
    assert not (batch_tokens == PAD_ID).all(dim=0).any()  # no all-padding columns


def test_select_takes_first_n_per_group():
    records = [{"digits": d, "i": i} for d in (3, 4) for i in range(5)]
    chosen = select(records, by_length, per_group=2)
    assert [(r["digits"], r["i"]) for r in chosen] == [(3, 0), (3, 1), (4, 0), (4, 1)]
    assert select(records, by_length, None) == records


def test_evaluate_reports_per_group_and_overall():
    model = tiny_model()
    records = [record(p) for p, _ in examples(20)]
    accuracy = evaluate(model, records, by_length, batch_size=8, device=CPU)
    assert set(accuracy) == {"3", "4", "all"}
    assert all(0.0 <= v <= 1.0 for v in accuracy.values())


def test_training_lowers_the_loss_and_writes_logs_and_checkpoints(tmp_path):
    data = examples(64)
    eval_records = [record(p) for p, _ in data[:16]]
    cfg = TrainConfig(steps=60, batch_size=16, log_every=20, eval_every=60, checkpoint_every=30)
    model = train(tiny_model(), data, cfg, tmp_path / "run", {"easy": EvalSet(eval_records, by_length)}, CPU)

    events = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    losses = [e["loss"] for e in events if e["event"] == "train"]
    assert len(losses) == 3 and losses[-1] < losses[0]
    assert [e["set"] for e in events if e["event"] == "eval"] == ["easy"]
    assert events[-1]["event"] == "done" and events[-1]["seconds"]["train"] > 0

    checkpoints = sorted((tmp_path / "run" / "checkpoints").iterdir())
    assert [c.name for c in checkpoints] == ["step_000030.pt", "step_000060.pt"]
    # A reloaded checkpoint is the same model.
    reloaded = load_model(checkpoints[-1])
    for (name, a), (_, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        torch.testing.assert_close(a.cpu(), b, msg=name)


def test_same_seed_same_training(tmp_path):
    """P9: a run is reproducible from its seed (on CPU, where it is exact)."""
    data = examples(32)
    cfg = TrainConfig(steps=10, batch_size=8, log_every=10, eval_every=10, checkpoint_every=10)
    a = train(tiny_model(), data, cfg, tmp_path / "a", {}, CPU)
    b = train(tiny_model(), data, cfg, tmp_path / "b", {}, CPU)
    for x, y in zip(a.parameters(), b.parameters()):
        assert torch.equal(x, y)


def test_refuses_to_reuse_a_run_directory(tmp_path):
    cfg = TrainConfig(steps=1, batch_size=8, log_every=1, eval_every=1, checkpoint_every=1)
    train(tiny_model(), examples(8), cfg, tmp_path / "run", {}, CPU)
    with pytest.raises(FileExistsError):
        train(tiny_model(), examples(8), cfg, tmp_path / "run", {}, CPU)


# ----------------------------------------------------------------------------
# Stability: learning-rate schedule and gradient clipping (P10)
# ----------------------------------------------------------------------------


def test_lr_schedule_warmup_stable_decay():
    """10,000 steps: rise over 1,000, hold until 8,000, cosine down to 1% at the end."""
    cfg = TrainConfig()
    factors = [lr_factor(step, cfg) for step in range(cfg.steps)]
    assert factors[0] == 0.0
    assert factors[500] == pytest.approx(0.5)
    assert factors[1000] == factors[5000] == factors[7999] == 1.0
    assert factors[9000] == pytest.approx(0.505)  # halfway down the cosine
    assert factors[-1] == pytest.approx(0.01, abs=1e-5)
    assert all(a <= b for a, b in zip(factors[:1000], factors[1:1001]))  # warm-up rises
    assert all(a >= b for a, b in zip(factors[8000:], factors[8001:]))  # decay falls


def test_lr_schedule_without_warmup_or_decay():
    cfg = TrainConfig(steps=100, warmup_fraction=0.0, decay_fraction=0.0)
    assert {lr_factor(step, cfg) for step in range(100)} == {1.0}


def test_schedule_cannot_exceed_the_run():
    with pytest.raises(ValueError):
        TrainConfig(warmup_fraction=0.5, decay_fraction=0.6)


def test_log_shows_spikes_gradients_and_learning_rate(tmp_path):
    cfg = TrainConfig(steps=40, batch_size=8, log_every=10, eval_every=40, checkpoint_every=40)
    train(tiny_model(), examples(32), cfg, tmp_path / "run", {}, CPU)
    entries = [json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()]
    logged = [e for e in entries if e["event"] == "train"]
    for e in logged:
        assert e["loss_max"] >= e["loss"] > 0
        assert e["grad_norm_max"] >= e["grad_norm"] > 0
    lrs = [e["lr"] for e in logged]
    peak = tiny_model().cfg.lr
    assert lrs[-1] == pytest.approx(peak * 0.01)  # decayed to 1% by the end
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["device"] == "CPU" and config["train"]["grad_clip"] == 1.0


def test_clipping_limits_how_far_weights_move(tmp_path):
    """With a tiny clip, the same batches move the weights far less."""

    def distance_moved(grad_clip: float, name: str) -> float:
        start = tiny_model()
        before = [p.detach().clone() for p in start.parameters()]
        cfg = TrainConfig(steps=5, batch_size=8, log_every=5, eval_every=5, checkpoint_every=5,
                          warmup_fraction=0.0, decay_fraction=0.0, grad_clip=grad_clip)
        trained = train(start, examples(16), cfg, tmp_path / name, {}, CPU)
        return sum((p - b).norm().item() for p, b in zip(trained.parameters(), before))

    # AdamW normalises step sizes, so even tiny gradients move weights -- but
    # clipping to near zero makes the direction noisy and the moves smaller.
    assert distance_moved(1e-9, "tiny") <= distance_moved(1.0, "normal")
