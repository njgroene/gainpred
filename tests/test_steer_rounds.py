"""Steered rounds (amendment 25): the decision rule and the round loop, with scripted solvers.

The solvers add piece sums perfectly (pieces are at most 7 digits), but give a wrong *direct*
answer on problems of chosen lengths (8-15 digits). There the agreement rule's kept answers
differ from what the "model" says, so novelty is 100% and the length is taught; elsewhere
novelty is 0% and the length is skipped.
"""
import json

import pytest
import torch

from gainpred.config import ModelConfig
from gainpred.data import write_all
from gainpred.formats import solve
from gainpred.model import Transformer
from gainpred.rounds import RoundConfig, self_improve, steer_decision
from gainpred.teachers import PerfectCalculator
from gainpred.tokenizer import PAD_ID, VOCAB_SIZE
from gainpred.train import TrainConfig, save_checkpoint


# ---------------------------------------------------------------- the decision rule

@pytest.mark.parametrize("mode", ["novelty", "novelty_rehearse"])
def test_novelty_teaches_at_or_above_the_level(mode):
    assert steer_decision(mode, 0.03, 0.5, 0.1, False, 0.03) == ("teach", None)
    assert steer_decision(mode, 0.0299, 0.5, 1.0, True, 0.03) == ("skip", None)
    assert steer_decision(mode, None, 0.0, None, True, 0.03) == ("skip", None)


def test_frozen_teaches_only_when_the_verdict_is_pays_off():
    assert steer_decision("frozen", 0.9, 0.5, 0.9, True) == ("teach", "pays off")
    assert steer_decision("frozen", 0.9, 0.5, 0.9, None) == ("teach", "pays off")  # no evaluation yet
    assert steer_decision("frozen", 0.9, 0.5, 0.9, False) == ("skip", "does harm")  # check 0 first
    assert steer_decision("frozen", 0.49, 0.5, 0.9, True) == ("skip", "does nothing")
    assert steer_decision("frozen", 0.9, 0.04, 0.9, True) == ("skip", "does nothing")  # under 5% kept
    assert steer_decision("frozen", None, 0.0, None, True) == ("skip", "does nothing")
    assert steer_decision("frozen", 0.9, 0.5, 0.37, True) == ("skip", "does harm")
    assert steer_decision("frozen", 0.9, 0.5, None, True) == ("teach", "pays off")


def test_frozen_matches_the_p9b_rule_on_every_scored_case():
    from gainpred.p9b import LEVEL_2B
    from gainpred.writeup import load_cells
    for c in load_cells():
        if c.verdict:
            assert LEVEL_2B == 0.38
            assert steer_decision("frozen", c.check_1, c.kept_share, c.check_2b, c.check_0)[1] == c.verdict


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        steer_decision("always", 1.0, 1.0, 1.0, True)


# ---------------------------------------------------------------- the round loop (tiny model, CPU)

class WrongDirect(PerfectCalculator):
    """Right on pieces; one too many on direct problems of the given lengths."""

    def __init__(self, lengths):
        self.lengths = set(lengths)

    def __call__(self, problems, temperature=0.0):
        return [solve(p) + (p.n_digits in self.lengths) for p in problems]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("steer")
    write_all(root / "data")
    torch.manual_seed(0)
    model = Transformer(ModelConfig(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, n_layers=1, d_model=32, n_heads=2, d_ff=64))
    save_checkpoint(root / "runs" / "start" / "checkpoints" / "step_000001.pt", model, model.configure_optimizer(), 1,
                    TrainConfig())
    return root, model


CFG = dict(teacher="agreement", kept_per_length=20, replay_per_length=20, check_sample=10, steps=4, decay_steps=2,
           batch_size=8, eval_per_length=5, solver_batch_size=64)


def run(setup, name, rounds, wrong_by_round, steer="novelty"):
    root, _ = setup
    cfg = RoundConfig(rounds=rounds, steer=steer, **CFG)
    model = self_improve(cfg, name, "start", data_dir=root / "data", device=torch.device("cpu"), runs_dir=root / "runs",
                         make_solver=lambda model, r: WrongDirect(wrong_by_round(r)))
    return model, [json.loads(line) for line in (root / "runs" / name / "rounds.jsonl").read_text().splitlines()]


@pytest.mark.slow
def test_a_skipped_length_is_not_built_or_trained(setup):
    root = setup[0]
    _, ev = run(setup, "skip8", 1, lambda r: {9} if r == 1 else set())
    steer = {e["digits"]: e["decision"] for e in ev if e["event"] == "steer" and e["round"] == 1}
    assert steer == {8: "skip", 9: "teach"}
    assert [(e["round"], e["digits"]) for e in ev if e["event"] == "build"] == [(1, 9)]
    replay = next(e for e in ev if e["event"] == "replay")
    examples = json.loads((root / "runs" / "skip8" / "round_1" / "config.json").read_text())["examples"]
    assert examples == 20 + replay["readable"]  # only the taught length's 20 answers, plus replay
    # checks come before the build in a steered round
    kinds = [e["event"] for e in ev]
    assert kinds.index("checks") < kinds.index("build")


@pytest.mark.slow
def test_a_round_with_nothing_taught_leaves_the_weights_unchanged(setup):
    root, start = setup
    model, ev = run(setup, "nothing", 1, lambda r: set())
    assert [e for e in ev if e["event"] == "round_skipped"] == [{"event": "round_skipped", "round": 1}]
    assert not [e for e in ev if e["event"] in ("build", "replay")]
    assert {e["round"] for e in ev if e["event"] == "eval"} == {0, 1}  # still evaluated
    assert not (root / "runs" / "nothing" / "round_1").exists()
    for a, b in zip(model.state_dict().values(), start.state_dict().values()):
        assert torch.equal(a.cpu(), b.cpu())
    assert next(e for e in ev if e["event"] == "closing_done")["repair_rounds"] == 0


@pytest.mark.slow
def test_closing_check_triggers_repair_rounds_and_stops_after_two(setup):
    _, ev = run(setup, "repair", 1, lambda r: {15})  # the "model" never learns 15 digits
    assert [(e["round"], e["digits"]) for e in ev if e["event"] == "build"] == [(2, 15), (3, 15)]
    assert {e["round"] for e in ev if e["event"] == "closing_checks"} == {2, 3, 4}
    assert {e["digits"] for e in ev if e["event"] == "closing_checks"} == set(range(8, 16))
    done = next(e for e in ev if e["event"] == "closing_done")
    assert done["repair_rounds"] == 2 and done["would_teach"] == [15]
    assert {e["round"] for e in ev if e["event"] == "eval"} == {0, 1, 2, 3}


@pytest.mark.slow
def test_rehearsal_adds_stored_answers_of_skipped_lengths_only(setup):
    root = setup[0]
    _, ev = run(setup, "rehearse", 2, lambda r: {1: {9}, 2: {8}}.get(r, set()), steer="novelty_rehearse")
    assert [(e["round"], e["digits"]) for e in ev if e["event"] == "build"] == [(1, 9), (2, 8)]
    rehearse = [e for e in ev if e["event"] == "rehearse"]
    assert rehearse == [{"event": "rehearse", "round": 2, "digits": 9, "examples": 20, "from_round": 1}]
    replay = [e for e in ev if e["event"] == "replay"][1]
    examples = json.loads((root / "runs" / "rehearse" / "round_2" / "config.json").read_text())["examples"]
    assert examples == 20 + 20 + replay["readable"]  # built 8, rehearsed 9, replay; nothing for 10-11


@pytest.mark.slow
def test_novelty_mode_does_not_rehearse(setup):
    _, ev = run(setup, "norehearse", 2, lambda r: {1: {9}, 2: {8}}.get(r, set()))
    assert not [e for e in ev if e["event"] == "rehearse"]


# ---------------------------------------------------------------- schedule only (amendment 26, arm g)

@pytest.mark.slow
def test_schedule_builds_the_two_newest_lengths_and_rehearses_the_rest(setup):
    root = setup[0]
    # The solver would make every length look new; the schedule ignores that: no checks at all.
    _, ev = run(setup, "schedule", 2, lambda r: set(range(8, 16)), steer="schedule")
    assert [(e["round"], e["digits"]) for e in ev if e["event"] == "build"] == [(1, 8), (1, 9), (2, 10), (2, 11)]
    assert not [e for e in ev if e["event"] in ("checks", "closing_checks", "closing_done")]
    steer = [(e["round"], e["digits"], e["decision"]) for e in ev if e["event"] == "steer"]
    assert steer == [(1, 8, "build"), (1, 9, "build"), (2, 8, "rehearse"), (2, 9, "rehearse"), (2, 10, "build"),
                     (2, 11, "build")]
    rehearse = [(e["round"], e["digits"], e["examples"], e["from_round"]) for e in ev if e["event"] == "rehearse"]
    assert rehearse == [(2, 8, 20, 1), (2, 9, 20, 1)]
    replay = [e for e in ev if e["event"] == "replay"][1]
    examples = json.loads((root / "runs" / "schedule" / "round_2" / "config.json").read_text())["examples"]
    assert examples == 20 + 20 + 20 + 20 + replay["readable"]
    done = [e for e in ev if e["event"] == "round_done"]
    assert all("checks" not in e["seconds"] or e["seconds"]["checks"] == 0 for e in done)
