"""The ladder (amendment 29): cuts, test set, the check, the arm rules and a CPU dry run.

The dry runs use a tiny model on the CPU for training and scripted "models" for every answer
(exact, or wrong in a chosen way), so each branch of the round happens for a known reason.
The check sample is the real 2,000 sums: a scripted solver is plain arithmetic, so it is cheap.
"""
import gzip
import json
import random

import pytest
import torch

from gainpred import ladder as ld
from gainpred.config import ModelConfig
from gainpred.cuts import piece_problems
from gainpred.formats import Addition, Unreadable, solve
from gainpred.teachers import PerfectCalculator, combine


# ---------------------------------------------------------------- A. the cut table

def test_cut_table_statements_of_the_read_through():
    pieces = {n: [w for cut in ld.CUTS[n] for w in cut] for n in ld.LENGTHS}
    assert list(ld.CUTS) == list(range(8, 40))
    assert min(min(p) for p in pieces.values()) >= 3
    for n in ld.LENGTHS:
        a, b = (n, n + 1) if n % 2 == 0 else (n - 1, n)
        assert ld.round_lengths((a - 6) // 2) == (a, b)
        assert max(pieces[n]) < min(a, b)
    assert min(n for n in ld.LENGTHS if max(pieces[n]) >= 8) == 13
    assert min(n for n in ld.LENGTHS if min(pieces[n]) >= 8) == 18
    assert max(max(p) for p in pieces.values()) == 21
    shown = {n: " ".join(f"{a}+{b}" for a, b in ld.CUTS[n]) for n in (8, 12, 13, 15, 18, 23, 39)}
    assert shown == {8: "3+5 4+4 5+3", 12: "5+7 6+6 7+5", 13: "5+8 6+7 7+6", 15: "6+9 7+8 8+7",
                     18: "8+10 9+9 10+8", 23: "10+13 11+12 12+11", 39: "18+21 19+20 20+19"}


def test_rounds_and_stages():
    assert [ld.round_lengths(r) for r in (1, 4, 5, 16)] == [(8, 9), (14, 15), (16, 17), (38, 39)]
    assert [ld.stage_of(r) for r in (1, 4, 5, 8, 9, 16)] == [1, 1, 2, 2, 3, 4]


@pytest.mark.parametrize("n", [8, 13, 24, 39])
def test_no_piece_starts_with_zero(n):
    rng = random.Random(n)
    for _ in range(300):
        p = ld.draw(rng, n)
        for cut in ld.CUTS[n]:
            for piece, width in zip(piece_problems(p, cut), cut):
                assert len(str(piece.a)) == len(str(piece.b)) == width


def test_streams_have_no_arm_and_differ_by_seed():
    assert ld.stream_key(5, 3, 12, "check") == "ladder | s5 | 3 | 12 | check"
    assert ld.stream(5, 1, 8, "check").random() == ld.stream(5, 1, 8, "check").random()
    assert ld.stream(5, 1, 8, "check").random() != ld.stream(6, 1, 8, "check").random()


# ---------------------------------------------------------------- B. reference values with exact pieces

@pytest.mark.parametrize("n", [8, 39])
def test_reference_values_with_exact_piece_answers(n):
    problems = ld.draw_fresh(random.Random(n), n, 2_000, set())
    solver = PerfectCalculator()
    report, pool = ld.check_report(3, problems, ld.answer_cuts(problems, solver), solver(problems))
    assert 0.35 <= report["cut1_acc"] <= 0.42
    assert 0.22 <= report["confirmed_share"] <= 0.27
    assert 0.31 <= report["agreed_share"] <= 0.38
    assert report["agreed_acc"] == 1.0 and report["m_agreed"] == len(pool)
    assert report["direct_acc"] == 1.0 and report["novelty_agreed"] == 0.0
    assert report["arm4_verdict"] == "skip_length"  # M is about 690 and nothing is new


def test_arm4_rule_and_its_thresholds():
    assert ld.arm4_builds(99, 0.0, 2_000) and not ld.arm4_builds(100, 0.0, 2_000)
    assert ld.arm4_builds(500, 0.02, 2_000) and not ld.arm4_builds(500, 0.0199, 2_000)
    assert ld.arm4_builds(0, None, 2_000)


def test_expand_to_exactly_the_target():
    pool = list(range(683))
    rows = ld.expand(pool, 10_000, random.Random(0))
    assert len(rows) == 10_000 and all(rows.count(x) in (14, 15) for x in pool)
    assert rows == ld.expand(pool, 10_000, random.Random(0))


def test_update_count_from_unrepeated_examples():
    from gainpred.floors import updates_for
    assert updates_for(45_000) == 3_000 and updates_for(125_000) == 3_500 and updates_for(345_000) == 9_500


# ---------------------------------------------------------------- scripted solvers

class Exact(PerfectCalculator):
    def __init__(self):
        self.asked = []

    def __call__(self, problems, temperature=0.0):
        self.asked += problems
        return super().__call__(problems)


class DirectOff(Exact):
    """Exact pieces; whole sums of the listed lengths answered one too high."""

    def __init__(self, lengths):
        super().__init__()
        self.lengths = set(lengths)

    def __call__(self, problems, temperature=0.0):
        self.asked += problems
        return [solve(p) + (p.n_digits in self.lengths) for p in problems]


class PiecesUnreadable(Exact):
    """Every sum under 8 digits (so every piece) unreadable: no cut gives a readable answer.
    (Pieces one too high would not do: the cuts then agree on about 13% of sums.)"""

    def __call__(self, problems, temperature=0.0):
        self.asked += problems
        return [Unreadable("x") if p.n_digits < 8 else solve(p) for p in problems]


def unreadable_rule(p: Addition) -> bool:
    """Whole sums (8+ digits) unreadable when the first number is even, pieces when it is a
    multiple of 11 (rare enough that the round-1 setup stop does not fire)."""
    return p.a % 2 == 0 if p.n_digits >= 8 else p.a % 11 == 0


class UnreadableSome(Exact):
    def __call__(self, problems, temperature=0.0):
        self.asked += problems
        return [Unreadable("x") if unreadable_rule(p) else solve(p) for p in problems]


# ---------------------------------------------------------------- the dry-run fixture

SMALL = dict(kept_per_length=30, replay_per_length=10, batch_size=8, passes=1, min_steps=4, eval_per_length=5)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from gainpred import floors as fl
    from gainpred.data import write_all
    from gainpred.model import Transformer
    from gainpred.tokenizer import PAD_ID, VOCAB_SIZE
    from gainpred.train import TrainConfig, save_checkpoint

    root = tmp_path_factory.mktemp("ladder")
    write_all(root / "data" / "addition")
    fl.write_floor_data(root / "data" / "floors")
    manifest = ld.write_test_set(root / "data" / "addition", root / "data" / "floors", root / "data" / "ladder")
    torch.manual_seed(0)
    model = Transformer(ModelConfig(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, n_layers=1, d_model=32, n_heads=2, d_ff=64))
    seed = root / "seed" / "step_000001.pt"
    save_checkpoint(seed, model, model.configure_optimizer(), 1, TrainConfig())
    return {"root": root, "seed": seed, "ladder": root / "runs" / "ladder", "manifest": manifest}


def go(world, run, arm, solver_for, **kw):
    cfg = ld.LadderConfig(**{**SMALL, "arm": arm, **kw.pop("cfg", {})})
    root = world["root"]
    reason = ld.run_arm(cfg, run, kw.pop("seed_checkpoint", world["seed"]), world["ladder"], root / "data" / "addition",
                        root / "data" / "floors", root / "data" / "ladder", torch.device("cpu"),
                        make_solver=lambda m, r: solver_for(r), **kw)
    ev = [json.loads(line) for line in (world["ladder"] / run / "rounds.jsonl").read_text().splitlines()]
    return reason, ev


def of(ev, kind, **match):
    return [e for e in ev if e["event"] == kind and all(e.get(k) == v for k, v in match.items())]


def logged_sums(world, run, name):
    rows = gzip.decompress((world["ladder"] / run / "logs" / name).read_bytes()).decode().splitlines()
    return [json.loads(r) for r in rows]


# ---------------------------------------------------------------- the test set (D17)

def test_test_set(world):
    from gainpred.data import load, problems_of
    rows = load(world["root"] / "data" / "ladder" / ld.TEST_FILE)
    assert len(rows) == 32 * 2_000 == world["manifest"]["test_middle_8_39"]["rows"]
    assert world["manifest"]["test_middle_8_39"]["seed"] == 1_006
    for n in ld.LENGTHS:
        group = [r for r in rows if r["digits"] == n]
        assert sum(r["carry_across_cut1"] for r in group) == 1_000 == len(group) - 1_000
    for r in rows[::97]:
        p = Addition(r["a"], r["b"])
        assert r["answer"] == solve(p) and ld.carry_at_cut1(p) == r["carry_across_cut1"]
    problems = problems_of(rows)
    assert len(set(problems)) == len(problems)
    assert not set(problems) & ld.existing_test_problems(world["root"] / "data" / "addition",
                                                         world["root"] / "data" / "floors")
    again = ld.make_test_set(ld.existing_test_problems(world["root"] / "data" / "addition",
                                                       world["root"] / "data" / "floors"))
    assert again == rows


# ---------------------------------------------------------------- C. the dry run

@pytest.mark.slow
def test_arm4_builds_both_lengths(world):
    reason, ev = go(world, "a4-both", 4, lambda r: DirectOff({8, 9}), cfg={"last_round": 1})
    assert reason == "finished"
    assert [e["arm4_verdict"] for e in of(ev, "check")] == ["build", "build"]
    assert all(e["novelty_agreed"] == 1.0 for e in of(ev, "check"))
    assert [e["digits"] for e in of(ev, "build")] == [8, 9] and not of(ev, "skip_length")
    done = of(ev, "round_done")[0]
    assert done["built"] == [8, 9] and not done["round_skipped"] and done["n_raw_examples"] == done["n_effective_examples"]


@pytest.mark.slow
def test_arm4_builds_one_length_and_expands_the_skipped_pool(world):
    reason, ev = go(world, "a4-one", 4, lambda r: DirectOff({9}), cfg={"last_round": 1})
    assert reason == "finished"
    c8, c9 = of(ev, "check")
    assert c8["arm4_verdict"] == "skip_length" and c9["arm4_verdict"] == "build"
    assert [e["digits"] for e in of(ev, "skip_length")] == [8] and [e["digits"] for e in of(ev, "build")] == [9]
    pool, source = ld.load_pool(world["ladder"] / "a4-one", 8)
    assert source == "check_pool" and len(pool) == c8["m_agreed"] >= 100
    assert all(a == solve(p) for p, a in pool)
    done = of(ev, "round_done")[0]
    assert done["composition"]["8"] == 10_000 and done["composition"]["9"] == 30
    assert done["n_raw_examples"] == c8["m_agreed"] + 30 + done["composition"]["replay"]
    assert done["n_effective_examples"] == 10_000 + 30 + done["composition"]["replay"]
    from gainpred.floors import updates_for
    assert done["num_updates"] == updates_for(done["n_raw_examples"], 1, 8, 4)
    assert done["num_updates"] != updates_for(done["n_effective_examples"], 1, 8, 4)


@pytest.mark.slow
def test_arm4_skips_a_whole_round_then_rehearses_both_pools(world):
    solvers = {1: Exact(), 2: DirectOff({10, 11})}
    reason, ev = go(world, "a4-skip", 4, lambda r: solvers[r], cfg={"last_round": 2})
    assert reason == "finished"
    d1, d2 = of(ev, "round_done")
    assert d1["round_skipped"] and d1["num_updates"] == 0 and d1["built"] == [] and d1["sec_train"] == 0
    assert not (world["ladder"] / "a4-skip" / "round_1").exists()  # no training, no new weights
    tests = {e["round"]: e["acc"] for e in of(ev, "test")}
    assert set(tests) == {0, 1, 2} and set(tests[1]) == {str(d) for d in range(3, 10)}
    assert all(tests[1][str(d)] == tests[0][str(d)] for d in range(3, 10))  # the same model, tested again
    index = json.loads((world["ladder"] / "a4-skip" / "kept" / "index.json").read_text())
    assert index["8"]["source"] == index["9"]["source"] == "check_pool"
    assert not d2["round_skipped"] and d2["composition"]["8"] == d2["composition"]["9"] == 10_000
    assert d2["n_raw_examples"] == index["8"]["rows"] + index["9"]["rows"] + 60 + d2["composition"]["replay"]


@pytest.mark.slow
def test_arm4_builds_because_too_few_agree_then_zero_kept_stops_cleanly(world):
    solvers = {1: Exact(), 2: PiecesUnreadable()}
    reason, ev = go(world, "a4-none", 4, lambda r: solvers[r], cfg={"last_round": 3})
    assert reason == "shortfall_stop"
    checks = of(ev, "check", round=2)
    assert all(e["m_agreed"] == 0 < 100 and e["novelty_agreed"] is None and e["arm4_verdict"] == "build" for e in checks)
    b = of(ev, "build", round=2)[0]
    assert b["answers_kept"] == 0 and b["sums_drawn"] == 20 * 30 and b["shortfall_flag"]
    assert of(ev, "call_done")[0]["reason"] == "shortfall_stop" and not of(ev, "round_done", round=2)
    assert not of(ev, "check", round=3)


@pytest.mark.slow
def test_setup_stop_ends_before_building(world):
    reason, ev = go(world, "a1-setup", 1, lambda r: PiecesUnreadable())
    assert reason == "setup_stop"
    assert len(of(ev, "check")) == 2 and not of(ev, "build") and not of(ev, "round_done")
    assert not (world["ladder"] / "a1-setup" / "kept").exists()


@pytest.mark.slow
def test_budget_stop_prevents_the_next_round(world):
    now = [0.0]
    reason, ev = go(world, "a3-budget", 3, lambda r: Exact(), cfg={"last_round": 3}, budget_seconds=1_000,
                    clock=lambda: now[0], after_round=lambda: now.__setitem__(0, 2_000.0))
    assert reason == "budget_stop"
    assert [e["round"] for e in of(ev, "round_done")] == [1] and not of(ev, "check", round=2)
    assert of(ev, "call_done")[0]["before_round"] == 2


@pytest.mark.slow
def test_stage_rule_stops_an_arm_that_did_not_learn_its_stage(world):
    reason, ev = go(world, "a1-stage", 1, lambda r: Exact(), cfg={"last_round": 6})
    assert reason == "stage_not_learned"
    assert [e["round"] for e in of(ev, "round_done")] == [1, 2, 3, 4]
    assert of(ev, "stage")[0] == {"event": "stage", "stage": 1, "learned": False}
    assert not of(ev, "check", round=5)


@pytest.mark.slow
def test_arms_1_and_2_keep_exactly_what_their_rules_say_and_share_check_sums(world):
    _, ev1 = go(world, "a1-keep", 1, lambda r: UnreadableSome(), cfg={"last_round": 1})
    _, ev2 = go(world, "a2-keep", 2, lambda r: UnreadableSome(), cfg={"last_round": 1})
    for d in (8, 9):
        drawn1 = logged_sums(world, "a1-keep", f"round1-{d}-build.jsonl.gz")
        kept1 = dict(((r["a"], r["b"]), r["kept"]) for r in drawn1 if r["kept"] is not None)
        assert kept1 == {(r["a"], r["b"]): str(int(r["a"]) + int(r["b"])) for r in drawn1 if int(r["a"]) % 2}
        assert len(kept1) == 30
        stored, _ = ld.load_pool(world["ladder"] / "a1-keep", d)
        assert {(str(p.a), str(p.b)): str(a) for p, a in stored} == kept1
        drawn2 = logged_sums(world, "a2-keep", f"round1-{d}-build.jsonl.gz")
        expected = {}
        for r in drawn2:
            p = Addition(int(r["a"]), int(r["b"]))
            pieces = piece_problems(p, ld.CUTS[d][0])
            answers = [Unreadable("x") if unreadable_rule(q) else solve(q) for q in pieces]
            built = combine(answers, ld.CUTS[d][0])
            if not isinstance(built, Unreadable):
                expected[(r["a"], r["b"])] = str(built)
        kept2 = {(r["a"], r["b"]): r["kept"] for r in drawn2 if r["kept"] is not None}
        assert kept2 == expected and len(kept2) == 30
    # D9: the same seed draws the same check sums in every arm
    for d in (8, 9):
        s1 = [(r["a"], r["b"]) for r in logged_sums(world, "a1-keep", f"round1-{d}-check.jsonl.gz")]
        s2 = [(r["a"], r["b"]) for r in logged_sums(world, "a2-keep", f"round1-{d}-check.jsonl.gz")]
        assert s1 == s2 and len(s1) == 2_000
    c1, c2 = of(ev1, "check"), of(ev2, "check")
    for a, b in zip(c1, c2):
        for k in ("direct_acc", "cut1_acc", "confirmed_share", "agreed_share", "novelty_agreed", "arm4_verdict"):
            assert a[k] == b[k]
    assert all(e["novelty_arm"] == 0.0 for e in c1)


@pytest.mark.slow
def test_every_logged_field_of_section_6(world):
    _, ev = go(world, "a3-fields", 3, lambda r: Exact(), cfg={"last_round": 1})
    check_fields = {"direct_acc", "cut1_acc", "confirmed_share", "agreed_share", "agreed_acc", "novelty_arm",
                    "novelty_agreed", "arm4_verdict"}
    assert all(check_fields <= set(e) for e in of(ev, "check"))
    assert all({"sums_drawn", "answers_kept", "shortfall_flag", "kept_acc", "kept_carry_share"} <= set(e)
               for e in of(ev, "build"))
    done = of(ev, "round_done")[0]
    assert {"n_raw_examples", "n_effective_examples", "num_updates", "round_skipped"} <= set(done)
    assert all(done[k] > 0 for k in ("sec_check", "sec_build", "sec_train", "sec_test", "sec_total"))
    acc = of(ev, "test", round=1)[0]["acc"]
    assert set(acc) == {str(d) for d in range(3, 10)}
    assert {"acc_total", "acc_carry", "acc_no_carry"} <= set(acc["8"]) and set(acc["3"]) == {"acc_total"}
    assert all(e["kept_acc"] == 1.0 for e in of(ev, "build"))


@pytest.mark.slow
def test_relaunch_continues_from_the_last_completed_round(world):
    reason, _ = go(world, "a3-again", 3, lambda r: Exact(), cfg={"last_round": 1})
    assert reason == "finished"
    reason, ev = go(world, "a3-again", 3, lambda r: Exact(), cfg={"last_round": 2}, first_round=2, tag="relaunch",
                    prior_seconds=5.0)
    assert reason == "finished"
    config = json.loads((world["ladder"] / "a3-again" / "config-rounds2-relaunch.json").read_text())
    assert config["start_checkpoint"].endswith("round_1/checkpoints/step_000100.pt")
    done = of(ev, "round_done", round=2)[0]
    assert done["composition"]["8"] == done["composition"]["9"] == 30
