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


# ================================================================ amendment 30: the divergence rule
#
# The dry runs above use a tiny random CPU model, which is never 90% right at 3-7 digits, so
# the divergence rule would fire in every one of their rounds. They test amendment 29's rules
# and keep running with the rule switched off (the line below only adds to their configuration;
# their code is unchanged). The tests below switch it on and script the test results with
# ``test_hook``: 100% at 3-7 digits, except where a test says otherwise.

SMALL["diverged_level"] = 0.0
ON = {"diverged_level": 0.90}


def scripted(low=(), short=0.5, long=None):
    """A test hook: acc_total at 3-7 is 1.0, except at 5 digits ``short`` for the (round, kind)
    pairs in ``low``; ``long`` {(round, kind): value} sets every length from 8 up."""
    def hook(round_, kind, acc):
        acc = {d: dict(v) for d, v in acc.items()}
        for d in map(str, range(3, 8)):
            acc[d]["acc_total"] = short if (round_, kind) in low and d == "5" else 1.0
        if long and (round_, kind) in long:
            for d in acc:
                if int(d) >= 8:
                    acc[d]["acc_total"] = long[(round_, kind)]
        return acc
    return hook


def state_digest(model) -> str:
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode() + v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.fixture
def trainings(monkeypatch):
    """Every training run_arm starts: its start weights, examples, updates, batch seed, folder."""
    import gainpred.train as tr
    real, calls = tr.train, []

    def spy(model, examples, cfg, run_dir, eval_sets, device, extra_config=None):
        calls.append({"state": state_digest(model), "examples": list(examples), "steps": cfg.steps,
                      "seed": cfg.seed, "folder": run_dir.name, "warmup": cfg.warmup_fraction,
                      "decay": cfg.decay_fraction})
        return real(model, examples, cfg, run_dir, eval_sets, device, extra_config)

    monkeypatch.setattr(tr, "train", spy)
    return calls


def checkpoint_digest(world, run, folder):
    from gainpred.floors import last_checkpoint, load_long
    return state_digest(load_long(last_checkpoint(world["ladder"] / run / folder)))


@pytest.mark.slow
def test_a_diverged_round_is_redone_once_from_the_same_weights_on_the_same_examples(world, trainings):
    run = "a3-redo"
    reason, ev = go(world, run, 3, lambda r: Exact(), cfg={**ON, "last_round": 2}, test_hook=scripted({(1, "test")}))
    assert reason == "finished"
    assert [t["folder"] for t in trainings] == ["round_1", "round_1-redo", "round_2"]
    first, redo, nxt = trainings
    assert redo["state"] == first["state"] and redo["examples"] == first["examples"] and len(first["examples"]) > 0
    assert redo["steps"] == first["steps"] and (redo["warmup"], redo["decay"]) == (first["warmup"], first["decay"])
    assert first["seed"] == ld.stream_seed(5, 1, "batches") != redo["seed"] == ld.stream_seed(5, 1, "batches-redo")
    assert [(e["round"], e["acc_short"]["5"]) for e in of(ev, "diverged")] == [(1, 0.5)]
    assert [bool(e.get("redo")) for e in of(ev, "test", round=1)] == [False, True]
    done = of(ev, "round_done")
    assert [d["redone"] for d in done] == [True, False] and not of(ev, "diverged", round=2)
    # the next round starts from the redo's weights, never from the first attempt's
    assert nxt["state"] == checkpoint_digest(world, run, "round_1-redo") != checkpoint_digest(world, run, "round_1")
    assert ld._latest_weights(world["ladder"] / run, 2).parent.parent.name == "round_1-redo"


@pytest.mark.slow
def test_the_stage_rule_uses_the_redos_test(world, trainings):
    # the first attempt of round 4 would learn stage 1 (95% at 8-15) but is diverged; the redo is not
    # diverged and is at 50% at 8-15, so stage 1 is not learned
    hook = scripted({(4, "test")}, long={(4, "test"): 0.95, (4, "redo"): 0.5})
    reason, ev = go(world, "a3-stage-redo", 3, lambda r: Exact(), cfg={**ON, "last_round": 5}, test_hook=hook)
    assert reason == "stage_not_learned"
    assert ld.stage_learned(of(ev, "test", round=4)[0]["acc"], 1) and not ld.stage_learned(of(ev, "test", round=4)[1]["acc"], 1)
    assert of(ev, "stage") == [{"event": "stage", "stage": 1, "learned": False}]
    assert [t["folder"] for t in trainings][-2:] == ["round_4", "round_4-redo"] and not of(ev, "check", round=5)


@pytest.mark.slow
def test_diverged_twice_ends_the_arm(world, trainings):
    reason, ev = go(world, "a3-twice", 3, lambda r: Exact(), cfg={**ON, "last_round": 3},
                    test_hook=scripted({(1, "test"), (1, "redo")}))
    assert reason == "diverged_twice"
    assert of(ev, "call_done")[0]["round"] == 1 and len(of(ev, "diverged")) == 1
    assert [t["folder"] for t in trainings] == ["round_1", "round_1-redo"] and not of(ev, "check", round=2)


@pytest.mark.slow
def test_a_round_at_90_percent_is_not_redone(world, trainings):
    reason, ev = go(world, "a3-at90", 3, lambda r: Exact(), cfg={**ON, "last_round": 1},
                    test_hook=scripted({(1, "test")}, short=0.90))
    assert reason == "finished" and not of(ev, "diverged") and len(of(ev, "test", round=1)) == 1
    assert [t["folder"] for t in trainings] == ["round_1"] and not (world["ladder"] / "a3-at90" / "round_1-redo").exists()
    assert not ld.diverged({str(d): {"acc_total": 0.9} for d in range(3, 8)})
    assert ld.diverged({**{str(d): {"acc_total": 1.0} for d in range(3, 8)}, "7": {"acc_total": 0.8999}})


@pytest.mark.slow
def test_a_round_arm4_skipped_is_never_redone(world, trainings):
    solvers = {1: Exact(), 2: DirectOff({10, 11})}
    reason, ev = go(world, "a4-skip-redo", 4, lambda r: solvers[r], cfg={**ON, "last_round": 2},
                    test_hook=scripted({(1, "test")}))
    assert reason == "finished"
    assert of(ev, "round_done", round=1)[0]["round_skipped"] and not of(ev, "diverged")
    assert [t["folder"] for t in trainings] == ["round_2"]


def test_the_redo_folder_is_the_rounds_result_whatever_the_names(tmp_path):
    def make(name, step=10):
        (tmp_path / name / "checkpoints").mkdir(parents=True)
        (tmp_path / name / "checkpoints" / f"step_{step:06d}.pt").write_bytes(b"")

    make("round_3")
    make("round_3-relaunch")
    assert ld._latest_weights(tmp_path, 5).parent.parent.name == "round_3-relaunch"
    make("round_3-redo8")  # a call tagged "redo8": an ordinary relaunch folder (sorts before "relaunch"), not a redo
    assert ld._latest_weights(tmp_path, 5).parent.parent.name == "round_3-relaunch"
    make("round_3-redo")
    make("round_3-zzz")
    assert ld._latest_weights(tmp_path, 5).parent.parent.name == "round_3-redo"
    assert ld._latest_weights(tmp_path, 4).parent.parent.name == "round_3-redo"
    make("round_33")  # never round 3's
    assert ld._latest_weights(tmp_path, 5).parent.parent.name == "round_3-redo"
    (tmp_path / "round_4-redo").mkdir()  # a redo without a checkpoint (stopped early) is not a result
    make("round_4")
    assert ld._latest_weights(tmp_path, 5).parent.parent.name == "round_4"


def original_without_the_rule(world, run, last_round):
    """A run as amendment 29 ran it: rule off, and its last round's test scripted as diverged."""
    reason, _ = go(world, run, 3, lambda r: Exact(), cfg={"last_round": last_round},
                   test_hook=scripted({(last_round, "test")}))
    assert reason == "finished"


@pytest.mark.slow
def test_the_continuation_redoes_the_round_without_check_or_build_then_goes_on(world, trainings):
    run = "a3-cont"
    original_without_the_rule(world, run, 2)
    original = list(trainings)
    assert [t["folder"] for t in original] == ["round_1", "round_2"]
    before = (world["ladder"] / run / "rounds.jsonl").read_text().splitlines()
    reason, ev = go(world, run, 3, lambda r: Exact(), cfg={**ON, "last_round": 3}, continue_round=2, tag="redo2",
                    prior_seconds=7.0, test_hook=scripted())
    assert reason == "finished"
    new = ev[len(before):]
    assert [e["event"] for e in new][:2] == ["start_check", "diverged"]
    assert new[0]["passed"] and new[0]["max_diff_points"] == 0.0 and new[0]["compared_with_round"] == 1
    assert set(new[0]["diff_points"]) == {str(d) for d in range(3, 10)}
    assert new[1]["round"] == 2 and new[1]["after_the_fact"]
    assert not of(new, "check", round=2) and not of(new, "build", round=2) and not of(new, "skip_length", round=2)
    assert of(new, "check", round=3) and of(new, "build", round=3)
    redo, nxt = trainings[2:]
    assert redo["folder"] == "round_2-redo" and nxt["folder"] == "round_3"
    # the same start weights, examples (stored answers + the replay labelled by the start model) and updates
    assert redo["state"] == original[1]["state"] == checkpoint_digest(world, run, "round_1")
    assert redo["examples"] == original[1]["examples"] and redo["steps"] == original[1]["steps"]
    assert redo["seed"] == ld.stream_seed(5, 2, "batches-redo") != original[1]["seed"]
    assert nxt["state"] == checkpoint_digest(world, run, "round_2-redo")
    d2 = of(new, "round_done", round=2)[0]
    assert d2["continuation"] and d2["redone"] and d2["built"] == [] and d2["sec_check"] == 0
    assert of(new, "test", round=2)[0]["redo"]
    # nothing the original call wrote was changed
    assert (world["ladder"] / run / "rounds.jsonl").read_text().splitlines()[:len(before)] == before
    config = json.loads((world["ladder"] / run / "config-rounds2-redo2.json").read_text())
    assert config["continue_round"] == 2 and "/round_1/checkpoints/" in config["start_checkpoint"]


@pytest.mark.slow
def test_the_continuation_refuses_without_stored_answers(world, trainings):
    run = "a3-nopools"
    original_without_the_rule(world, run, 1)
    log_before = (world["ladder"] / run / "rounds.jsonl").read_text()
    with pytest.raises(FileNotFoundError):  # round 2 never built anything
        go(world, run, 3, lambda r: Exact(), cfg=ON, continue_round=2, tag="c2")
    index = json.loads((world["ladder"] / run / "kept" / "index.json").read_text())
    (world["ladder"] / run / "kept" / index["9"]["file"]).unlink()
    with pytest.raises(FileNotFoundError):
        go(world, run, 3, lambda r: Exact(), cfg=ON, continue_round=1, tag="c1")
    assert (world["ladder"] / run / "rounds.jsonl").read_text() == log_before
    assert not list((world["ladder"] / run).glob("config-rounds*-c*.json")) and len(trainings) == 1


@pytest.mark.slow
def test_the_start_check_ends_the_call_before_training(world, trainings):
    run = "a3-startcheck"
    original_without_the_rule(world, run, 1)

    def off_by(points):
        def hook(round_, kind, acc):
            acc = scripted()(round_, kind, acc)
            if kind == "start_check":
                acc["9"]["acc_total"] += points / 100
            return acc
        return hook

    reason, ev = go(world, run, 3, lambda r: Exact(), cfg=ON, continue_round=1, tag="c1", test_hook=off_by(0.3))
    assert reason == "start_check_failed"
    check = of(ev, "start_check")[-1]
    assert not check["passed"] and abs(check["max_diff_points"] - 0.3) < 1e-6
    assert of(ev, "start_check_failed") and of(ev, "call_done")[-1]["reason"] == "start_check_failed"
    assert len(trainings) == 1 and not of(ev, "diverged") and not (world["ladder"] / run / "round_1-redo").exists()
