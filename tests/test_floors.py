"""Pilot (building on self-taught pieces): splits, test set, streams, corruption, teachers, a tiny run.

The end-to-end tests use a tiny model on the CPU and a scripted "model" that adds perfectly
(and records what it was asked), so they check the plumbing, not learning.
"""
import gzip
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from gainpred import floors as fl
from gainpred.config import ModelConfig
from gainpred.formats import Addition, Unreadable, solve
from gainpred.teachers import PerfectCalculator

SPEC = {  # the spec's table, copied by hand
    16: "7+9 8+8 9+7", 17: "7+10 8+9 9+8", 18: "8+10 9+9 10+8", 19: "8+11 9+10 10+9", 20: "9+11 10+10 11+9",
    21: "9+12 10+11 11+10", 22: "10+12 11+11 12+10", 23: "10+13 11+12 12+11", 24: "8+16 16+8 7+17",
    25: "8+17 17+8 9+16", 26: "8+18 18+8 10+16", 27: "8+19 19+8 11+16", 28: "8+20 20+8 12+16",
    29: "8+21 21+8 13+16", 30: "8+22 22+8 14+16", 31: "8+23 23+8 15+16", 32: "8+24 24+8 7+25",
    33: "8+25 25+8 9+24", 34: "8+26 26+8 10+24", 35: "8+27 27+8 11+24", 36: "8+28 28+8 12+24",
    37: "8+29 29+8 13+24", 38: "8+30 30+8 14+24", 39: "8+31 31+8 15+24"}


# ---------------------------------------------------------------- check 2: the splits

def test_splits_are_exactly_the_spec_table_and_its_rule():
    assert {n: " ".join(f"{a}+{b}" for a, b in s) for n, s in fl.SPLITS.items()} == SPEC
    assert all(fl.split_rule(n) == fl.SPLITS[n] for n in fl.SPLITS)


@pytest.mark.parametrize("n", list(fl.SPLITS))
def test_split_rules(n):
    splits = fl.SPLITS[n]
    floor = fl.floor_of(n)
    assert all(sum(s) == n and len(s) == 2 for s in splits)
    assert len({s[0] for s in splits}) == 3  # three different cut positions
    need = {2: 8, 3: 16, 4: 24}[floor]  # at least one piece from the floor below
    assert all(max(s) >= need for s in splits)
    assert all(min(s) >= 7 for s in splits)
    assert (min(min(s) for s in splits) == 7) == (n in (16, 17, 24, 32))
    if n >= 18 and floor == 2:
        assert all(min(s) >= 8 for s in splits)  # both pieces self-taught


def test_floors_3_and_4_use_every_length_of_the_floor_below_as_a_piece():
    for floor in (3, 4):
        pieces = {w for n in fl.FLOORS[floor] for s in fl.SPLITS[n] for w in s}
        assert set(fl.FLOORS[floor - 1]) <= pieces


@pytest.mark.parametrize("n", list(fl.SPLITS))
def test_no_piece_starts_with_zero(n):
    rng = random.Random(n)
    for _ in range(300):
        p = fl.draw_floor(rng, n)
        for split in fl.SPLITS[n]:
            for piece, width in zip(fl.piece_problems(p, split), split):
                assert len(str(piece.a)) == width and len(str(piece.b)) == width


def test_rounds_teach_two_lengths_each():
    assert [fl.round_lengths(2, k) for k in (1, 2, 3, 4)] == [(16, 17), (18, 19), (20, 21), (22, 23)]
    assert fl.round_lengths(4, 4) == (38, 39)


# ---------------------------------------------------------------- check 3: the test set

@pytest.fixture(scope="module")
def floor_tests():
    path = fl.FLOOR_DATA_DIR / "test_floors.jsonl"
    if not path.exists():
        pytest.skip("data/floors not generated")
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_floor_test_set(floor_tests):
    from collections import Counter
    assert len(floor_tests) == 48_000
    groups = Counter((r["digits"], r["carry_across_cut1"]) for r in floor_tests)
    assert groups == {(d, c): 1_000 for d in range(16, 40) for c in (False, True)}
    for r in floor_tests[::97]:
        p = Addition(r["a"], r["b"])
        assert r["answer"] == solve(p) and fl.carry_crosses(p, fl.SPLITS[r["digits"]][0]) == r["carry_across_cut1"]
    assert len({(r["a"], r["b"]) for r in floor_tests}) == 48_000
    content = (fl.FLOOR_DATA_DIR / "test_floors.jsonl").read_bytes()
    assert json.loads((fl.FLOOR_DATA_DIR / "manifest.json").read_text())["test_floors"]["sha256"] == \
        hashlib.sha256(content).hexdigest()


def test_floor_test_set_regenerates_byte_for_byte(floor_tests):
    from gainpred.data import to_jsonl
    assert to_jsonl(fl.make_test_floors()) == (fl.FLOOR_DATA_DIR / "test_floors.jsonl").read_bytes()


def test_fresh_problems_carry_at_split_1_more_often_than_the_test_set():
    rng = random.Random(0)
    share = sum(fl.carry_crosses(p, fl.SPLITS[20][0]) for p in (fl.draw_floor(rng, 20) for _ in range(4000))) / 4000
    assert 0.55 < share < 0.67  # the test set has exactly 0.5


# ---------------------------------------------------------------- check 8 (amended): streams and repeats

def test_streams_differ_by_repeat_and_by_every_part():
    a = fl.draw_fresh(fl.stream("seed2", 0, 2, 1, 16, "build"), 16, 50, set())
    assert a == fl.draw_fresh(fl.stream("seed2", 0, 2, 1, 16, "build"), 16, 50, set())
    for other in (("seed2", 1, 2, 1, 16, "build"), ("seed3", 0, 2, 1, 16, "build"), ("seed2", 0, 3, 1, 16, "build"),
                  ("seed2", 0, 2, 2, 16, "build"), ("seed2", 0, 2, 1, 16, "checks")):
        assert a != fl.draw_fresh(fl.stream(*other), 16, 50, set())
    # the stream text is the owner's: "pilot | starting model | repeat | floor | round | digits | purpose"
    assert fl.stream_key("seed2", 1, 2, 3, 20, "build") == "pilot | seed2 | 1 | 2 | 3 | 20 | build"


def test_two_repeats_differ_only_in_the_repeat_number():
    r0 = asdict(fl.PilotConfig(teacher="no_carry", version="corrupted", corruption=0.15, repeat=0))
    r1 = asdict(fl.PilotConfig(teacher="no_carry", version="corrupted", corruption=0.15, repeat=1))
    assert {k for k in r0 if r0[k] != r1[k]} == {"repeat"}


def test_the_same_repeat_draws_the_same_problems_in_every_condition():
    """Streams take the starting model and repeat, never the teacher or the version of floor 1;
    and every teacher reads a prefix of the same stream whatever its keep rate."""
    rows_a, rows_k = [], []
    corr = fl.Corruption("seed2", 0.15)
    fl.collect("agreement", PerfectCalculator(), fl.stream("seed2", 1, 2, 1, 18, "build"), 18, 30, 600, set(), corr,
               rows_a)
    fl.collect("answer_key", PerfectCalculator(), fl.stream("seed2", 1, 2, 1, 18, "build"), 18, 30, 600, set(), None,
               rows_k)
    assert [(r["a"], r["b"]) for r in rows_k] == [(r["a"], r["b"]) for r in rows_a[:30]]


# ---------------------------------------------------------------- the spec's worked example

EXAMPLE = Addition(347530151542738677, 199090414712738008)


def test_the_specs_example_built_from_three_splits():
    built, info = fl.build_floor([EXAMPLE], (0, 1, 2), PerfectCalculator())
    assert [b.answer for b in built[0]] == [546620566255476685, 5466205651255476685, 546620566255476685]
    assert [(r["a"], r["b"]) for r in info[0][0]["pieces"]] == [(34753015, 19909041), (1542738677, 4712738008)]
    assert fl.decide("agreement", [EXAMPLE], built)[0].kept == 546620566255476685 == solve(EXAMPLE)
    assert fl.decide("no_carry", [EXAMPLE], built)[0].kept == 546620566255476685


class OneWrongPiece(PerfectCalculator):
    """The spec's corrupted example: the left answer of the 8 + 10 split is 54692056."""

    def __call__(self, problems, temperature=0.0):
        return [54692056 if (p.a, p.b) == (34753015, 19909041) else solve(p) for p in problems]


def test_a_wrong_piece_passes_the_no_carry_rule_and_not_the_agreement_rule():
    built, _ = fl.build_floor([EXAMPLE], (0, 1, 2), OneWrongPiece())
    assert fl.decide("no_carry", [EXAMPLE], built)[0].kept == 546920566255476685  # wrong, kept
    assert fl.decide("agreement", [EXAMPLE], built)[0].kept is None  # three different answers


# ---------------------------------------------------------------- check 4: the teacher asks the model for self-taught pieces

class Recording(PerfectCalculator):
    def __init__(self):
        self.asked = []

    def __call__(self, problems, temperature=0.0):
        self.asked += problems
        return super().__call__(problems)


@pytest.mark.parametrize("n", [16, 18, 23, 24, 31, 32, 39])
def test_the_teacher_asks_the_model_for_every_piece_and_at_least_one_self_taught(n):
    rng = random.Random(1)
    problems = [fl.draw_floor(rng, n) for _ in range(20)]
    solver = Recording()
    _, info = fl.build_floor(problems, (0, 1, 2), solver)
    assert len(solver.asked) == 20 * 6  # two pieces, three splits: six questions per problem
    assert all(p.n_digits < n for p in solver.asked)
    for per_problem in info:
        assert [s["split"] for s in per_problem] == [0, 1, 2]
        assert all(any(r["self_taught"] for r in s["pieces"]) for s in per_problem)
        assert all(r["said"] == r["used"] and r["right"] for s in per_problem for r in s["pieces"])


def test_the_no_carry_rule_builds_split_1_only():
    solver = Recording()
    fl.build_floor([EXAMPLE], fl.teacher_splits("no_carry"), solver)
    assert [(p.a, p.b) for p in solver.asked] == [(34753015, 19909041), (1542738677, 4712738008)]


# ---------------------------------------------------------------- check 5: corruption

def test_corruption_rate_only_self_taught_seeded_and_never_dependent_on_the_answer():
    rng = random.Random(3)
    problems = [fl.draw_floor(rng, 18) for _ in range(3000)]
    built, info = fl.build_floor(problems, (0, 1, 2), PerfectCalculator(), fl.Corruption("seed2", 0.15))
    rows = [r for per in info for s in per for r in s["pieces"]]
    replaced = [r for r in rows if r["replaced"]]
    assert 0.135 < len(replaced) / len(rows) < 0.165
    again = fl.build_floor(problems, (0, 1, 2), PerfectCalculator(), fl.Corruption("seed2", 0.15))[1]
    assert again == info  # seeded: identical on a second build
    for r in replaced:
        said, used = r["said"], r["used"]
        assert len(said) == len(used) and sum(x != y for x, y in zip(said, used)) == 1 and used[0] != "0"
    # the decision to replace is the same whatever the model answered
    p = problems[0]
    picks = [fl.corrupt(answer, "seed2", p, 1, 0, 0.15)[1] for answer in (123456789, 987654321, 555555555)]
    assert len(set(picks)) == 1
    unreadable = Unreadable("x")
    assert fl.corrupt(unreadable, "seed2", p, 1, 0, 1.0) == (unreadable, False)  # never replaced


def test_trusted_pieces_are_never_replaced():
    rng = random.Random(4)
    problems = [fl.draw_floor(rng, 16) for _ in range(2000)]  # split 1 is 7 + 9, split 3 is 9 + 7
    _, info = fl.build_floor(problems, (0, 1, 2), PerfectCalculator(), fl.Corruption("seed2", 0.5))
    rows = [r for per in info for s in per for r in s["pieces"]]
    assert not any(r["replaced"] for r in rows if not r["self_taught"])
    assert any(r["replaced"] for r in rows if r["self_taught"])


def test_corruption_is_a_function_of_starting_model_problem_split_and_piece():
    p = EXAMPLE
    a = [fl.corruption_selected("seed2", p, s, j, 0.5) for s in range(3) for j in range(2)]
    assert a == [fl.corruption_selected("seed2", p, s, j, 0.5) for s in range(3) for j in range(2)]
    assert any(fl.corruption_selected(m, q, 0, 0, 0.15) != fl.corruption_selected("seed2", q, 0, 0, 0.15)
               for m in ("seed3",) for q in (Addition(10**17 + i, 10**17 + 2 * i) for i in range(200)))


# ---------------------------------------------------------------- check 7: the answer key

def test_answer_key_keeps_the_true_sum_of_every_problem_drawn():
    rows = []
    kept, drawn, *_ = fl.collect("answer_key", PerfectCalculator(), fl.stream("seed2", 0, 2, 1, 20, "build"), 20, 25,
                                 500, set(), None, rows)
    assert drawn == 25 and all(a == solve(p) for p, a in kept)
    built, _ = fl.build_floor([EXAMPLE], (0, 1, 2), OneWrongPiece())
    assert fl.decide("answer_key", [EXAMPLE], built)[0].kept == solve(EXAMPLE)


# ---------------------------------------------------------------- change D: updates per round

def test_updates_per_round():
    assert fl.updates_for(125_000) == 3_500  # first round on floor 2
    assert fl.updates_for(345_000) == 9_500  # last round on floor 4
    assert fl.updates_for(100_000) == 3_000  # never below 3,000
    assert fl.updates_for(109_714) == 3_000 and fl.updates_for(109_715) == 3_100  # 7 x 109,715 / 256 is just over 3,000
    assert fl.PilotConfig().decay_fraction * 3_000 == pytest.approx(1_000)


def test_undertrained_decay_shortened_in_proportion():
    assert [round(1_000 * s / 3_000) for s in (500, 1_000, 1_500)] == [167, 333, 500]


# ---------------------------------------------------------------- check 1: length limit (owner's answer 15)

def test_long_loading_changes_only_max_len(tmp_path):
    from gainpred.model import Transformer
    from gainpred.tokenizer import PAD_ID, VOCAB_SIZE
    from gainpred.train import TrainConfig, load_model, save_checkpoint

    torch.manual_seed(0)
    model = Transformer(ModelConfig(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, n_layers=1, d_model=32, n_heads=2, d_ff=64))
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, model.configure_optimizer(), 1, TrainConfig())
    short, long = load_model(path), fl.load_long(path)
    assert short.cfg.max_len == 128 == ModelConfig(vocab_size=33, pad_id=31).max_len and long.cfg.max_len == 136
    for a, b in zip(short.state_dict().values(), long.state_dict().values()):
        assert torch.equal(a, b)
    from gainpred.teachers import ModelSolver
    p39 = fl.draw_floor(random.Random(0), 39)
    ModelSolver(long, torch.device("cpu"))([p39])  # 133 positions fit
    with pytest.raises(ValueError):
        ModelSolver(short, torch.device("cpu"))([p39])


# ---------------------------------------------------------------- checks 6, 9, 10 and the round loop: a tiny run

SMALL = dict(kept_per_length=20, replay_per_length=20, check_sample=10, batch_size=8, passes=1, min_steps=4,
             solver_batch_size=64, eval_per_length=5)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from gainpred.data import write_all
    from gainpred.model import Transformer
    from gainpred.tokenizer import PAD_ID, VOCAB_SIZE
    from gainpred.train import TrainConfig, save_checkpoint

    root = tmp_path_factory.mktemp("pilot")
    write_all(root / "data" / "addition")
    fl.write_floor_data(root / "data" / "floors")
    torch.manual_seed(0)
    model = Transformer(ModelConfig(vocab_size=VOCAB_SIZE, pad_id=PAD_ID, n_layers=1, d_model=32, n_heads=2, d_ff=64))
    solid = root / "solid" / "step_000001.pt"
    save_checkpoint(solid, model, model.configure_optimizer(), 1, TrainConfig())
    pilot = root / "runs" / "pilot"
    meta = fl.build_rehearsal(pilot, "seed2", solid, root / "data" / "addition", root / "data" / "floors",
                              torch.device("cpu"), per_length=20, make_solver=lambda m: PerfectCalculator())
    solver = Recording()

    def go(run, floor, first, last, start=solid, **kw):
        cfg = fl.PilotConfig(**{**SMALL, **kw.pop("cfg", {})})
        fl.run_floor(cfg, run, floor, first, last, start, pilot, root / "data" / "addition", root / "data" / "floors",
                     torch.device("cpu"), make_solver=lambda m, r: solver, **kw)
        return [json.loads(line) for line in (pilot / run / "rounds.jsonl").read_text().splitlines()]

    go("agr-r0", 2, 1, 2, seen=True)
    ev = go("agr-r0", 2, 3, 4)
    floor2_end = fl.last_checkpoint(pilot / "agr-r0" / "floor2" / "round_4")
    ev = go("agr-r0", 3, 1, 1, start=floor2_end)
    key = go("key-r0", 3, 1, 1, start=floor2_end, base_run="agr-r0", cfg={"teacher": "answer_key"})
    return {"root": root, "pilot": pilot, "ev": ev, "key": key, "meta": meta, "solver": solver}


@pytest.mark.slow
def test_rehearsal_set_built_once_audited_and_fingerprinted(tiny):
    meta = tiny["meta"]
    assert meta["rows"] == 160 and all(a["wrong"] == 0 and a["kept"] == 20 for a in meta["audit"].values())
    rows, sha = fl.load_rehearsal(tiny["pilot"], "seed2")
    assert len(rows) == 160 and sha == meta["sha256"]
    with pytest.raises(FileExistsError):
        fl.build_rehearsal(tiny["pilot"], "seed2", Path("x"), Path("x"), Path("x"), torch.device("cpu"))


@pytest.mark.slow
def test_round_loop_order_composition_and_updates(tiny):
    ev = [e for e in tiny["ev"] if e["floor"] == 2]
    assert [(e["round"], e["digits"]) for e in ev if e["event"] == "build"] == \
        [(1, 16), (1, 17), (2, 18), (2, 19), (3, 20), (3, 21), (4, 22), (4, 23)]
    for k in (1, 2, 3, 4):  # checks before the build in every round
        kinds = [e["event"] for e in ev if e.get("round") == k and e["event"] in ("checks", "floor1_checks", "build")]
        assert kinds == ["checks", "checks", "floor1_checks", "build", "build"]
    done = {e["round"]: e for e in ev if e["event"] == "round_done"}
    c3 = done[3]["composition"]
    assert c3["new_20"] == c3["new_21"] == 20 and all(c3[f"stored_{d}"] == 20 for d in (16, 17, 18, 19))
    assert c3["floor1_rehearsal"] == 160 and "stored_22" not in c3
    for e in done.values():
        assert e["examples"] == sum(e["composition"].values())
        assert e["updates"] == fl.updates_for(e["examples"], 1, 8, 4)
    cfg = json.loads((tiny["pilot"] / "agr-r0" / "floor2" / "round_3" / "config.json").read_text())
    assert cfg["train"]["steps"] == done[3]["updates"] and cfg["train"]["decay_fraction"] == pytest.approx(1 / 3)
    evals = [e for e in ev if e["event"] == "eval" and e["set"] == "floors"]
    assert {k.split("/")[0] for k in evals[-1]["accuracy"] if k != "all"} == {str(d) for d in range(16, 24)}
    assert not [e for e in ev if e["event"] == "eval" and e["round"] == 0 and e["set"] == "floors"]


@pytest.mark.slow
def test_checks_seen_coming_and_floor_below(tiny):
    ev = tiny["ev"]
    seen = next(e for e in ev if e["event"] == "seen_coming")
    assert set(seen["confirmed_share"]) == {"16", "17"} and set(seen["floor1_novelty"]) == {str(d) for d in range(8, 16)}
    assert seen["confirmed_share"]["16"]["teacher"] == "no_carry"
    below = next(e for e in ev if e["event"] == "floor_below_checks")
    assert below["floor"] == 3 and set(below["checks"]) == {str(d) for d in range(16, 24)}
    key_checks = [e for e in tiny["key"] if e["event"] == "checks"]
    assert key_checks and all(e["confirmed_share"] == "not applicable" for e in key_checks)


@pytest.mark.slow
def test_stored_answers_are_the_ones_kept_and_the_control_rehearses_the_same(tiny):
    pilot = tiny["pilot"]
    index = json.loads((pilot / "agr-r0" / "kept" / "index.json").read_text())
    assert index["16"]["round"] == 1 and index["23"]["round"] == 4 and index["24"]["floor"] == 3
    rows = fl._undump(pilot / "agr-r0" / "logs" / "floor2-round1-16-build.jsonl.gz")
    kept = [(int(r["a"]), int(r["b"]), int(r["kept"])) for r in rows if r["kept"] is not None]
    assert [(p.a, p.b, a) for p, a in fl.load_kept(pilot / "agr-r0", 16)] == kept
    control = json.loads((pilot / "key-r0" / "kept" / "index.json").read_text())
    assert all(control[str(d)]["sha256"] == index[str(d)]["sha256"] for d in range(16, 24))
    path = fl.kept_path(pilot / "agr-r0", 17)
    original = path.read_bytes()
    try:
        path.write_bytes(gzip.compress(b'{"a": 1, "b": 2, "answer": 3}\n'))
        with pytest.raises(RuntimeError):
            fl.load_kept(pilot / "agr-r0", 17)
    finally:
        path.write_bytes(original)


@pytest.mark.slow
def test_a_floor_is_never_overwritten(tiny):
    with pytest.raises(FileExistsError):
        fl.run_floor(fl.PilotConfig(**SMALL), "agr-r0", 2, 1, 1, Path("x"), tiny["pilot"], Path("x"), Path("x"),
                     torch.device("cpu"))


@pytest.mark.slow
def test_logs_record_every_problem_drawn_with_its_pieces(tiny):
    pilot = tiny["pilot"]
    ev = [e for e in tiny["ev"] if e["event"] == "build"]
    for e in ev:
        rows = fl._undump(pilot / "agr-r0" / "logs" / f"floor{e['floor']}-round{e['round']}-{e['digits']}-build.jsonl.gz")
        assert len(rows) == e["drawn"] and all(len(r["splits"]) == 3 for r in rows)
    assert any(p.n_digits >= 16 for p in tiny["solver"].asked)  # floor 3 asks floor-2 pieces


@pytest.mark.slow
def test_everything_stays_in_the_pilot_folder(tiny):
    root = tiny["root"]
    top = {p.name for p in root.iterdir()}
    assert top == {"data", "solid", "runs"} and {p.name for p in (root / "runs").iterdir()} == {"pilot"}


# ---------------------------------------------------------------- amendment 27: the direct-answers control

class WrongWholeSums(Recording):
    """Adds pieces perfectly, but answers every whole problem of 16+ digits one too high, and
    gives an unreadable answer to every problem whose first number is even."""

    def __call__(self, problems, temperature=0.0):
        self.asked += problems
        return [(Unreadable("x") if p.a % 2 == 0 else solve(p) + 1) if p.n_digits >= 16 else solve(p)
                for p in problems]


def test_direct_teacher_keeps_every_readable_whole_answer_and_never_uses_the_true_sum():
    problems = [EXAMPLE, Addition(EXAMPLE.a + 1, EXAMPLE.b)]  # first number odd, then even
    d = fl.decide("direct", problems, [[], []], WrongWholeSums()(problems))
    assert d[0].kept == solve(EXAMPLE) + 1 and d[1].kept is None  # wrong but readable: kept
    with pytest.raises(ValueError):
        fl.decide("direct", problems, [[], []])


def test_direct_teacher_asks_whole_problems_only_and_draws_the_agreement_rules_stream():
    solver = WrongWholeSums()
    rows = []
    kept, drawn, *_ = fl.collect("direct", solver, fl.stream("seed2", 1, 2, 1, 17, "build"), 17, 30, 500, set(),
                                 None, rows)
    assert solver.asked and all(p.n_digits == 17 for p in solver.asked)  # no pieces
    assert len(kept) == 30 and drawn == len(rows) and all(a == solve(p) + 1 for p, a in kept)
    assert all(r["splits"] == [] and "direct" in r for r in rows)
    agree = []
    fl.collect("agreement", PerfectCalculator(), fl.stream("seed2", 1, 2, 1, 17, "build"), 17, drawn, 500, set(),
               None, agree)
    assert [(r["a"], r["b"]) for r in rows] == [(r["a"], r["b"]) for r in agree][:drawn]
    assert fl.teacher_splits("direct") == ()


@pytest.mark.slow
def test_direct_control_run_end_to_end(tiny):
    """Two rounds of floor 2: wrong whole answers are kept, the checks are the agreement rule's,
    and round 2 rehearses this run's own stored answers."""
    pilot, root = tiny["pilot"], tiny["root"]
    solid = root / "solid" / "step_000001.pt"
    solver = WrongWholeSums()
    fl.run_floor(fl.PilotConfig(**{**SMALL, "teacher": "direct"}), "direct-r0", 2, 1, 2, solid, pilot,
                 root / "data" / "addition", root / "data" / "floors", torch.device("cpu"),
                 make_solver=lambda m, r: solver)
    ev = [json.loads(line) for line in (pilot / "direct-r0" / "rounds.jsonl").read_text().splitlines()]
    checks = [e for e in ev if e["event"] == "checks"]
    assert [e["digits"] for e in checks] == [16, 17, 18, 19] and all(e["teacher"] == "agreement" for e in checks)
    assert all(e["audit_direct_right"] == 0.0 for e in checks)
    builds = [e for e in ev if e["event"] == "build"]
    assert all(e["kept"] == 20 and e["kept_right"] == 0.0 and e["shortfall"] == 0 for e in builds)
    assert all(e["kept_share"] < 1 for e in builds)  # unreadables discarded
    for d in (16, 17):
        assert all(a == solve(p) + 1 for p, a in fl.load_kept(pilot / "direct-r0", d))
    done = {e["round"]: e for e in ev if e["event"] == "round_done"}
    assert done[2]["composition"]["stored_16"] == done[2]["composition"]["stored_17"] == 20
    assert json.loads((pilot / "direct-r0" / "config-floor2-rounds1-2.json").read_text())["pilot"]["teacher"] == "direct"


# ---------------------------------------------------------------- amendment 28: the control one floor up

def _tree(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_start_check_allows_at_most_two_tenths_of_a_point():
    assert fl.start_check_passes({"24": 0.999, "25": 0.9615}, {"24": 1.0, "25": 0.9615})
    assert fl.start_check_passes({"24": 0.998, "25": 0.9615}, {"24": 1.0, "25": 0.9615})
    assert not fl.start_check_passes({"24": 0.9975, "25": 0.9615}, {"24": 1.0, "25": 0.9615})
    assert not fl.start_check_passes({"24": 1.0}, {"24": 1.0, "25": 0.9615})


@pytest.mark.slow
def test_control_one_floor_up_copies_floor_two_checks_the_start_and_rehearses_its_own_floor_three(tiny):
    """direct3: starts from the agreement run's floor-2 model, copies (never moves or changes) its
    floor-2 answers, stops before building if the start check fails, and on floor 4 rehearses its
    own floor-3 answers."""
    pilot, root = tiny["pilot"], tiny["root"]
    start = fl.last_checkpoint(pilot / "agr-r0" / "floor2" / "round_4")
    agr_ev = [json.loads(line) for line in (pilot / "agr-r0" / "rounds.jsonl").read_text().splitlines()]
    expected = {str(e["digits"]): e["audit_direct_right"] for e in agr_ev
                if e.get("floor") == 3 and e["event"] == "checks" and e["round"] == 1}
    assert set(expected) == {"24", "25"}
    before = _tree(pilot / "agr-r0")

    def go(run, floor, first, last, checkpoint, solver, **kw):
        fl.run_floor(fl.PilotConfig(**{**SMALL, "teacher": "direct"}), run, floor, first, last, checkpoint, pilot,
                     root / "data" / "addition", root / "data" / "floors", torch.device("cpu"),
                     make_solver=lambda m, r: solver, **kw)
        return [json.loads(line) for line in (pilot / run / "rounds.jsonl").read_text().splitlines()]

    # wrong whole answers: 0% right instead of 100%, so the run stops before building anything
    with pytest.raises(RuntimeError, match="start check failed"):
        go("direct3-bad", 3, 1, 1, start, WrongWholeSums(), copy_lower_from="agr-r0", expect_direct_right=expected)
    bad = [json.loads(line) for line in (pilot / "direct3-bad" / "rounds.jsonl").read_text().splitlines()]
    assert [e for e in bad if e["event"] == "start_check"][0]["passed"] is False
    assert not any(e["event"] in ("build", "round_done") for e in bad)
    assert not (pilot / "direct3-bad" / "floor3").exists()

    ev = go("direct3-r0", 3, 1, 4, start, Recording(), copy_lower_from="agr-r0", expect_direct_right=expected)
    assert [e for e in ev if e["event"] == "start_check"][0]["passed"] is True
    src_index = json.loads((pilot / "agr-r0" / "kept" / "index.json").read_text())
    own_index = json.loads((pilot / "direct3-r0" / "kept" / "index.json").read_text())
    for d in range(16, 24):
        assert own_index[str(d)] == src_index[str(d)]
        assert fl.kept_path(pilot / "direct3-r0", d).read_bytes() == fl.kept_path(pilot / "agr-r0", d).read_bytes()
    assert all(own_index[str(d)]["floor"] == 3 for d in range(24, 32))
    config = json.loads((pilot / "direct3-r0" / "config-floor3-rounds1-4.json").read_text())
    assert config["base_run"] is None and config["copy_lower_from"] == "agr-r0"
    assert config["start_checkpoint"] == str(start) and set(config["copied"]) == {str(d) for d in range(16, 24)}
    assert _tree(pilot / "agr-r0") == before  # the agreement run's folder is untouched

    # floor 4 continues from its own floor 3 and rehearses its own floor-3 answers (agr-r0 has none at 26-31)
    ev = go("direct3-r0", 4, 1, 1, fl.last_checkpoint(pilot / "direct3-r0" / "floor3" / "round_4"), Recording())
    done = [e for e in ev if e.get("floor") == 4 and e["event"] == "round_done"][0]
    assert all(done["composition"][f"stored_{d}"] == 20 for d in range(16, 32))
    assert not any(e["event"] == "start_check" for e in ev if e.get("floor") == 4)
    assert _tree(pilot / "agr-r0") == before
