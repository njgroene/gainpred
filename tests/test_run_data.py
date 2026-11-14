"""Tests for the longest-run datasets and hard problems (P1, P3, P6, P7), mirroring test_data.py."""

import random
from collections import Counter

import pytest

from gainpred.formats import RUN_QUESTIONS, run_answer_text, solve_run
from gainpred.run_data import (
    EASY_LENGTHS,
    HARD_LENGTHS,
    JOINED_LENGTHS,
    RANDOM_JOIN,
    RUN_ACROSS,
    TEST_EASY_PER_GROUP,
    TEST_STUDY_PER_GROUP,
    TRAIN_START_PER_LENGTH,
    HardRun,
    draw_hard,
    draw_pool,
    draw_run_across,
    hard_problems_of,
    left_lengths,
    load,
    make_all,
    make_train_start,
    questions_of,
    withheld,
    write_all,
)
from gainpred.formats import RunQuestion


@pytest.fixture(scope="session")
def sets():
    return make_all()


# --- P9 "Generators" -----------------------------------------------------------


def test_every_answer_matches_a_plain_solver(sets):
    for name in ("test_easy", "train_start"):
        for r, q in zip(sets[name], questions_of(sets[name])):
            assert r["answer"] == run_answer_text(solve_run(q)) and r["run_length"] == solve_run(q).length
    for r in sets["test_study"]:
        assert r["answer"] == run_answer_text(solve_run(RunQuestion("runlen", r["string"])))


def test_regenerating_gives_identical_files(tmp_path):
    first, second = write_all(tmp_path / "first"), write_all(tmp_path / "second")
    assert first == second
    for name in ("test_study", "test_easy", "train_start"):
        assert (tmp_path / "first" / f"{name}.jsonl").read_bytes() == (tmp_path / "second" / f"{name}.jsonl").read_bytes()
    assert load(tmp_path / "first" / "test_study.jsonl") == make_all()["test_study"]


# --- sizes and groups (P1, P3, P7) -----------------------------------------------


def test_set_sizes(sets):
    assert Counter((r["question"], r["length"]) for r in sets["test_easy"]) == {
        (k, n): TEST_EASY_PER_GROUP for k in RUN_QUESTIONS for n in EASY_LENGTHS}
    assert Counter((r["question"], r["length"]) for r in sets["train_start"]) == {
        (k, n): TRAIN_START_PER_LENGTH for k in RUN_QUESTIONS for n in EASY_LENGTHS}
    groups = Counter(r["group"] for r in sets["test_study"])
    assert set(groups.values()) == {TEST_STUDY_PER_GROUP} and len(groups) == 1 + len(JOINED_LENGTHS) + 2


def test_starting_training_asks_each_string_every_question(sets):
    by_string = Counter(r["string"] for r in sets["train_start"])
    assert set(by_string.values()) == {3}


def test_starting_training_questions_are_a_parameter():
    records = make_train_start(set(), per_length=10, questions=("runlen",))
    assert len(records) == 10 * len(EASY_LENGTHS) and {r["question"] for r in records} == {"runlen"}


def test_study_groups_are_labelled_correctly(sets):
    for r, p in zip(sets["test_study"], hard_problems_of(sets["test_study"])):
        assert p.length in HARD_LENGTHS and len(p.left) in EASY_LENGTHS and len(p.right) in EASY_LENGTHS
        if r["group"] == RANDOM_JOIN:
            assert p.kind == RANDOM_JOIN and p.a is None
        elif r["group"].startswith("joined_length_"):
            assert p.kind == RUN_ACROSS and p.a + p.b == int(r["group"].rsplit("_", 1)[1])
        else:
            assert p.kind == RUN_ACROSS and (p.a in (1, 2)) == (r["group"] == "left_part_1-2")


def test_run_across_lengths_are_exactly_a_plus_b(sets):
    for p in hard_problems_of(sets["test_study"]):
        if p.kind == RUN_ACROSS:
            stretch = p.string[p.join - p.a : p.join + p.b]
            assert len(set(stretch)) == 1
            assert p.string[p.join - p.a - 1] != stretch[0] and p.string[p.join + p.b] != stretch[0]


# --- separation (P7) ---------------------------------------------------------------


def test_no_string_appears_twice_in_any_set_or_across_sets(sets):
    seen: dict[str, str] = {}
    for name, records in sets.items():
        strings = {r["string"] for r in records}
        if name != "train_start":  # train_start repeats each string once per question
            assert len(strings) == len(records), name
        for s in strings:
            assert s not in seen, (s, seen.get(s), name)
            seen[s] = name


def test_digits_are_spread_evenly(sets):
    counts = Counter("".join(r["string"] for r in sets["train_start"][::3]))
    total = sum(counts.values())
    for digit in "0123456789":
        assert abs(counts[digit] / total - 0.1) < 0.002


# --- hard problems (P3, P6) --------------------------------------------------------


def test_left_lengths_keep_both_pieces_easy():
    assert list(left_lengths(12)) == [6] and list(left_lengths(16)) == [6, 7, 8, 9, 10] and list(left_lengths(20)) == [10]


def test_pool_is_half_and_half_and_varied():
    rng = random.Random(0)
    pool = draw_pool(rng, 16, 4000, exclude=set())
    kinds = Counter(p.kind for p in pool)
    assert abs(kinds[RUN_ACROSS] / 4000 - 0.5) < 0.03
    a_b = Counter((p.a, p.b) for p in pool if p.kind == RUN_ACROSS)
    assert set(a_b) == {(a, b) for a in range(1, 5) for b in range(1, 5)}
    assert set(Counter(p.join for p in pool)) == {6, 7, 8, 9, 10}


def test_pool_excludes_given_strings_and_is_reproducible():
    first = draw_pool(random.Random(5), 12, 200, exclude=set())
    assert first == draw_pool(random.Random(5), 12, 200, exclude=set())
    taken = {first[0].string}
    assert first[0].string not in {p.string for p in draw_pool(random.Random(5), 12, 200, exclude=taken)}


def test_run_across_construction():
    rng = random.Random(1)
    for _ in range(2000):
        p = draw_run_across(rng, rng.choice(list(HARD_LENGTHS)))
        symbol = p.string[p.join]
        assert p.string[p.join - p.a : p.join + p.b] == symbol * (p.a + p.b)
        assert p.string[p.join - p.a - 1] != symbol and p.string[p.join + p.b] != symbol


def test_withheld_drops_only_run_across_with_a_3_or_4():
    pool = [draw_hard(random.Random(i), 14) for i in range(400)]
    kept = withheld(pool)
    assert all(p.kind == RANDOM_JOIN or p.a in (1, 2) for p in kept)
    assert [p for p in pool if p not in kept] == [p for p in pool if p.kind == RUN_ACROSS and p.a in (3, 4)]


def test_hard_run_validates():
    with pytest.raises(ValueError):
        HardRun("123456123456", 0, RANDOM_JOIN)
    with pytest.raises(ValueError):
        HardRun("123456123456", 6, RUN_ACROSS)  # a, b missing


# --- frozen ------------------------------------------------------------------------

FROZEN_SHA256 = {
    "test_study": "31091c9a29e4fe0a80bb547b786132bf1f44fa8c51d8d596cbe3442063a2d415",
    "test_easy": "ad6b083844945031ee111e3a142efd5da52f73b669b55de997b6fb5615fe6769",
    "train_start": "26a7ece9ab919965763b1cb8d125c3441c41cc368b1b89ebcf236b79c12c6998",
}


def test_datasets_are_frozen(tmp_path):
    """Fingerprints as first generated on 14 November 2026. Change only on purpose, with a P10 amendment."""
    manifest = write_all(tmp_path)
    assert {name: info["sha256"] for name, info in manifest.items()} == FROZEN_SHA256
