"""Tests for the addition datasets: P9's "Generators" acceptance test and P7's rules.

Generating everything takes a few seconds, so it is done once per test session
(the ``sets`` fixture) and shared by the tests below.
"""

from collections import Counter

import pytest

from gainpred.cuts import ADDITION_CUTS, EASY_DIGITS, HARD_DIGITS, carry_crosses
from gainpred.data import (
    TEST_EASY_PER_LENGTH,
    TEST_REPRODUCTION_DIGITS,
    TEST_REPRODUCTION_PER_LENGTH,
    TEST_STUDY_PER_GROUP,
    TRAIN_START_PER_LENGTH,
    hard_nonzero_positions,
    load,
    make_all,
    problems_of,
    write_all,
)
from gainpred.formats import Addition


@pytest.fixture(scope="session")
def sets():
    return make_all()


# ----------------------------------------------------------------------------
# P9 "Generators": answers are right, and regeneration is identical
# ----------------------------------------------------------------------------


def test_every_answer_matches_plain_arithmetic(sets):
    for name, records in sets.items():
        for r in records:
            assert r["answer"] == r["a"] + r["b"], (name, r)


def test_regenerating_gives_identical_files(tmp_path):
    """Two independent generations must produce byte-identical files."""
    first = write_all(tmp_path / "first")
    second = write_all(tmp_path / "second")
    assert first == second
    for name in first:
        assert (tmp_path / "first" / f"{name}.jsonl").read_bytes() == (
            tmp_path / "second" / f"{name}.jsonl"
        ).read_bytes()
    # And the files read back to the same records that were generated.
    assert load(tmp_path / "first" / "test_easy.jsonl") == make_all()["test_easy"]


# ----------------------------------------------------------------------------
# Sizes and groups (P1, P7)
# ----------------------------------------------------------------------------


def test_set_sizes(sets):
    def per_length(name):
        return Counter(r["digits"] for r in sets[name])

    assert per_length("test_study") == {d: 2 * TEST_STUDY_PER_GROUP for d in HARD_DIGITS}
    assert per_length("test_reproduction") == {d: TEST_REPRODUCTION_PER_LENGTH for d in TEST_REPRODUCTION_DIGITS}
    assert per_length("test_easy") == {d: TEST_EASY_PER_LENGTH for d in EASY_DIGITS}
    assert per_length("train_start") == {d: TRAIN_START_PER_LENGTH for d in EASY_DIGITS}


def test_both_numbers_have_the_stated_length(sets):
    for records in sets.values():
        for r in records:
            assert len(str(r["a"])) == len(str(r["b"])) == r["digits"]


def test_study_groups_are_labelled_correctly(sets):
    """Each study problem's label matches whether a carry really crosses cut 1,
    and each length has exactly 1,000 of each."""
    counts = Counter()
    for r in sets["test_study"]:
        problem = Addition(r["a"], r["b"])
        assert carry_crosses(problem, ADDITION_CUTS[r["digits"]][0]) == r["carry_across_cut1"]
        counts[r["digits"], r["carry_across_cut1"]] += 1
    assert set(counts.values()) == {TEST_STUDY_PER_GROUP}


def test_hard_problems_have_no_piece_starting_with_zero(sets):
    """P2: under each of the three cuts, no piece of either number starts with 0."""
    for r in sets["test_study"]:
        for number in (r["a"], r["b"]):
            digits = str(number)
            assert all(digits[i] != "0" for i in hard_nonzero_positions(r["digits"])), r


# ----------------------------------------------------------------------------
# Separation (P7)
# ----------------------------------------------------------------------------


def test_problems_are_unique_within_each_set(sets):
    for name, records in sets.items():
        problems = problems_of(records)
        assert len(set(problems)) == len(problems), name


def test_no_problem_appears_in_two_sets(sets):
    """In particular, no test problem is in the training set (P7)."""
    seen: dict[Addition, str] = {}
    for name, records in sets.items():
        for problem in problems_of(records):
            assert problem not in seen, f"{problem} in both {seen.get(problem)} and {name}"
            seen[problem] = name


def test_digits_are_spread_evenly(sets):
    """A cheap check that the digit drawing is not biased: in the training set,
    the non-leading digits 0-9 should each appear about 10% of the time."""
    counts = Counter()
    for r in sets["train_start"]:
        for number in (r["a"], r["b"]):
            counts.update(str(number)[1:])
    total = sum(counts.values())
    for digit in "0123456789":
        assert abs(counts[digit] / total - 0.1) < 0.002, (digit, counts[digit] / total)


# ----------------------------------------------------------------------------
# The datasets are frozen
# ----------------------------------------------------------------------------

FROZEN_SHA256 = {
    "test_study": "27272dece2e2644a8b2ac0765c76e8abe8989ba74b7b921e45aebb453b124b28",
    "test_reproduction": "30e86244025f33a818228c32a3bbf804ab73dcc0c3e417b8b370410d3021a840",
    "test_easy": "383152bad978a884461f0078b5ebd8a3b8f42fa427f74d0997d0c14e1a569179",
    "train_start": "1f43a85d4e1958c8276236f4c31b53a048bee98d6abe00aed79a367622dac325",
}


def test_datasets_are_frozen(tmp_path):
    """The fingerprints of the files as first generated on 13 November 2026.

    If this fails, something changed the data: a seed, a size, the drawing
    code, or the Python version's ``random()``. Every result so far was
    computed on the old data. Update these values only on purpose, and record
    the change and the reason in the amendments log (P10).
    """
    manifest = write_all(tmp_path)
    assert {name: info["sha256"] for name, info in manifest.items()} == FROZEN_SHA256
