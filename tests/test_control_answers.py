"""Amendment 23: categories, piece facts, and checks against the saved answers."""
import json
from pathlib import Path

import pytest

from gainpred.control_answers import (CATEGORIES, RESULTS, SAVED_LENGTHS, category, logged_accuracy, models,
                                      piece_facts, prediction, tabulate)

SAVED = RESULTS.exists()


def row(answer, correct="7|7", longer="3|4"):
    return {"answer": answer, "correct": correct, "pick_the_longer": longer}


def test_categories():
    assert category(row("7|7")) == ("correct", None)
    assert category(row("7|6")) == ("right symbol, length 6", None)
    assert category(row("7|5")) == ("right symbol, other wrong length", 5)
    assert category(row("2|7")) == ("anything else", None)
    assert category(row("?7|")) == ("anything else", None)
    # a correct answer of length 6 is "correct", never "right symbol, length 6"
    assert category(row("7|6", correct="7|6")) == ("correct", None)
    t = tabulate([row("7|7"), row("7|6"), row("7|5"), row("3|4"), row("?x")])
    assert [t[c] for c in CATEGORIES] == [1, 1, 1, 2] and t["lengths"] == {5: 1} and t["pick the longer"] == 1


def test_piece_facts_take_the_left_piece_on_a_tie():
    # left "1122333" longest 3|3; right "444555" longest 4|3: a tie goes to the left piece
    r = {"string": "1122333444555", "join": 7, "a": 3, "b": 1}
    f = piece_facts(r)
    assert f["left_longest"] == "3|3" and f["right_longest"] == "4|3" and f["pick_the_longer"] == "3|3"


def test_prediction_threshold_is_half_of_the_wrong_answers():
    rows = [{"role": "b", "seed": 2, "crossing": 7, **row(a)} for a in ("7|6", "7|6", "7|5", "2|7", "7|7")]
    assert prediction(rows, (2,)) == (2, 4, True)
    rows[0]["answer"] = "7|4"
    assert prediction(rows, (2,)) == (1, 4, False)


@pytest.fixture(scope="module")
def saved():
    if not SAVED:
        pytest.skip("results/control_answers.jsonl not fetched")
    return [json.loads(line) for line in RESULTS.read_text().splitlines()]


def test_every_model_and_length_has_its_1000_problems(saved):
    for m in models():
        for n in SAVED_LENGTHS:
            assert sum(r["role"] == m["role"] and r["seed"] == m["seed"] and r["crossing"] == n for r in saved) == 1000


def test_saved_answers_reproduce_the_logged_accuracy(saved):
    if not Path("runs/runs-edges_differ-seed2-u30000/rounds.jsonl").exists():
        pytest.skip("runs/ is local only")
    for m in models():
        logged = logged_accuracy(Path("runs"), m)
        for n in SAVED_LENGTHS:
            rs = [r for r in saved if r["role"] == m["role"] and r["seed"] == m["seed"] and r["crossing"] == n]
            assert sum(r["answer"] == r["correct"] for r in rs) / len(rs) == logged[f"joined_length_{n}"], (m, n)


def test_saved_rows_are_consistent_with_their_strings(saved):
    from gainpred.formats import longest_run, run_answer_text
    for r in saved[:4000]:
        assert run_answer_text(longest_run(r["string"])) == r["correct"]
        assert r["left_part"] + r["right_part"] == r["crossing"]
        facts = piece_facts({"string": r["string"], "join": _join(r), "a": r["left_part"], "b": r["right_part"]})
        assert facts["pick_the_longer"] == r["pick_the_longer"]


def _join(r):
    """The join, from the test file (rows keep the problem's line number)."""
    if not hasattr(_join, "records"):
        _join.records = [json.loads(line) for line in Path("data/longest_run/test_study.jsonl").read_text().splitlines()]
    rec = _join.records[r["problem"]]
    assert rec["string"] == r["string"]
    return rec["join"]


def test_categories_partition_every_model_and_length(saved):
    for m in models():
        for n in SAVED_LENGTHS:
            t = tabulate([r for r in saved if r["role"] == m["role"] and r["seed"] == m["seed"] and r["crossing"] == n])
            assert sum(t[c] for c in CATEGORIES) == t["n"] == 1000
            assert sum(t["lengths"].values()) == t["right symbol, other wrong length"]


def test_paper_table_and_numbers_match_the_saved_answers(saved):
    from gainpred.writeup import control_numbers, control_tex
    text = control_tex()
    assert r"\label{tab:control-answers}" in text and "computed after all experiments" in text
    nums = control_numbers()
    assert len(nums) == 10  # nine pooled shares, plus the control arm before the round (gpCtrlBeforeSevenCorrect)
    generated = Path("paper/generated/numbers.tex").read_text()
    for k, v in nums.items():
        assert rf"\newcommand{{\{k}}}{{{v}}}  % after" in generated, k
