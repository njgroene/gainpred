"""Amendment 24: novelty against the change in accuracy (Part A), teacher accuracy (Part B)."""
from pathlib import Path

import pytest

from gainpred.teacher_accuracy import CSV, case_row, consistent, kind, plan, predictions
from gainpred.writeup import novelty_cases, novelty_gain_tex, novelty_numbers, sound_between

GENERATED = Path("paper/generated")


# ---------------------------------------------------------------- Part A: expected values (owner, 24 Nov 2026)

@pytest.fixture(scope="module")
def nums():
    return novelty_numbers()


def test_all_270_evaluation_cases_by_kind():
    from collections import Counter
    counts = Counter(c["kind"] for c in novelty_cases())
    assert sum(counts.values()) == 270
    assert counts == {"No-carry rule": 60, "Agreement rule": 60, "No selection": 60, "Sample and vote": 60,
                      "No selection after the no-carry rule": 18, "Weak starting models": 12}


def test_change_within_novelty_plus_3_in_269_of_270(nums):
    assert (nums["gpNovGainWithin"], nums["gpNovGainCases"]) == ("269", "270")


def test_the_one_exception(nums):
    out = [c for c in novelty_cases() if abs(c["change"]) > c["novelty"] + 3]
    assert len(out) == 1 and out[0]["run"] == "si-nocarry-seed2-authors"
    assert (nums["gpNovGainExcModel"], nums["gpNovGainExcRound"], nums["gpNovGainExcDigits"]) == ("2", "3", "12")
    assert (nums["gpNovGainExcBefore"], nums["gpNovGainExcAfter"], nums["gpNovGainExcNovelty"]) == \
        ("62.6", "100.0", "30.6")


def test_sound_teachers_120_cases(nums):
    assert nums["gpNovGainSound"] == "120"


def test_sound_novelty_matches_student_error(nums):
    assert (nums["gpNovGainSoundError"], nums["gpNovGainSoundErrorGap"]) == ("118", "6.8")


def test_sound_change_matches_novelty(nums):
    assert (nums["gpNovGainSoundChange"], nums["gpNovGainSoundChangeGap"]) == ("118", "6.8")


def test_sound_cases_with_novelty_between_5_and_95(nums):
    assert nums["gpNovGainSoundBetween"] == "17"


def test_no_selection_after_the_no_carry_rule_novelty_near_the_natural_carry_rate(nums):
    assert (nums["gpNovGainRunfCases"], nums["gpNovGainRunfLow"], nums["gpNovGainRunfHigh"]) == ("12", "60.9", "62.3")


def test_sample_and_vote_hardly_moves(nums):
    assert (nums["gpNovGainVote"], nums["gpNovGainVoteMax"]) == ("60", "1.6")


def test_weak_starting_models_change_range(nums):
    assert (nums["gpNovGainWeak"], nums["gpNovGainWeakLow"], nums["gpNovGainWeakHigh"]) == ("12", "0.6", "98.0")


def test_macros_are_in_numbers_tex_marked_after(nums):
    text = (GENERATED / "numbers.tex").read_text()
    for k, v in nums.items():
        assert rf"\newcommand{{\{k}}}{{{v}}}  % after" in text, k


def test_novelty_gain_table_has_the_17_cases_sorted_by_novelty():
    cases = sound_between(novelty_cases())
    assert len(cases) == 17 and [c["novelty"] for c in cases] == sorted(c["novelty"] for c in cases)
    tex = novelty_gain_tex()
    assert r"\label{tab:novelty-gain}" in tex and tex.count(r" \\") == 18  # header + 17 rows
    assert (GENERATED / "table_novelty_gain.tex").exists()


def test_novelty_gain_figure_and_wrapper():
    tex = (GENERATED / "figure_novelty_gain.tex").read_text()
    assert r"\label{fig:novelty-gain}" in tex and "computed after the evaluation runs" in tex
    assert (GENERATED / "figure_novelty_gain.pdf").exists()


# ---------------------------------------------------------------- Part B: definitions on made-up problems

def prob(a, b, kept, source, built, direct):
    return {"a": a, "b": b, "kept": kept, "source": source, "built": built, "direct": direct}


def test_case_row_quantities():
    ps = [prob(1, 2, 3, 0, ["3", "3"], "3"),        # right, confirmed, not new
          prob(1, 2, 4, 0, ["4", "4"], "3"),        # wrong but confirmed, new
          prob(1, 2, 4, 0, ["4", "3"], "3"),        # wrong, unconfirmed
          prob(5, 5, 10, 0, ["10", "?x"], "?x"),    # right, unconfirmed (unreadable never confirms)
          prob(5, 5, None, None, ["9", "?y"], "10")]  # set aside
    r = case_row(ps)
    assert r["kept"] == 4 and r["problems"] == 5
    assert r["teacher_accuracy"] == 0.5 and r["confirmed_share"] == 0.5 and r["wrong_confirmed"] == 0.25
    assert r["accuracy_confirmed"] == 0.5 and r["accuracy_unconfirmed"] == 0.5 and r["unconfirmed"] == 2
    assert r["novelty"] == 0.75 and r["student_accuracy"] == 0.75 and r["student_accuracy_all"] == 0.8
    assert consistent(r)


def test_the_kept_cut_never_confirms_itself():
    # agreement-style source 1: cut 1 built the same answer, so it is confirmed; cut 2 alone is not
    assert case_row([prob(1, 1, 2, 1, ["2", "2"], "2")])["confirmed_share"] == 1.0
    assert case_row([prob(1, 1, 2, 1, ["3", "2"], "2")])["confirmed_share"] == 0.0


def test_plan_is_the_210_cases_of_the_splitting_teachers():
    p = plan()
    assert len(p) == 18 and sum(len(x["cases"]) for x in p) == 210
    assert {x["kind"] for x in p} == {"No-carry rule", "Agreement rule", "No selection",
                                      "No selection after the no-carry rule", "Weak starting models"}
    assert kind("si-noselect-after-nocarry-seed3-authors-r2", "no_selection") == \
        "No selection after the no-carry rule"


def test_predictions_leave_out_the_weak_models():
    base = {"seed": 2, "wrong_confirmed": 0.0, "accuracy_confirmed": 1.0}
    rows = [{**base, "kind": "No-carry rule"}, {**base, "kind": "Weak starting models", "wrong_confirmed": 0.5,
                                                 "accuracy_confirmed": 0.1}]
    assert predictions(rows)["pooled"]["p1"] and predictions(rows)["pooled"]["p2"]
    rows[0]["wrong_confirmed"] = 0.021
    assert not predictions(rows)["pooled"]["p1"] and not predictions(rows)["seed 2"]["p1"]


# ---------------------------------------------------------------- Part B: the saved results

@pytest.fixture(scope="module")
def ta():
    if not CSV.exists():
        pytest.skip("results/teacher_accuracy.csv not fetched")
    from gainpred.teacher_accuracy import load_rows
    return load_rows()


def test_every_case_reproduces_the_logged_checks(ta):
    assert len(ta) == 210
    for r in ta:
        assert r["kept"] == int(r["logged_kept"])
        if r["kept"]:
            assert r["novelty"] == float(r["logged_novelty"]) and r["confirmed_share"] == float(r["logged_confirmed"])


def test_consistency_check_holds_in_every_case(ta):
    assert all(consistent(r) for r in ta)


def test_saved_answers_recompute_the_csv(ta):
    from gainpred.teacher_accuracy import answers_path, load_answers
    p = plan()[0]
    if not answers_path(p["name"]).exists():
        pytest.skip("per-problem answers not fetched")
    for e in load_answers(p["name"]):
        assert len(e["problems"]) == 2000
        row = next(r for r in ta if r["run"] == p["run"] and r["round"] == e["round"] and r["digits"] == e["digits"])
        assert case_row(e["problems"])["teacher_accuracy"] == row["teacher_accuracy"]


def test_teacher_accuracy_paper_files(ta):
    from gainpred.writeup import teacher_accuracy_numbers
    assert r"\label{tab:teacher-accuracy}" in (GENERATED / "table_teacher_accuracy.tex").read_text()
    assert r"\label{fig:teacher-accuracy}" in (GENERATED / "figure_teacher_accuracy.tex").read_text()
    text = (GENERATED / "numbers.tex").read_text()
    for k, v in teacher_accuracy_numbers().items():
        assert rf"\newcommand{{\{k}}}{{{v}}}  % after" in text, k
