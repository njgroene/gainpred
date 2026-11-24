"""Amendment 26: steered rounds with rehearsal, schedule only, stitched fixed arm; tables and macros."""
from pathlib import Path

import pytest

from gainpred import steer as st

GENERATED = Path("paper/generated")


@pytest.fixture(scope="module")
def ready():
    from gainpred.writeup import rehearsal_ready
    if not rehearsal_ready():
        pytest.skip("amendment-26 runs not fetched")


def test_stitched_fixed_arm_for_starting_points_0_and_1():
    for s in st.LEARNING:
        run = st.load("fixed", s)
        if run is None:
            pytest.skip("fixed runs not in results/")
        assert sorted(e["round"] for e in run.of("round_done")) == [1, 2, 3, 4]
        assert run.round_dirs == {1: f"si-agreement-seed{s}-authors", 2: f"si-agreement-seed{s}-authors",
                                  3: f"si-agreement-seed{s}-5cut", 4: f"si-agreement-seed{s}-5cut"}
        assert {e["set"] for e in run.of("eval") if e["round"] == 4} == {"easy", "study", "study5"}
        assert sum(e["kept"] for e in run.of("build")) == 200_000


def test_identity_passes_for_every_new_run(ready):
    for arm in ("rehearse", "schedule"):
        for s in st.ALL_SEEDS:
            ok, diffs = st.identity(arm, s)
            assert ok is True, diffs
            assert st.identity_not_compared(arm, s) == (["carries_by_place"] if s in st.LEARNING else [])


def test_schedule_only_builds_two_new_lengths_a_round_and_no_checks(ready):
    for s in st.ALL_SEEDS:
        run = st.load("schedule", s)
        assert [(e["round"], e["digits"]) for e in run.of("build")] == [(1, 8), (1, 9), (2, 10), (2, 11), (3, 12),
                                                                        (3, 13), (4, 14), (4, 15)]
        assert not run.of("checks") and not run.of("closing_checks")


def test_predictions_t1_t2_t3(ready):
    t = st.t_predictions()
    assert t["T1"]["pooled"]["held"] and t["T2"]["held"] and t["T3"]["pooled"]["held"]
    assert (t["T2"]["kept"], t["T2"]["fixed"]) == (160_000, 400_000)
    assert max(r["fall"] for r in st.not_rebuilt_falls("rehearse", st.LEARNING)) <= 3


def test_steered_and_schedule_differ_only_at_14_digits_round_4_of_models_2_and_4(ready):
    diffs = [(d["seed"], d["round"], d["digits"], d["steered"]) for s in st.ALL_SEEDS for d in st.decision_differences(s)]
    assert diffs == [(2, 4, 14, False), (4, 4, 14, False)]


def test_tables_fit_without_scaling(ready):
    for name, label in (("table_rehearsal", "rehearsal"), ("table_rehearsal_lengths", "rehearsal-lengths")):
        text = (GENERATED / f"{name}.tex").read_text()
        assert rf"\label{{tab:{label}}}" in text and r"\resizebox" not in text and r"\small" in text
    text = (GENERATED / "table_rehearsal.tex").read_text()
    assert "Starting models 0--1 (predictions recorded)" in text and "Starting models 2--4 (pilot, no prediction)" in text
    assert "check 1" not in text and "starting point" not in text


def test_macros(ready):
    from gainpred.writeup import rehearsal_numbers
    nums = rehearsal_numbers()
    assert (nums["gpRehTOne"], nums["gpRehTTwo"], nums["gpRehTThree"]) == ("held", "held", "held")
    assert (nums["gpRehKeptPct"], nums["gpRehRebuilt"], nums["gpRehLengthRounds"], nums["gpRehRepairRounds"]) == \
        ("40", "16", "40", "0")
    assert (nums["gpRehWorstGap"], nums["gpRehWorstGapDigits"], nums["gpRehWorstFall"]) == ("0.1", "10", "0.2")
    assert (nums["gpRehPilotKeptPct"], nums["gpRehPilotWorstGap"]) == ("37", "0.5")
    assert (nums["gpSchedKeptPct"], nums["gpSchedWorstGap"]) == ("40", "0.8")
    assert (nums["gpRehVsSchedKeptDiff"], nums["gpRehPilotVsSchedKeptDiff"]) == ("0", "-20")
    text = (GENERATED / "numbers.tex").read_text()
    for k, v in nums.items():
        assert rf"\newcommand{{\{k}}}{{{v}}}  % after" in text, k


def test_amendment_25_write_up_has_the_note_and_unchanged_numbers():
    text = Path("results/writeup/steer.md").read_text()
    assert text.splitlines()[2].startswith("**Read this first.**")
    assert "| Steered, novelty at least 3% | 6–8 | 4 (11 lengths) | 55 | 78 | 109 | 80.8 (43.0–99.8) | 55.2 | 99.9 |" in text
