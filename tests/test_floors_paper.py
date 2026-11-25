"""Amendment 27, parts 3 and 4: the outcome rule, the stitched repeat, macro names, and that the
paper files are new files only. The log-reading tests skip when the pilot's logs are absent."""
import pytest

from gainpred import floors_analysis as fa
from gainpred import floors_paper as fp
from gainpred.floors_report import RUNS


def test_outcome_rule_as_written_in_the_task():
    # before above 50%
    assert fa.outcome(90, 92.9, 100) == "did nothing" and fa.outcome(90, 87, 100) == "did nothing"
    assert fa.outcome(90, 86.9, 100) == "did harm" and fa.outcome(90, 93.1, 100) == "unclassified"
    # before 50% or lower
    assert fa.outcome(30, 95, 100) == "paid off"           # +65, within 10 of best
    assert fa.outcome(30, 50, 60) == "paid off"            # exactly +20 and exactly 10 below best
    assert fa.outcome(30, 39.9, 100) == "did nothing"      # less than 10 higher (falls count too)
    assert fa.outcome(30, 10, 100) == "did nothing"
    assert fa.outcome(30, 45, 100) == "taught wrong answers"  # higher, 55 below best
    assert fa.outcome(30, 45, 74) == "unclassified"        # higher, 29 below best, gain under 20
    assert fa.outcome(50, 75, 90) == "unclassified"        # +25 but 15 below best, and not 30 below


def test_macro_names_are_letters_only_and_spell_digits():
    assert fp.words(16) == "Sixteen" and fp.words(39) == "ThirtyNine" and fp.words(1500) == "FifteenHundred"
    name = fp.macro_name("Agr", "Floor", 2, "KeptRight")
    assert name == "gpClimbAgrFloorTwoKeptRight" and name.isalpha()
    reg = fp.Reg()
    reg.add(("X", 1), "5", "m")
    with pytest.raises(ValueError):
        reg.add(("X", 1), "6", "m")


def test_paper_writes_only_its_own_new_files():
    assert all(n.startswith(("table_climb", "figure_climb", "numbers_pilot")) for n in fp.NEW_FILES)
    assert "numbers.tex" not in fp.NEW_FILES


needs_logs = pytest.mark.skipif(not (RUNS / "solid-agr-r1-again" / "rounds.jsonl").exists(),
                                reason="pilot logs not fetched")


@needs_logs
def test_repeat_one_takes_floor_four_from_the_relaunch_only():
    evs = fa.run_events("solid-agr", 1)
    assert {e["floor"] for e in evs if "floor" in e} == {2, 3, 4}
    f4 = [e for e in evs if e.get("floor") == 4 and e["event"] == "round_done"]
    assert [e["round"] for e in f4] == [1, 2, 3, 4]  # not the two rounds of the crashed call as well


@needs_logs
def test_analyses_cover_every_length_and_case():
    a = fa.skip_table()
    assert len(a["rows"]) == 3 and a["confirmed_all_100"] and a["lowest_kept_share"] > 0.05
    er = fa.error_rates()
    assert len(er["before"]) == len(er["after"]) == 72
    cases = fa.rule_cases()
    assert len(cases) == 4 * 3 * 8 + 3 * 16  # floor 2 for four conditions, floors 3-4 for the agreement rule
    assert fa.rule_counts(cases)["total"]["cases"] == len(cases)


@needs_logs
def test_amendment_28_macros_come_after_every_earlier_one_and_from_the_logs():
    reg = fp.Reg()
    fp.hand_typed(reg)
    assert reg.items["gpClimbRehearsalRows"][0] == "80,000"
    low, high = (int(reg.items[k][0].replace(",", "")) for k in ("gpClimbUpdatesLow", "gpClimbUpdatesHigh"))
    assert 3_000 <= low <= high
    assert float(reg.items["gpClimbGainGapRight"][0]) >= 0
    assert "table_climb_control3.tex" in fp.NEW_FILES
    assert all(r.startswith("direct3-r") for r in fp.CONTROL3) and not set(fp.CONTROL3) & set(fp.CONTROL)
