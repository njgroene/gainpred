"""Amendment 25: steered rounds -- identity check, predictions, tables and macros."""
from pathlib import Path

import pytest

from gainpred import steer as st

GENERATED = Path("paper/generated")


@pytest.fixture(scope="module")
def ready():
    from gainpred.writeup import steer_ready
    if not steer_ready():
        pytest.skip("steered runs not fetched")


def test_identity_check_passes_for_every_steered_run(ready):
    for arm in st.arms_present():
        if arm == "fixed":
            continue
        for s in st.SEEDS:
            ok, diffs = st.identity(arm, s)
            assert ok is not False, diffs
            assert not st.check_differences(arm, s)


def test_round_one_teaches_both_lengths_so_identity_applies(ready):
    for arm in ("steer3", "frozen"):
        assert all(st.identity(arm, s)[0] is True for s in st.SEEDS)


def test_every_scheduled_length_round_is_decided_once(ready):
    for arm in ("steer3", "frozen"):
        for s in st.SEEDS:
            rows = [r for r in st.round_rows(arm, s) if not r["repair"]]
            assert len(rows) == st.LENGTH_ROUNDS
            assert all((r["kept"] > 0) == (r["decision"] == "teach") for r in rows)


def test_s1_and_s2_as_computed(ready):
    s1, s2 = st.s1(), st.s2()
    assert [s1[s]["held"] for s in st.SEEDS] == [True, True, False] and not s1["pooled"]["held"]
    assert f"{s1[4]['worst_hard']:.1f}" == "55.2" and s1[4]["worst_hard_digits"] == 12
    assert (s2["kept"], s2["fixed"], s2["held"]) == (330_000, 600_000, False)


def test_trigger_fired(ready):
    t = st.trigger()
    assert t and all(r["fall"] > 3 and r["decision"] == "skip" for r in t)


def test_tables_and_macros(ready):
    from gainpred.writeup import steer_numbers
    assert r"\label{tab:steer}" in (GENERATED / "table_steer.tex").read_text()
    lengths = (GENERATED / "table_steer_lengths.tex").read_text()
    assert r"\label{tab:steer-lengths}" in lengths and lengths.count(r" \\") == 9  # header + 8 lengths
    nums = steer_numbers()
    assert nums["gpSteerLevel"] == "3" and nums["gpSteerLengthRounds"] == "60"
    assert (nums["gpSteerSOne"], nums["gpSteerSTwo"]) == ("failed", "failed")
    assert (nums["gpSteerWorstGap"], nums["gpSteerWorstGapDigits"], nums["gpSteerWorstGapModel"]) == ("55.2", "12", "4")
    assert nums["gpSteerKeptPct"] == "55" and nums["gpSteerTaught"] == "22"
    assert nums["gpSteerRehearsalRun"] == ("yes" if "rehearse" in st.arms_present() else "no")
    text = (GENERATED / "numbers.tex").read_text()
    for k, v in nums.items():
        assert rf"\newcommand{{\{k}}}{{{v}}}  % after" in text, k


def test_steered_runs_are_not_in_p9b():
    import csv
    runs = {r["run"] for r in csv.DictReader(Path("results/p9b_cells.csv").open())}
    assert not any("steer" in r for r in runs) and len(runs) > 0
