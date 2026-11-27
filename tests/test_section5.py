"""Section 5 artefacts: the slim withheld table, the section 5 numbers, and caption wording."""
import re
from pathlib import Path

from gainpred.writeup import all_groups, section5_numbers, withheld_slim_rows

GENERATED = Path("paper/generated")


def test_slim_withheld_table_has_three_rows_and_the_label():
    rows = withheld_slim_rows()
    assert [r[0].split(":")[0] for r in rows] == ["Shown combinations", "Withheld combinations", "Answers never shown"]
    assert [r[1] for r in rows] == ["478", "877", "2,000"]
    assert all(re.fullmatch(r"\d+\.\d–\d+\.\d → \d+\.\d–\d+\.\d", r[2]) for r in rows)
    text = (GENERATED / "table_withheld_slim.tex").read_text()
    assert r"\label{tab:withheld}" in text
    assert r"\label{tab:withheld-full}" in (GENERATED / "table_withheld.tex").read_text()


def test_exp2_caption_counts_the_groups_all_pools():
    groups = all_groups()
    assert len(groups) == 10 and sum(g != "random_join" for g in groups) == 9
    text = " ".join((GENERATED / "table_exp2.tex").read_text().split())
    assert "All pools ten groups of 1,000 problems, nine of which contain a crossing run" in text


def test_section5_numbers_are_in_numbers_tex():
    nums = section5_numbers()
    generated = (GENERATED / "numbers.tex").read_text()
    for k, v in nums.items():
        assert rf"\newcommand{{\{k}}}{{{v}}}" in generated, k
    assert float(nums["gpKeptNoCarryMax"]) < 0.05 and nums["gpKeptNoCarry"] == "0.0"
    assert float(nums["gpKeptAgreementLow"]) < float(nums["gpNaturalCarryLow"])
    assert float(nums["gpBuildRatio"]) > 1


def test_exp2_rows_order_bold_and_zero_ranges():
    from gainpred.writeup import exp2_paper_rows
    rows, bold = exp2_paper_rows()
    assert [r[0] for r in rows] == ["Before the round", "Pick the longer", "Edges-differ rule (Shin et al.)",
                                    "Join at the split (ours)"]
    assert bold == {(3, j) for j in range(1, 6)}  # ours is best in every column
    assert rows[1][2:] == ["0.0"] * 4 and rows[0][4] == "0.0"  # zero columns print no range
    assert rows[0][3] == "1.5 (0.0–4.5)"  # a range that is not all zero stays
    text = (GENERATED / "table_exp2.tex").read_text()
    assert "withheld" not in text.lower() and r"\textbf{98.0} (96.9--98.6)" in text
    assert "Best in each column in bold." in " ".join(text.split())
    lines = text.splitlines()
    assert lines.index(r"\midrule", lines.index(r"\midrule") + 1) < next(
        i for i, l in enumerate(lines) if l.startswith("Join at the split (ours)"))


def test_hand_typed_section5_numbers_have_macros():
    """64.8 (control arm before the round, crossing runs of 7) and 57.3 (edges-differ starting
    models, crossing runs of 5) were typed by hand in Section 5; they now come from the data."""
    from gainpred.writeup import control_numbers, section5_numbers
    assert control_numbers()["gpCtrlBeforeSevenCorrect"] == "64.8"
    assert section5_numbers()["gpLrEdgesBeforeFive"] == "57.3"
    text = Path("paper/generated/numbers.tex").read_text()
    assert r"\newcommand{\gpCtrlBeforeSevenCorrect}{64.8}  % after" in text
    assert r"\newcommand{\gpLrEdgesBeforeFive}{57.3}" + "\n" in text
