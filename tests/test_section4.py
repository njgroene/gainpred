"""Section 4's generated numbers and tables (items 1-8), against the values in the owner's brief
of 19 November 2026, and the paper's vocabulary in every generated file (item 9).

Everything is built from the committed results/p9b_cells.csv and the logs; nothing is rescored."""
import re
from pathlib import Path

import pytest

from gainpred import p9b
from gainpred.writeup import (evaluation_cases, margins, no_outcome_tex, rise_groups, round_level_tex,
                              section4_numbers, weak_tex)

GENERATED = Path("paper/generated")


@pytest.fixture(scope="module")
def nums():
    return section4_numbers()


def test_numbers_section_4_cites(nums):
    expected = {
        "gpCases": "258", "gpCorrect": "249", "gpCorrectPct": "96.5",
        "gpCasesAll": "270", "gpNoOutcome": "12", "gpCorrectAllPct": "92.2",
        "gpCorrectModelTwo": "84", "gpCorrectModelThree": "82", "gpCorrectModelFour": "83",
        "gpCasesModelTwo": "86", "gpCasesModelThree": "86", "gpCasesModelFour": "86",
        "gpHarmful": "43", "gpDidHarm": "15", "gpTaughtWrong": "28",
        "gpByBasics": "10", "gpByBasicsCorrect": "1", "gpByNovelty": "175", "gpByNoveltyCorrect": "175",
        "gpByMinKept": "6", "gpByAgreement": "73", "gpByAgreementCorrect": "73",
        "gpCorrectNoBasics": "255", "gpCorrectNoBasicsPct": "98.8",
        "gpBaselineCorrect": "224", "gpBaselineCorrectPct": "86.8", "gpVariantCorrect": "242",
        "gpVariantCorrectPct": "93.8", "gpOursHarmFlagged": "43", "gpBaselineHarmFlagged": "38",
        "gpVariantHarmFlagged": "33", "gpBaselineCostPct": "56", "gpChecksCostPct": "9",
        "gpOursWeakCorrect": "1", "gpBaselineWeakCorrect": "4", "gpVariantWeakCorrect": "6", "gpWeakCases": "10",
        "gpRoundsCorrect": "52", "gpRounds": "57", "gpRiseNothingUnderTwo": "25",
    }
    assert {k: nums[k] for k in expected} == expected


def test_numbers_file_defines_every_number_once():
    text = (GENERATED / "numbers.tex").read_text()
    names = re.findall(r"\\newcommand\{\\(\w+)\}", text)
    assert len(names) == len(set(names)) and "gpCorrect" in names
    assert r"\input{generated/numbers}" in Path("paper/main.tex").read_text()


def test_margins(nums):
    m = margins(evaluation_cases())  # raises if a level inside either margin changes a prediction
    assert (nums["gpNoveltyMarginLow"], nums["gpNoveltyMarginHigh"]) == ("40.2", "60.9")
    assert (nums["gpConfirmedMarginLow"], nums["gpConfirmedHarmCases"]) == ("27.8", "42")
    assert (nums["gpConfirmedMarginHigh"], nums["gpConfirmedPaysCases"]) == ("50.7", "31")
    # The frozen levels lie inside the margins.
    assert m["novelty_hi"] < p9b.LEVEL_1 <= m["novelty_lo"] and m["confirmed_hi"] < 0.38 <= m["confirmed_lo"]


def test_rise_table():
    r = rise_groups(evaluation_cases())
    got = {k: (len(v), round(min(v), 1), round(max(v), 1)) for k, v in r.items()}
    assert got == {"Did little or nothing": (27, -1.6, 8.5), "Taught wrong answers": (28, 12.8, 50.0),
                   "Paid off": (37, 69.9, 100.0)}
    assert sum(x < 2 for x in r["Did little or nothing"]) == 25


def test_cases_table_columns():
    text = (GENERATED / "table_p9b_cases.tex").read_text()
    assert "Outcomes & Decided by" in text
    assert "did nothing 36, taught wrong answers 24" in text and "novelty / agreement" in text


def test_weak_and_no_outcome_tables():
    weak = weak_tex()
    assert "The 12 cases from weak starting models" in weak and weak.count("without an outcome &") == 2
    none = no_outcome_tex()
    assert "The 12 evaluation cases without an outcome" in none
    assert none.count("started above 50") == 10 and none.count("rose 20 points or more") == 2


def test_baseline_table():
    text = (GENERATED / "table_baseline.tex").read_text()
    assert "Harmful cases flagged" in text
    for row in ("43/43", "38/43", "33/43", "1/10 (10.0)", "4/10 (40.0)", "6/10 (60.0)"):
        assert row in text


def test_round_level_table():
    text = round_level_tex(evaluation_cases())
    assert "52 of 57" in text and "defined after the evaluation runs" in text and "tab:round-level" in text


OLD_TERMS = [r"\bverdicts?\b", r"\bcells?\b", r"\bmatched\b", r"\b(un)?classified\b", r"\bconfirming\b",
             r"\blearning runs?\b", r"\bstarting points?\b", r"\bcheck (0|1|2b|3)\b", r"\b2b\b", r"\bcuts?\b",
             r"\breference answers?\b", r"\bstudy[- ](set|group)\b", r"\brun f\b", r"\bjoined\b",
             r"\brun-across\b"]


def test_generated_files_use_the_papers_words():
    for f in sorted(GENERATED.glob("*.tex")):
        for line in f.read_text().splitlines():
            if line.startswith("% Generated"):
                continue
            for term in OLD_TERMS:
                assert not re.search(term, line, re.I), f"{f.name}: {term!r} in {line!r}"
