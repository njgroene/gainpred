"""Tests of gainpred.ladder_paper: the ladder's numbers, tables, figures and data files (task 4).

They read the committed logs in results/ladder/ and write only into a temporary folder."""

from __future__ import annotations

import csv
import re

import pytest

from gainpred import ladder_paper as lp
from gainpred.ladder_report import of


@pytest.fixture(scope="module")
def arms():
    return lp.load_arms()


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    root = tmp_path_factory.mktemp("ladder_paper")
    paper, results = root / "paper", root / "results"
    paths = lp.write_all(paper, results)
    return paper, results, paths


def test_section2_values(arms):
    """The nine values the task gives (section 2), as computed here."""
    assert lp.section2(arms) == lp.EXPECTED


def test_macro_names_are_letters_only(arms):
    reg = lp.build_register(arms)
    assert reg.items
    for name in reg.items:
        assert re.fullmatch(r"gpLad[A-Za-z]+", name), name


def test_tex_and_md_list_the_same_macros(written):
    paper, results, _ = written
    tex = set(re.findall(r"\\newcommand\{\\(gpLad[A-Za-z]+)\}", (paper / "numbers_ladder.tex").read_text()))
    md = set(re.findall(r"^\| `\\(gpLad[A-Za-z]+)` \|", (results / "macros.md").read_text(), re.M))
    assert tex and tex == md


def test_per_length_has_one_row_per_check(written, arms):
    _, results, _ = written
    with open(results / "per_length.csv") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == lp.PER_LENGTH_COLUMNS
    got = sorted((int(r["seed"]), int(r["arm"]), int(r["digits"]), int(r["round"])) for r in rows)
    want = sorted((s, a, c["digits"], c["round"]) for (s, a), arm in arms.items() for c in of(arm.ev, "check"))
    assert got == want


def test_writes_only_the_named_files(written):
    paper, results, paths = written
    on_disk = {p.relative_to(paper).as_posix() for p in paper.rglob("*") if p.is_file()}
    assert on_disk == set(lp.PAPER_FILES)
    assert all(re.match(r"(numbers|table|figure)_ladder", n) for n in on_disk)
    on_disk = {p.relative_to(results).as_posix() for p in results.rglob("*") if p.is_file()}
    assert on_disk == set(lp.RESULT_FILES)
    assert len(paths) == len(lp.PAPER_FILES) + len(lp.RESULT_FILES)


def test_tex_files_are_marked_and_wrapped(written):
    paper, _, _ = written
    for p in paper.glob("*.tex"):
        lines = p.read_text().splitlines()
        assert lines[0] == lp.NOTE
        assert max(map(len, lines)) <= 80, p.name


def test_lengths_table_has_32_rows(written):
    paper, _, _ = written
    body = (paper / "table_ladder_lengths.tex").read_text().split(r"\midrule", 1)[1].split(r"\bottomrule")[0]
    rows = [r for r in re.sub(r"\s*\n\s*", " ", body).replace(r"\midrule", "").split(r"\\") if "&" in r]
    assert [int(r.split("&")[0]) for r in rows] == list(range(8, 40))
    assert all(len(r.split("&")) == 13 for r in rows)


def test_redo_inside_a_call_counts_both_attempts(arms):
    """Seed 5, arm 3, round 14 was redone inside its call: one round_done, two attempts run."""
    a3 = arms[5, 3]
    assert a3.updates_run() - a3.logged_updates() == 8_400
    r14 = [t for t in a3.attempts() if t["round"] == 14]
    assert [t["attempt"] for t in r14] == [1, 2] and [t["seconds"] for t in r14] == [True, False]


def test_decisions_figure_data(arms):
    """Task 5, check A: per seed, red (built) and blue (not built) circles, and the lowest blue one."""
    pts = lp.decision_points(arms)
    assert [len(pts[s]) for s in lp.SEEDS] == [32, 32, 32]
    assert [sum(p["built"] for p in pts[s]) for s in lp.SEEDS] == [21, 17, 19]
    assert [sum(not p["built"] for p in pts[s]) for s in lp.SEEDS] == [11, 15, 13]
    assert [lp.f1(min(p["direct"] for p in pts[s] if not p["built"])) for s in lp.SEEDS] == ["96.0", "98.5", "98.4"]
