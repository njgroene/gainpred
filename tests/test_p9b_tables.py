"""Section 4's summary and confusion tables, built from the committed results/p9b_cells.csv."""
import re

from gainpred import p9b
from gainpred.writeup import CONFIRMING, load_cells, p9b_tables


def evaluation_cases():
    return [c for c in load_cells() if c.seed in CONFIRMING and c.outcome not in ("unclassified", "missing")]


def counts(tex: str, first: str) -> list[int]:
    line = next(line for line in tex.splitlines() if line.startswith(first + " &"))
    return [int(x) for x in re.findall(r"& (\d+)", line)]


def test_summary_and_confusion_agree_and_match_the_expected_values():
    conf = evaluation_cases()
    summary, confusion = p9b_tables(conf)
    grid = [counts(confusion, r) for r in ("Paid off", "Did nothing", "Did harm")]
    assert grid == [[31, 0, 6], [0, 175, 3], [0, 0, 43]]
    n = len(conf)
    assert sum(map(sum, grid)) == n == 258
    rule = sum(c.outcome in p9b.MATCH[c.verdict] for c in conf)
    assert sum(grid[i][i] for i in range(3)) == rule
    assert f"Our prediction rule & {rule}/{n}" in summary
    for row, v in zip(grid, ("pays off", "does nothing", "does harm")):
        assert f"Always ``{v}'' & {sum(row)}/{n}" in summary
    assert f"for the {n} evaluation cases. Every error is a prediction of harm" in confusion
    assert r"\cmidrule(l){2-4}" in confusion


def test_tables_are_not_written_when_they_disagree():
    """A case whose outcome falls in no row leaves the grid one short of N: generation fails."""
    from dataclasses import replace

    import pytest

    conf = evaluation_cases()
    with pytest.raises(ValueError):
        p9b_tables([replace(conf[0], outcome="unclassified")] + conf[1:])
