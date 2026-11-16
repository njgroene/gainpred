"""The paper builds (paper/, tectonic). Skipped where tectonic is not installed; CI builds it too.

A broken \\input, a missing generated file or a LaTeX error in a generated table fails here.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"


@pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not installed")
def test_paper_builds(tmp_path):
    work = tmp_path / "paper"
    shutil.copytree(PAPER, work, ignore=shutil.ignore_patterns("main.pdf", "*.aux", "*.log"))
    result = subprocess.run(["tectonic", "-X", "compile", "main.tex"], cwd=work, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    assert (work / "main.pdf").stat().st_size > 10_000


def test_sections_only_input_generated_files_that_exist():
    for section in sorted((PAPER / "sections").glob("*.tex")):
        for line in section.read_text().splitlines():
            if line.startswith("\\input{generated/"):
                name = line[len("\\input{"):-1]
                assert (PAPER / f"{name}.tex").exists(), f"{section.name}: {name}.tex missing (run gainpred.writeup)"


def test_wrapping_never_changes_what_tex_reads():
    """Wrapping only turns single spaces into line breaks plus indentation: same words, no blank lines."""
    from gainpred.texwrap import wrap_line

    line = r"Shin et al.\ rely on rules; " + "word " * 40 + r"96.5\% of cases \\"
    out = wrap_line(line)
    assert all(len(o) <= 80 for o in out) and all(o.strip() for o in out)
    assert " ".join(o.strip() for o in out) == line.strip()
    assert not any(o.endswith("\\") and not o.endswith("\\\\") for o in out)  # control space kept whole
    row = " & ".join(["12.3 $\\rightarrow$ 45.6"] * 8) + r" \\"
    assert all(o.lstrip().startswith("&") for o in wrap_line(row)[1:])  # table rows break between cells
    assert wrap_line("x " * 50 + "% comment") == ["x " * 50 + "% comment"]  # mid-line comment: untouched


def test_generated_files_are_wrapped():
    for f in sorted((PAPER / "generated").glob("*.tex")):
        long = [n for n, line in enumerate(f.read_text().splitlines(), 1) if len(line) > 80]
        assert not long, f"{f.name}: lines {long} over 80 characters (run gainpred.writeup)"


def test_fill_carries_overflow_without_crossing_paragraphs_or_lists():
    from gainpred.texwrap import wrap_text

    text = "  " + "alpha " * 15 + "\n  beta gamma\n  1/ first item\n\n  next paragraph"
    out = wrap_text(text)
    assert all(len(line) <= 80 for line in out.splitlines())
    assert out.split() == text.split()  # same words, same order
    assert "\n  1/ first item\n\n  next paragraph" in out  # list line and paragraph break kept
    assert all(line.startswith("  ") for line in out.splitlines() if line)  # indent kept, not grown
