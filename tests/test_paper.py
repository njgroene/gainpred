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
