"""Wrap LaTeX and BibTeX source lines so they read easily in an editor.

Usage::

    uv run python -m gainpred.texwrap paper/main.tex paper/sections/*.tex paper/refs.bib

Why this does not change the compiled paper
===========================================
TeX reads a single line break exactly like a space, and ignores spaces at the
start of a line. So breaking a long line at one of its spaces, and indenting
the continuation, gives the same output. Only an *empty* line means something
new (a paragraph break), and wrapping never makes one. BibTeX also treats line
breaks inside a field as spaces.

What is never broken
====================
* A space right after a backslash (``Shin et al.\\ rely``): ``\\ `` is TeX's
  "control space", so it stays on one line with the next word.
* A line with a comment after some text (an unescaped ``%`` mid-line): the
  comment would swallow what follows, so such lines are left as they are.
* A word longer than the width: it stays whole, on a line of its own.

Comment lines (starting with ``%``) are wrapped as comments: every
continuation starts with ``%`` again.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

WIDTH = 80
# Spaces we may break at: a single space, not a control space. Runs of spaces
# (alignment, as in ``title   = {``) stay where they are.
_BREAK = re.compile(r"(?<![\\ ]) (?! )")
_MID_COMMENT = re.compile(r"(?<!\\)%")


def wrap_line(line: str, width: int = WIDTH) -> list[str]:
    """One source line as one or more lines of at most ``width`` characters where possible."""
    if len(line) <= width:
        return [line]
    indent = line[: len(line) - len(line.lstrip())]
    body = line.strip()
    if body.startswith("%"):
        body = body.lstrip("%").strip()
        first = indent + "% "
        cont = indent + "% "
    else:
        if _MID_COMMENT.search(body):
            return [line]
        first = indent
        # BibTeX field (``key = {...``): continue under the opening brace's content.
        m = re.match(r"\w+\s*=\s*[{\"]", body)
        cont = indent + (" " * m.end() if m else "  ")
    if " & " in body:
        # A table row: break between cells (before the ``&``), and inside a
        # cell only when the cell alone is too long.
        cells = body.split(" & ")
        words = []
        for i, c in enumerate(cells):
            unit = c if i == 0 else "& " + c
            words += [unit] if len(cont) + len(unit) <= width else [w for w in _BREAK.split(unit) if w]
    else:
        words = [w for w in _BREAK.split(body) if w]
    out, cur = [], first + words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = cont + w
        else:
            cur += " " + w
    out.append(cur)
    return out


def wrap_text(text: str, width: int = WIDTH) -> str:
    lines = []
    for line in text.split("\n"):
        lines.extend(wrap_line(line, width))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--width", type=int, default=WIDTH)
    args = parser.parse_args()
    for f in args.files:
        old = f.read_text()
        new = wrap_text(old, args.width)
        if new != old:
            f.write_text(new)
            print(f"wrapped {f}")


if __name__ == "__main__":
    main()
