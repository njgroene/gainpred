#!/bin/sh
# Build main.pdf and supplement.pdf (each cites the other's labels).
set -e
pdflatex -interaction=nonstopmode -halt-on-error main > build.log
bibtex main >> build.log
pdflatex -interaction=nonstopmode -halt-on-error supplement >> build.log
bibtex supplement >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main >> build.log
pdflatex -interaction=nonstopmode -halt-on-error supplement >> build.log
pdflatex -interaction=nonstopmode -halt-on-error supplement >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main >> build.log
pdflatex -interaction=nonstopmode -halt-on-error main > main.last.log
