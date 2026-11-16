# Build the paper. `make` uses tectonic (one binary, fetches packages itself);
# `make latexmk` uses a full TeX installation. `make generated` rebuilds the
# tables and figures in generated/ from results/ (never edit those by hand).

.PHONY: all tectonic latexmk generated wrap clean

all: tectonic

tectonic:
	tectonic -X compile main.tex

latexmk:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

generated:
	cd .. && uv run python -m gainpred.writeup

# Break source lines longer than 80 characters (a line break is a space to TeX,
# so the PDF does not change). Generated files are wrapped when written.
wrap:
	cd .. && uv run python -m gainpred.texwrap paper/main.tex paper/sections/*.tex paper/refs.bib

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.fls main.fdb_latexmk
