# Build the paper. `make` uses tectonic (one binary, fetches packages itself);
# `make latexmk` uses a full TeX installation. `make generated` rebuilds the
# tables and figures in generated/ from results/ (never edit those by hand).

.PHONY: all tectonic latexmk generated wrap watch clean

all: tectonic

tectonic:
	tectonic -X compile main.tex

latexmk:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

generated:
	cd .. && uv run python -m gainpred.writeup

# Rebuild the PDF whenever a source changes (Ctrl-C to stop). Watches main.tex,
# refs.bib, sections/ and generated/ only, so writing main.pdf does not retrigger.
# Needs watchexec (brew install watchexec). Open main.pdf in a viewer that
# reloads changed files (Skim; VS Code's PDF view) to see each build.
watch:
	watchexec --watch main.tex --watch refs.bib --watch sections --watch generated \
		--exts tex,bib,pdf --debounce 500ms --clear --notify -- $(MAKE) tectonic

# Break source lines longer than 80 characters (a line break is a space to TeX,
# so the PDF does not change). Generated files are wrapped when written.
wrap:
	cd .. && uv run python -m gainpred.texwrap paper/main.tex paper/sections/*.tex paper/refs.bib

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.fls main.fdb_latexmk
