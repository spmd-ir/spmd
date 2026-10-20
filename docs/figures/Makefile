# SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Renders every figure here to a cropped PDF and a 300 dpi PNG. Needs pdflatex
# with TikZ and the preview package, and pdftocairo (poppler). The PDF and PNG
# are committed, so reading the figures needs neither. build/ is removed after
# a successful run and kept, with the LaTeX log, after a failed one.

FIGURES := $(basename $(wildcard *.tex))

all: $(FIGURES:=.pdf) $(FIGURES:=.png)

%.pdf: %.tex
	@mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build $<
	cp build/$@ $@
	rm -rf build

%.png: %.pdf
	pdftocairo -png -r 300 -singlefile $< $*

clean:
	rm -rf build

.PHONY: all clean
