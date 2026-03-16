TEX = primo
BIB = ../shared/primo.bib

.PHONY: all clean

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex $(BIB)
	pdflatex -interaction=nonstopmode $(TEX)
	bibtex $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).pdf
