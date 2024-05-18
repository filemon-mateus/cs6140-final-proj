LATEXMK := latexmk -pdflua

.PRECIOUS: %.pdf

final: final.tex
	$(LATEXMK) -pvc $<

clean:
	$(LATEXMK) -c

distclean:
	$(LATEXMK) -C

.PHONY: final clean distclean
