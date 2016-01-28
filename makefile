all: html pdf

pdf:
	pandoc acknowledgements.md -o tex/acknowledgements.tex && \
	pandoc metadata.md introduction.md \
		graphics/introduction.md \
		graphics/related.md \
		graphics/model.md \
		graphics/experiments.md \
		graphics/discussion.md \
		computation/introduction.md \
		computation/related.md \
		computation/model.md \
		computation/experiments.md \
		computation/conclusion.md \
		references.md -s \
		--filter pandoc-crossref \
		--filter pandoc-citeproc \
		--toc \
		--chapter \
		--template template.tex \
		-o tex/rendered.tex && \
	cd tex && \
	pdflatex \
		-output-directory ../output \
		-interaction nonstopmode \
		-file-line-error rendered.tex

html:
	pandoc metadata.md acknowledgements.md introduction.md \
		graphics/introduction.md \
		graphics/related.md \
		graphics/model.md \
		computation/introduction.md \
		computation/related.md \
		computation/model.md \
		computation/experiments.md \
		computation/results.md \
		computation/conclusion.md \
		references.md -s \
		--filter pandoc-crossref \
		--filter pandoc-citeproc \
		--toc \
		--number-sections \
		--chapter \
		--to html5 \
		-o output/rendered.html --mathjax
	# pandoc thesis\ proposal.md --to html5 -s --filter pandoc-crossref --filter pandoc-citeproc -o output/rendered.html --mathjax

watch:
	reload -b -s output/rendered.html & make html & fswatch -o *.md */*.md | xargs -n1 -I% make html
