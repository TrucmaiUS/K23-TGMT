# IEEE LaTeX Template for VS Code

This repository contains a minimal IEEE paper template configured for the
`LaTeX Workshop` extension in VS Code.

## Project structure

- `main.tex`: root file used for compilation
- `frontmatter/`: title, author list, abstract, and keywords
- `sections/`: main paper sections
- `appendix/`: appendix files
- `ref/`: BibTeX database
- `figures/`: images and plots
- `.vscode/settings.json`: LaTeX Workshop build recipe

## Files you will edit most often

- `frontmatter/title.tex`
- `frontmatter/abstract.tex`
- `frontmatter/keywords.tex`
- `sections/introduction.tex`
- `sections/related_work.tex`
- `sections/proposed_method.tex`
- `sections/experimental_setup.tex`
- `sections/results_discussion.tex`
- `sections/conclusion.tex`
- `ref/refs.bib`

## How to use

1. Open this folder in VS Code.
2. Install the recommended extension `James-Yu LaTeX Workshop` if VS Code prompts you.
3. Open `main.tex`.
4. Save the file or run the recipe `pdflatex -> bibtex -> pdflatex x2` from LaTeX Workshop.

The generated PDF file is `main.pdf`.

## Notes

- The current template uses `\documentclass[conference]{IEEEtran}`.
- If you need an IEEE journal format, change it to
  `\documentclass[journal]{IEEEtran}` in `main.tex`.
- Put your images into `figures/` and include them with `\includegraphics`.
- This project does not depend on `latexmk` or Perl.
