# Materials for the article

- `table2_agreement.tex`: agreement metrics, counts, confidence intervals, and
  Holm-adjusted H1 p-values.
- `figure1_accuracy_by_collegium.pdf`: grouped Accuracy bars, with match counts.
- `figure2_repeatability_label_flip.pdf`: point estimates, 95% intervals, and
  separate Holm-adjusted tests of the two stability differences.
- `source_manifest.json`: source hashes and the exact values used for display.

Regenerate from the repository root:

```powershell
python -m pip install -r src/visualization/requirements.txt
python src/visualization/build_paper_figures.py
```

## LaTeX integration

The table needs `booktabs`; figures need `graphicx`:

```latex
\usepackage{booktabs}
\usepackage{graphicx}
% At the desired location in the paper:
\input{figures/table2_agreement.tex}
\input{figures/figure_captions.tex}
```

Paths above assume compilation from `paper/`. Table and figure numbers follow
the host document's counters; insert the agreement table after Table 1 to obtain
Table 2. Both PDFs have a width of 7.2 inches and are intended for a full-width
figure in a two-column paper (use `figure*` there). Avoid shrinking them to a
single narrow column. All text is English. Match counts come from stored case
flags; all metrics, intervals, and p-values come from saved CSV/JSON outputs.

The LaTeX source uses standard table commands and `booktabs`; a TeX compiler is
not bundled with the visualization environment.
