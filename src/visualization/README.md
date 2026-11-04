# Publication materials for main run 015

From the repository root, with Python 3.11 or later:

```powershell
python -m pip install -r src/visualization/requirements.txt
python src/visualization/build_paper_figures.py
```

The default source is `results/main/015/`; the default destination is
`paper/figures/`. Optional `--run-dir` and `--output-dir` arguments accept paths.
The script requires main run 015 and validates the expected complete case counts.
It does not import the experiment or metrics calculation pipelines, call APIs,
resample data, run statistical tests, or change saved experiment outputs.

Inputs (all under `results/main/015/metrics/`):

- `metrics.json`: saved inference point estimates, 95% intervals, H1/H3 adjusted
  p-values, alpha, and bootstrap settings.
- `summary.csv`: saved original-orientation Macro-F1 and
  Majority-to-individual Agreement.
- `by_collegium.csv`: saved Accuracy by collegium, scope `individual`.
- `per_bout_collegium.csv`: counts of saved `original_agreement=True` flags for
  integer match labels, cross-checked against the saved Accuracy. Winners and
  agreement flags are never recalculated.

Both model scopes contain the same 25 bouts (75 bout-collegium cases). Counts
and percentages describe majority agreement within collegia, not the overall
winner of a bout. Majority-to-individual Agreement has 225 comparisons per model.
Pairwise Repeatability refers to the three original-orientation votes per case.
Label Flip Rate uses the stored comparison after mapping swapped player labels
back to participant identities. Intervals are the saved 10,000-resample percentile
bootstrap intervals clustered by bout. No new confidence intervals are computed.

Figure 2 assesses each H3 component using its own saved adjusted p-value and alpha,
not the joint H3 decision. A confirmed joint H3 does not imply a significant
Label Flip Rate difference. Clef's saved repeatability CI collapses to a point at
100%; this is a property of the saved sample bootstrap, not proof of population
certainty. All intervals are based on only 25 independent bouts.

The script writes two vector PDFs with embedded TrueType fonts, a LaTeX table,
and `source_manifest.json` containing input SHA-256 hashes and full-precision
display values. Only formatting (percent scaling and decimal rounding) is applied.
Clef and Qwen use consistent colors, with hatching/marker shapes for grayscale use.

PDF visual QA may use PyMuPDF, separately from production dependencies:

```powershell
python -m pip install pymupdf
```
