"""Render publication materials from the saved run 015 metrics; no inference is run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

# Use a writable cache without changing the user's Matplotlib configuration.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "negotiation-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("clef", "qwen3-8-flash")
NAMES = {"clef": "Clef", "qwen3-8-flash": "Qwen3.8 Flash"}
COLORS = {"clef": "#0072B2", "qwen3-8-flash": "#D55E00"}
COLLEGIA = ("hiring", "negotiation", "ownership")
COLLEGIUM_LABELS = ("Choosing an employer", "Choosing a negotiator", "Entrusting property")


def one(items, **criteria):
    matches = [x for x in items if all(x.get(k) == v for k, v in criteria.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one record for {criteria}; found {len(matches)}")
    return matches[0]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_materials(run):
    metrics = json.loads((run / "metrics/metrics.json").read_text(encoding="utf-8"))
    if metrics["run_id"] != "015" or metrics["dataset"] != "main":
        raise ValueError("These publication materials require main run 015")
    summary = read_csv(run / "metrics/summary.csv")
    by_collegium = read_csv(run / "metrics/by_collegium.csv")
    cases = read_csv(run / "metrics/per_bout_collegium.csv")
    intervals = metrics["inference"]["intervals"]
    data = {}
    for model in MODELS:
        model_cases = [r for r in cases if r["model_alias"] == model]
        if len(model_cases) != 75 or any(r["original_status"] != "complete" for r in model_cases):
            raise ValueError("Expected 75 complete original bout-collegium cases per model")
        entries = {}
        for metric in ("agreement_rate", "cohens_kappa", "pairwise_repeatability", "label_flip_rate"):
            interval = one(intervals, interval_id=f"{model}:{metric}")
            point, ci = interval["point_estimate"], interval["confidence_interval"]
            if point["status"] != "ok" or ci["status"] != "ok" or ci["confidence_level"] != .95:
                raise ValueError(f"Unavailable estimate or 95% CI: {model}:{metric}")
            entries[metric] = {"value": point["value"], "lower": ci["lower"], "upper": ci["upper"]}
        for metric in ("macro_f1", "individual_judge_majority_to_individual"):
            row = one(summary, model_alias=model, scope="individual", metric=metric)
            if row["status"] != "ok":
                raise ValueError(f"Unavailable metric: {model}:{metric}")
            entries[metric] = float(row["value"])
        # Count existing agreement flags, rather than reconstructing winners or metrics.
        entries["matches"] = sum(r["original_agreement"] == "True" for r in model_cases)
        entries["n_cases"] = len(model_cases)
        if abs(entries["matches"] / len(model_cases) - entries["agreement_rate"]["value"]) > 1e-12:
            raise ValueError("Saved case flags disagree with saved accuracy")
        entries["by_collegium"] = {}
        for collegium in COLLEGIA:
            row = one(by_collegium, model_alias=model, scope="individual", collegium=collegium,
                      metric=f"agreement_rate_{collegium}")
            selected = [r for r in model_cases if r["collegium"] == collegium]
            n = int(row["N_cases"])
            matches = sum(r["original_agreement"] == "True" for r in selected)
            value = float(row["value"])
            if n != 25 or len(selected) != n or abs(matches / n - value) > 1e-12:
                raise ValueError("Saved collegium counts disagree with saved accuracy")
            entries["by_collegium"][collegium] = {"value": value, "matches": matches, "n": n}
        entries["h1"] = {
            metric: one(metrics["inference"]["tests"], test_id=f"H1:{model}:{metric}")["adjusted_p_value"]
            for metric in ("agreement_rate", "cohens_kappa")
        }
        data[model] = entries
    tests = {
        metric: one(metrics["inference"]["tests"], test_id=f"H3:{metric}_difference")
        for metric in ("pairwise_repeatability", "label_flip_rate")
    }
    return metrics, data, tests


def pct(value):
    return f"{100 * value:.2f}"


def write_table(data, output, metrics):
    rows = []
    def row(label, values):
        rows.append(label + " & " + " & ".join(values) + r" \\")
    def vals(fn):
        return [fn(data[m]) for m in MODELS]
    row(r"Agreement Rate (Accuracy), \%", vals(lambda d: pct(d["agreement_rate"]["value"])))
    row("Matches / bout-collegium cases", vals(lambda d: f"{d['matches']} / {d['n_cases']}"))
    row(r"Accuracy: 95\% CI, \%", vals(lambda d: f"[{pct(d['agreement_rate']['lower'])}, {pct(d['agreement_rate']['upper'])}]"))
    row(r"Cohen's $\kappa$", vals(lambda d: f"{d['cohens_kappa']['value']:.4f}"))
    row(r"$\kappa$: 95\% CI", vals(lambda d: f"[{d['cohens_kappa']['lower']:.4f}, {d['cohens_kappa']['upper']:.4f}]"))
    row("Macro-F1", vals(lambda d: f"{d['macro_f1']:.4f}"))
    row(r"Majority-to-individual Agreement, \%", vals(lambda d: pct(d["individual_judge_majority_to_individual"])))
    row(r"H1: Accuracy $> 0.5$, Holm-adjusted $p$", vals(lambda d: f"{d['h1']['agreement_rate']:.4f}"))
    row(r"H1: $\kappa > 0$, Holm-adjusted $p$", vals(lambda d: f"{d['h1']['cohens_kappa']:.4f}"))
    bootstrap = metrics["bootstrap"]
    text = r"""% Generated by src/visualization/build_paper_figures.py; do not edit values.
% Requires \usepackage{booktabs}. Table number follows the host document counter.
\begin{table}[tbp]
\centering
\caption{Agreement with human judges in the main experiment (run 015).}
\label{tab:main-agreement}
\small
\begin{tabular}{@{}lrr@{}}
\toprule
Metric & Clef & Qwen3.8 Flash \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\begin{minipage}{\linewidth}
\footnotesize
\textit{Note.} Original orientation; 25 bouts and three collegia yield 75
bout-collegium cases per model. Accuracy compares model and human majority
decisions within each case; it is not overall bout-winner accuracy.
Majority-to-individual Agreement compares each model majority with the three
individual human votes (225 comparisons per model).
""" + f"The 95\\% percentile bootstrap intervals use {bootstrap['replications']:,} resamples " + r"""clustered
by bout; with only 25 independent bouts, intervals may be unstable.
H1 tests Accuracy above 0.5 and Cohen's $\kappa$ above zero; reported $p$-values
use the saved Holm correction across the seven prespecified tests.
All values and intervals are taken from the saved run 015 outputs.
\end{minipage}
\end{table}
"""
    (output / "table2_agreement.tex").write_text(text, encoding="utf-8")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def figure1(data, output):
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    fig.subplots_adjust(left=.10, right=.98, bottom=.21, top=.82)
    width = .32
    for i, model in enumerate(MODELS):
        values = [data[model]["by_collegium"][c] for c in COLLEGIA]
        bars = ax.bar([x + (i - .5) * width for x in range(3)],
                      [v["value"] * 100 for v in values], width,
                      color=COLORS[model], label=NAMES[model], edgecolor="white",
                      hatch="" if i == 0 else "///", linewidth=.7, zorder=3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{v['value'] * 100:.0f}%\n({v['matches']}/{v['n']})",
                    ha="center", va="bottom", fontsize=9)
    ax.set(ylim=(0, 100), ylabel="Agreement Rate (Accuracy)", xticks=range(3),
           xticklabels=COLLEGIUM_LABELS, yticks=range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="y", color="#dddddd", linewidth=.6, zorder=0)
    fig.suptitle("Accuracy by judging collegium", y=.98, fontsize=13, fontweight="bold")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.54, .92),
               ncol=2, frameon=False)
    fig.text(.10, .055, "Original orientation; 25 bouts per collegium. Labels show matches / 25.", fontsize=9)
    fig.savefig(output / "figure1_accuracy_by_collegium.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def figure2(data, tests, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.30, top=.73, wspace=.28)
    settings = (("pairwise_repeatability", "(a) Pairwise Repeatability", "Original orientation"),
                ("label_flip_rate", "(b) Label Flip Rate", "After swapping player labels"))
    for ax, (metric, title, subtitle) in zip(axes, settings):
        for x, model in enumerate(MODELS):
            v = data[model][metric]
            lo, point, hi = (100 * v[k] for k in ("lower", "value", "upper"))
            ax.errorbar(x, point, yerr=[[point - lo], [hi - point]], fmt="o" if x == 0 else "s",
                        color=COLORS[model], markersize=7, capsize=5, elinewidth=1.5, zorder=3)
            ax.text(x, hi + 3, f"{point:.2f}%\n[{lo:.2f}, {hi:.2f}]", ha="center", va="bottom", fontsize=9)
        ax.set(xlim=(-.6, 1.6), ylim=(0, 118), xticks=(0, 1),
               xticklabels=[NAMES[m] for m in MODELS], yticks=range(0, 101, 20))
        ax.set_title(title + "\n" + subtitle, fontsize=10, pad=16)
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.grid(axis="y", color="#dddddd", linewidth=.6)
        test = tests[metric]
        significant = test["adjusted_p_value"] < test["alpha"]
        ax.text(.5, -.19, f"Holm-adjusted p = {test['adjusted_p_value']:.4f}\n"
                + ("Significant difference" if significant else "No significant difference"),
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
    axes[0].set_ylabel("Rate (%)")
    fig.suptitle("Repeatability and sensitivity to player labels", y=.97, fontsize=13, fontweight="bold")
    fig.text(.10, .045, "Points and whiskers: saved estimates and 95% bout-cluster bootstrap CIs.\n"
             "25 bouts; three collegia. Holm correction across seven prespecified tests.", fontsize=8.5)
    fig.savefig(output / "figure2_repeatability_label_flip.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/main/015")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    metrics, data, tests = load_materials(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    write_table(data, args.output_dir, metrics)
    figure1(data, args.output_dir)
    figure2(data, tests, args.output_dir)
    sources = [args.run_dir / "metrics" / name for name in
               ("metrics.json", "summary.csv", "by_collegium.csv", "per_bout_collegium.csv")]
    manifest = {"run_id": metrics["run_id"], "calculation_rules_version": metrics["calculation_rules_version"],
                "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "display_data": data, "h3_tests": tests}
    (args.output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created Table 2 and Figures 1-2 in {args.output_dir}")


if __name__ == "__main__":
    main()
