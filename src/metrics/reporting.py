"""Deterministic report rendering and conservative publication.

All artifacts are prepared and checked in a temporary directory next to the
run.  Existing reports are only touched after every new artifact is valid.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from src.metrics import (
    CALCULATION_RULES_VERSION,
    PERCENTILE_BOUNDS,
    PERCENTILE_METHOD,
)
from src.metrics.aggregation import AggregatedAnalysis, AggregatedCase, VoteTally
from src.metrics.bootstrap import BootstrapAnalysis
from src.metrics.calculations import CalculatedMetrics
from src.metrics.contracts import ConfidenceIntervalResult, MetricResult
from src.metrics.inference import InferenceAnalysis
from src.metrics.loading import AnalysisInputs
from src.metrics.preparation import PreparedAnalysis


MANAGED_REPORT_FILES = (
    "metrics.json",
    "summary.csv",
    "by_collegium.csv",
    "per_bout_collegium.csv",
    "exclusions.csv",
    "report.md",
)

SUMMARY_FIELDS = (
    "model_alias", "scope", "metric", "value", "N_bouts", "N_cases",
    "N_model_votes", "N_human_votes", "N_comparisons", "ci_lower", "ci_upper",
    "hypothesis_id", "test_id",
    "alternative", "raw_p_value", "adjusted_p_value", "decision",
    "status", "reason",
)
BY_COLLEGIUM_FIELDS = (
    "model_alias", "scope", "collegium", "metric", "value", "N_bouts",
    "N_cases", "N_model_votes", "N_human_votes", "N_comparisons", "status",
    "reason",
)
PER_BOUT_FIELDS = (
    "run_id", "dataset", "bout_id", "model_alias", "collegium",
    "original_status", "swapped_status", "human_vote_1", "human_vote_2",
    "human_vote_3", "human_player_1", "human_player_2", "human_majority",
    "original_vote_1", "original_vote_2", "original_vote_3",
    "original_player_1", "original_player_2", "original_majority",
    "original_agreement", "probability_player_1_1", "probability_player_1_2",
    "probability_player_1_3", "swapped_vote_1", "swapped_vote_2",
    "swapped_vote_3", "swapped_player_1", "swapped_player_2",
    "swapped_majority", "position_winner_changed",
    "position_vote_share_change", "exclusion_reasons",
)
EXCLUSION_FIELDS = (
    "bout_id", "model_alias", "variant", "collegium", "metric_scope",
    "metric", "reason_code", "reason", "actual_count", "required_count",
)


class ReportingError(RuntimeError):
    """The complete report set could not be staged, checked, or published."""


def _metric_payload(result: MetricResult) -> dict[str, Any]:
    return result.model_dump(mode="json")


def _interval_payload(result: ConfidenceIntervalResult) -> dict[str, Any]:
    return result.model_dump(mode="json")


def _model_metrics_payload(item: Any) -> dict[str, Any]:
    return {
        "model_alias": item.model_alias,
        "scope_id": item.scope_id,
        "bout_ids": list(item.bout_ids),
        "agreement": {
            "accuracy": _metric_payload(item.agreement.accuracy),
            "by_collegium": {
                name: _metric_payload(value)
                for name, value in item.agreement.by_collegium.items()
            },
            "cohens_kappa": _metric_payload(item.agreement.cohens_kappa),
            "macro_f1": _metric_payload(item.agreement.macro_f1),
        },
        "individual_judge": {
            "majority_to_individual": _metric_payload(
                item.individual_judge.majority_to_individual
            ),
            "all_pairs": _metric_payload(item.individual_judge.all_pairs),
        },
        "repeatability": {
            "unanimity_rate": _metric_payload(item.repeatability.unanimity_rate),
            "pairwise_repeatability": _metric_payload(
                item.repeatability.pairwise_repeatability
            ),
        },
        "probability": {
            name: _metric_payload(getattr(item.probability, name))
            for name in (
                "brier_score", "brier_score_individual", "vote_share_mae",
                "vote_share_mae_individual", "coverage",
            )
        },
    }


def _position_payload(item: Any) -> dict[str, Any]:
    return {
        "model_alias": item.model_alias,
        "scope_id": item.scope_id,
        "bout_ids": list(item.bout_ids),
        "label_flip_rate": _metric_payload(item.metrics.label_flip_rate),
        "vote_share_change": _metric_payload(item.metrics.vote_share_change),
        "by_collegium": {
            collegium: {
                "label_flip_rate": _metric_payload(values[0]),
                "vote_share_change": _metric_payload(values[1]),
            }
            for collegium, values in item.metrics.by_collegium.items()
        },
    }


def _scopes_payload(prepared: PreparedAnalysis) -> dict[str, Any]:
    scopes: dict[str, Any] = {
        scope.scope_id: {
            "kind": scope.kind,
            "model_alias": scope.model_alias,
            "bout_ids": list(scope.bout_ids),
            "N_bouts": scope.n_bouts,
        }
        for scope in (
            *prepared.model_scopes,
            prepared.common_scope,
            *prepared.position_scopes,
            prepared.position_common_scope,
        )
    }
    for scope in prepared.probability_scopes:
        scopes[scope.scope_id] = {
            "kind": "probability",
            "model_alias": scope.model_alias,
            "base_scope_id": scope.base_scope_id,
            "bout_ids": list(scope.bout_ids),
            "cases": [case.model_dump(mode="json") for case in scope.cases],
            "N_bouts": scope.n_bouts,
            "N_cases": scope.n_cases,
        }
    scopes["all_humans"] = {
        "kind": "human",
        "bout_ids": list(prepared.selected_bout_ids),
        "N_bouts": len(prepared.selected_bout_ids),
    }
    return scopes


def build_metrics_payload(
    inputs: AnalysisInputs,
    prepared: PreparedAnalysis,
    aggregated: AggregatedAnalysis,
    calculated: CalculatedMetrics,
    bootstrap: BootstrapAnalysis,
    inference: InferenceAnalysis,
    *,
    calculated_at: datetime,
) -> dict[str, Any]:
    """Build the lossless machine-readable report."""

    comparison = bootstrap.comparison
    interval_settings = (
        bootstrap.agreement_intervals[0].confidence_interval
        if bootstrap.agreement_intervals
        else comparison.confidence_interval
    )
    return {
        "run_id": inputs.metadata.run_id,
        "dataset": inputs.dataset,
        "run": {
            "mode": inputs.metadata.mode,
            "dataset": inputs.metadata.dataset,
            "started_at": inputs.metadata.started_at.isoformat(),
            "selected_bout_ids": list(inputs.metadata.selected_bout_ids),
        },
        "calculated_at_utc": calculated_at.astimezone(timezone.utc).isoformat(),
        "calculation_rules_version": CALCULATION_RULES_VERSION,
        "bootstrap": {
            "method": "paired_cluster_percentile_by_bout",
            "replications": interval_settings.replications,
            "seed": interval_settings.seed,
            "confidence_level": interval_settings.confidence_level,
            "percentile_bounds": list(PERCENTILE_BOUNDS),
            "percentile_method": PERCENTILE_METHOD,
        },
        "models": [model.model_dump(mode="json") for model in inputs.metadata.models],
        "input_files": [
            {"path": str(item.path), "sha256": item.sha256}
            for item in inputs.input_files
        ],
        "scopes": _scopes_payload(prepared),
        "metrics": {
            "model_scopes": [_model_metrics_payload(item) for item in calculated.model_scopes],
            "position_scopes": [_position_payload(item) for item in calculated.position_scopes],
            "human_scopes": [
                {
                    "scope_id": item.scope_id,
                    "bout_ids": list(item.bout_ids),
                    "pairwise_agreement": _metric_payload(item.metrics.pairwise_agreement),
                    "fleiss_kappa": _metric_payload(item.metrics.fleiss_kappa),
                }
                for item in calculated.human_scopes
            ],
        },
        "confidence_intervals": [
            {
                "model_alias": item.model_alias,
                "scope_id": item.scope_id,
                "bout_ids": list(item.bout_ids),
                "point_estimate": _metric_payload(item.point_estimate),
                "interval": _interval_payload(item.confidence_interval),
            }
            for item in bootstrap.agreement_intervals
        ],
        "model_comparison": {
            "comparison_id": f"{comparison.decision_alias}_minus_{comparison.llm_alias}",
            "decision_alias": comparison.decision_alias,
            "llm_alias": comparison.llm_alias,
            "scope_id": comparison.scope_id,
            "bout_ids": list(comparison.bout_ids),
            "difference_percentage_points": _metric_payload(
                comparison.difference_percentage_points
            ),
            "confidence_interval": _interval_payload(comparison.confidence_interval),
        },
        "inference": {
            "alpha": inference.alpha,
            "multiple_testing_correction": inference.correction,
            "family_size": inference.family_size,
            "exact_permutation_max_bouts": inference.exact_max_bouts,
            "monte_carlo_permutations": inference.monte_carlo_permutations,
            "seed": inference.seed,
            "intervals": [
                {
                    "interval_id": item.interval_id,
                    "metric": item.metric,
                    "scope_id": item.scope_id,
                    "model_alias": item.model_alias,
                    "comparison_id": item.comparison_id,
                    "bout_ids": list(item.bout_ids),
                    "point_estimate": _metric_payload(item.point_estimate),
                    "confidence_interval": _interval_payload(item.confidence_interval),
                }
                for item in inference.intervals
            ],
            "tests": [item.model_dump(mode="json") for item in inference.tests],
            "decisions": [
                item.model_dump(mode="json") for item in inference.decisions
            ],
        },
        "descriptive_bouts": [
            {
                "model_alias": item.model_alias,
                "bout_id": item.bout_id,
                "human_tally": asdict_tally(item.human_tally),
                "model_tally": asdict_tally(item.model_tally),
                "majority_agreement": item.majority_agreement,
            }
            for item in aggregated.bout_summaries
        ],
        "exclusions": [item.model_dump(mode="json") for item in prepared.exclusions],
    }


def asdict_tally(tally: VoteTally) -> dict[str, Any]:
    return {
        "player_1": tally.player_1,
        "player_2": tally.player_2,
        "majority": tally.majority,
    }


def _all_model_results(item: Any) -> tuple[MetricResult, ...]:
    return (
        item.agreement.accuracy,
        item.agreement.cohens_kappa,
        item.agreement.macro_f1,
        item.individual_judge.majority_to_individual,
        item.individual_judge.all_pairs,
        item.repeatability.unanimity_rate,
        item.repeatability.pairwise_repeatability,
        item.probability.brier_score,
        item.probability.brier_score_individual,
        item.probability.vote_share_mae,
        item.probability.vote_share_mae_individual,
        item.probability.coverage,
    )


def _summary_rows(
    calculated: CalculatedMetrics,
    bootstrap: BootstrapAnalysis,
    inference: InferenceAnalysis | None = None,
) -> list[dict[str, Any]]:
    intervals = {
        (item.model_alias, item.scope_id): item.confidence_interval
        for item in bootstrap.agreement_intervals
    }
    rows: list[dict[str, Any]] = []
    for item in calculated.model_scopes:
        interval = intervals[(item.model_alias, item.scope_id)]
        for result in _all_model_results(item):
            rows.append(_result_row(
                item.model_alias, _csv_scope(item.scope_id), result,
                interval if result.metric == "agreement_rate" else None,
            ))
    for item in calculated.position_scopes:
        for result in (item.metrics.label_flip_rate, item.metrics.vote_share_change):
            rows.append(_result_row(item.model_alias, _csv_scope(item.scope_id), result))
    for item in calculated.human_scopes:
        for result in (item.metrics.pairwise_agreement, item.metrics.fleiss_kappa):
            rows.append(_result_row("human", item.scope_id, result))
    comparison = bootstrap.comparison
    rows.append(_result_row(
        f"{comparison.decision_alias}_minus_{comparison.llm_alias}",
        comparison.scope_id,
        comparison.difference_percentage_points,
        comparison.confidence_interval,
    ))
    if inference is not None:
        intervals = {item.interval_id: item.confidence_interval for item in inference.intervals}
        decisions = {
            test_id: decision.status
            for decision in inference.decisions
            for test_id in decision.test_ids
        }
        for test in inference.tests:
            if test.hypothesis_id == "H1":
                interval_id = test.test_id.removeprefix("H1:")
            elif test.test_id == "H2:accuracy_difference":
                comparison = next(
                    item.comparison_id for item in inference.intervals
                    if item.comparison_id is not None
                )
                interval_id = f"{comparison}:accuracy_difference"
            elif test.test_id == "H3:pairwise_repeatability_difference":
                comparison = next(
                    item.comparison_id for item in inference.intervals
                    if item.comparison_id is not None
                )
                interval_id = f"{comparison}:pairwise_repeatability_difference"
            else:
                comparison = next(
                    item.comparison_id for item in inference.intervals
                    if item.comparison_id is not None
                )
                interval_id = f"{comparison}:label_flip_rate_difference"
            interval = intervals.get(interval_id)
            alias = (
                test.scope_id.removeprefix("model:")
                if test.hypothesis_id == "H1"
                else comparison
            )
            rows.append({
                "model_alias": alias,
                "scope": test.scope_id,
                "metric": test.metric,
                "value": test.effect,
                "N_bouts": test.n_bouts,
                "N_cases": test.n_cases,
                "N_model_votes": None,
                "N_human_votes": None,
                "N_comparisons": None,
                "ci_lower": interval.lower if interval is not None else None,
                "ci_upper": interval.upper if interval is not None else None,
                "hypothesis_id": test.hypothesis_id,
                "test_id": test.test_id,
                "alternative": test.alternative,
                "raw_p_value": test.raw_p_value,
                "adjusted_p_value": test.adjusted_p_value,
                "decision": decisions.get(test.test_id),
                "status": test.status,
                "reason": test.reason,
            })
    return rows


def _csv_scope(scope_id: str) -> str:
    if scope_id.startswith("model:"):
        return "individual"
    if scope_id.startswith("common:"):
        return "common"
    if scope_id.startswith("position:common:"):
        return "position_common"
    if scope_id.startswith("position:"):
        return "position"
    return scope_id


def _result_row(
    model_alias: str,
    scope: str,
    result: MetricResult,
    interval: ConfidenceIntervalResult | None = None,
) -> dict[str, Any]:
    return {
        "model_alias": model_alias,
        "scope": scope,
        "metric": result.metric,
        "value": result.value,
        "N_bouts": result.n_bouts,
        "N_cases": result.n_cases,
        "N_model_votes": result.n_model_votes,
        "N_human_votes": result.n_human_votes,
        "N_comparisons": result.n_comparisons,
        "ci_lower": interval.lower if interval is not None else None,
        "ci_upper": interval.upper if interval is not None else None,
        "hypothesis_id": None,
        "test_id": None,
        "alternative": None,
        "raw_p_value": None,
        "adjusted_p_value": None,
        "decision": None,
        "status": result.status,
        "reason": result.reason or (interval.reason if interval is not None else None),
    }


def _by_collegium_rows(calculated: CalculatedMetrics) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in calculated.model_scopes:
        for collegium, result in item.agreement.by_collegium.items():
            rows.append(_collegium_row(item.model_alias, _csv_scope(item.scope_id), collegium, result))
    for item in calculated.position_scopes:
        for collegium, results in item.metrics.by_collegium.items():
            for result in results:
                rows.append(_collegium_row(item.model_alias, _csv_scope(item.scope_id), collegium, result))
    return rows


def _collegium_row(model_alias: str, scope: str, collegium: str, result: MetricResult) -> dict[str, Any]:
    return {
        "model_alias": model_alias, "scope": scope, "collegium": collegium,
        "metric": result.metric, "value": result.value, "N_bouts": result.n_bouts,
        "N_cases": result.n_cases,
        "N_model_votes": result.n_model_votes,
        "N_human_votes": result.n_human_votes,
        "N_comparisons": result.n_comparisons,
        "status": result.status, "reason": result.reason,
    }


def _votes(values: Sequence[Any]) -> list[str | None]:
    winners = [value.winner for value in sorted(values, key=lambda vote: vote.ordinal)]
    return winners + [None] * (3 - len(winners))


def _probabilities(case: AggregatedCase, variant: str) -> list[float | None]:
    values = case.original_probabilities if variant == "original" else case.swapped_probabilities
    return list(values) + [None] * (3 - len(values))


def _tally_columns(prefix: str, tally: VoteTally) -> dict[str, Any]:
    return {
        f"{prefix}_player_1": tally.player_1,
        f"{prefix}_player_2": tally.player_2,
        f"{prefix}_majority": tally.majority,
    }


def _per_bout_rows(
    inputs: AnalysisInputs,
    prepared: PreparedAnalysis,
    aggregated: AggregatedAnalysis,
) -> list[dict[str, Any]]:
    exclusions: dict[tuple[str | None, str], list[str]] = {}
    for item in prepared.exclusions:
        exclusions.setdefault((item.model_alias, item.bout_id), []).append(
            f"{item.scope_id}: {item.reason}"
        )
    rows = []
    for case in aggregated.case_diagnostics:
        human = _votes(case.human_votes)
        original = _votes(case.original_votes)
        swapped = _votes(case.swapped_votes)
        probabilities = _probabilities(case, "original")
        row: dict[str, Any] = {
            "run_id": inputs.metadata.run_id, "dataset": inputs.dataset,
            "bout_id": case.bout_id, "model_alias": case.model_alias,
            "collegium": case.collegium,
            "original_status": "complete" if case.original_complete else "incomplete",
            "swapped_status": "complete" if case.swapped_complete else "incomplete",
            **{f"human_vote_{i}": value for i, value in enumerate(human, 1)},
            **_tally_columns("human", case.human_tally),
            **{f"original_vote_{i}": value for i, value in enumerate(original, 1)},
            **_tally_columns("original", case.original_tally),
            "original_agreement": case.original_agreement,
            **{f"probability_player_1_{i}": value for i, value in enumerate(probabilities, 1)},
            **{f"swapped_vote_{i}": value for i, value in enumerate(swapped, 1)},
            **_tally_columns("swapped", case.swapped_tally),
            "position_winner_changed": case.position_winner_changed,
            "position_vote_share_change": case.position_vote_share_change,
            "exclusion_reasons": " | ".join(dict.fromkeys(exclusions.get((case.model_alias, case.bout_id), ()))),
        }
        rows.append(row)
    return rows


def _exclusion_rows(prepared: PreparedAnalysis) -> list[dict[str, Any]]:
    return [
        {
            "bout_id": item.bout_id, "model_alias": item.model_alias,
            "variant": item.variant, "collegium": item.collegium,
            "metric_scope": item.scope_id, "metric": item.metric,
            "reason_code": item.reason_code, "reason": item.reason,
            "actual_count": item.actual_count, "required_count": item.required_count,
        }
        for item in prepared.exclusions
    ]


def _display(result: MetricResult, *, percent: bool = False) -> str:
    if result.status != "ok":
        return f"N/A ({result.reason})"
    assert result.value is not None
    return f"{100 * result.value:.2f}%" if percent else f"{result.value:.4f}"


def _report_markdown(
    inputs: AnalysisInputs,
    prepared: PreparedAnalysis,
    aggregated: AggregatedAnalysis,
    calculated: CalculatedMetrics,
    bootstrap: BootstrapAnalysis,
    inference: InferenceAnalysis,
    calculated_at: datetime,
) -> str:
    lines = [
        "# Отчёт по метрикам", "",
        f"- Запуск: `{inputs.metadata.run_id}`",
        f"- Выборка: `{inputs.dataset}`",
        f"- Рассчитано (UTC): `{calculated_at.astimezone(timezone.utc).isoformat()}`",
        f"- Версия алгоритма: `{CALCULATION_RULES_VERSION}`",
        f"- Выбрано поединков: {len(prepared.selected_bout_ids)}", "",
        "## Основные результаты", "",
        "| Модель | Скоуп | Accuracy | 95% bootstrap CI | Cohen's κ | Macro-F1 | N поединков |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    intervals = {(x.model_alias, x.scope_id): x.confidence_interval for x in bootstrap.agreement_intervals}
    for item in calculated.model_scopes:
        ci = intervals[(item.model_alias, item.scope_id)]
        ci_text = (f"{100*ci.lower:.2f}%–{100*ci.upper:.2f}%" if ci.status == "ok" else f"N/A ({ci.reason})")
        lines.append(
            f"| {item.model_alias} | {item.scope_id} | {_display(item.agreement.accuracy, percent=True)} | "
            f"{ci_text} | {_display(item.agreement.cohens_kappa)} | {_display(item.agreement.macro_f1)} | "
            f"{item.agreement.accuracy.n_bouts} |"
        )
    comparison = bootstrap.comparison
    lines += ["", "## Сравнение моделей", ""]
    if comparison.difference_percentage_points.status == "ok":
        diff = comparison.difference_percentage_points.value
        assert diff is not None
        ci = comparison.confidence_interval
        ci_text = f"{ci.lower:.2f}–{ci.upper:.2f} п.п." if ci.status == "ok" else f"N/A ({ci.reason})"
        lines.append(
            f"`{comparison.decision_alias} − {comparison.llm_alias}` на общей выборке: "
            f"{diff:.2f} п.п.; 95% CI: {ci_text}."
        )
    else:
        lines.append(f"N/A: {comparison.difference_percentage_points.reason}.")

    lines += [
        "", "## Проверка гипотез H1–H3", "",
        "Решения основаны только на семи заранее заданных перестановочных тестах "
        "с коррекцией Holm при α = 0,05. Bootstrap CI являются описательными.", "",
        "| Гипотеза | Тест | Эффект | 95% bootstrap CI | raw p | Holm p | N поединков | Решение |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    comparison_ids = [item.comparison_id for item in inference.intervals if item.comparison_id]
    inference_comparison = comparison_ids[0] if comparison_ids else "comparison"
    intervals_by_id = {item.interval_id: item.confidence_interval for item in inference.intervals}
    for test in inference.tests:
        interval_id = (
            test.test_id.removeprefix("H1:")
            if test.hypothesis_id == "H1"
            else f"{inference_comparison}:accuracy_difference"
            if test.test_id == "H2:accuracy_difference"
            else f"{inference_comparison}:pairwise_repeatability_difference"
            if test.test_id == "H3:pairwise_repeatability_difference"
            else f"{inference_comparison}:label_flip_rate_difference"
        )
        interval = intervals_by_id.get(interval_id)
        effect = "N/A" if test.effect is None else f"{test.effect:.4f}"
        ci_text = (
            f"{interval.lower:.4f}…{interval.upper:.4f}"
            if interval is not None and interval.status == "ok"
            else "N/A"
        )
        raw = "N/A" if test.raw_p_value is None else f"{test.raw_p_value:.6f}"
        adjusted = "N/A" if test.adjusted_p_value is None else f"{test.adjusted_p_value:.6f}"
        decision = next(
            item.status for item in inference.decisions if test.test_id in item.test_ids
        )
        lines.append(
            f"| {test.hypothesis_id} | `{test.test_id}` | {effect} | {ci_text} | "
            f"{raw} | {adjusted} | {test.n_bouts} | {decision} |"
        )
    lines += ["", "Итоговые решения:", ""]
    lines.extend(
        f"- **{item.decision_id}: {item.status}.** {item.conclusion}"
        for item in inference.decisions
    )

    lines += ["", "## Согласованность людей", "", "| Скоуп | Pairwise agreement | Fleiss' κ | N поединков |", "|---|---:|---:|---:|"]
    for item in calculated.human_scopes:
        lines.append(f"| {item.scope_id} | {_display(item.metrics.pairwise_agreement, percent=True)} | {_display(item.metrics.fleiss_kappa)} | {item.metrics.pairwise_agreement.n_bouts} |")

    lines += ["", "## Согласованность с отдельными судьями", "", "| Модель | Скоуп | Majority-to-individual | All-pairs |", "|---|---|---:|---:|"]
    for item in calculated.model_scopes:
        lines.append(
            f"| {item.model_alias} | {item.scope_id} | "
            f"{_display(item.individual_judge.majority_to_individual, percent=True)} | "
            f"{_display(item.individual_judge.all_pairs, percent=True)} |"
        )

    lines += ["", "## Разбивка по коллегиям", "", "Подробные значения Agreement Rate и позиционных метрик находятся в `by_collegium.csv`.", "", "## Устойчивость", "", "| Модель | Скоуп | Единогласие | Попарная повторяемость |", "|---|---|---:|---:|"]
    for item in calculated.model_scopes:
        lines.append(f"| {item.model_alias} | {item.scope_id} | {_display(item.repeatability.unanimity_rate, percent=True)} | {_display(item.repeatability.pairwise_repeatability, percent=True)} |")

    lines += ["", "## Позиционная чувствительность", "", "Интерпретировать эти показатели следует совместно с повторяемостью модели.", "", "| Модель | Скоуп | Label Flip Rate | Vote Share Change | N поединков |", "|---|---|---:|---:|---:|"]
    for item in calculated.position_scopes:
        lines.append(f"| {item.model_alias} | {item.scope_id} | {_display(item.metrics.label_flip_rate, percent=True)} | {_display(item.metrics.vote_share_change, percent=True)} | {item.metrics.label_flip_rate.n_bouts} |")

    lines += ["", "## Вероятностные метрики", "", "| Модель | Скоуп | Coverage | Brier | Brier individual | Vote Share MAE | MAE individual |", "|---|---|---:|---:|---:|---:|---:|"]
    for item in calculated.model_scopes:
        lines.append(
            f"| {item.model_alias} | {item.scope_id} | {_display(item.probability.coverage, percent=True)} | "
            f"{_display(item.probability.brier_score)} | {_display(item.probability.brier_score_individual)} | "
            f"{_display(item.probability.vote_share_mae)} | {_display(item.probability.vote_share_mae_individual)} |"
        )

    lines += ["", "## Описательные итоги по всем голосам", "", "| Модель | Полных поединков | Совпадений общего победителя | Доля |", "|---|---:|---:|---:|"]
    for alias in prepared.model_aliases:
        bouts = [item for item in aggregated.bout_summaries if item.model_alias == alias]
        matches = sum(item.majority_agreement for item in bouts)
        share = f"{100 * matches / len(bouts):.2f}%" if bouts else "N/A"
        lines.append(f"| {alias} | {len(bouts)} | {matches} | {share} |")
    lines += ["", "| Модель | Поединок | Счёт людей (P1:P2) | Победитель людей | Счёт модели (P1:P2) | Победитель модели | Совпадение |", "|---|---|---:|---|---:|---|---:|"]
    for item in aggregated.bout_summaries:
        lines.append(
            f"| {item.model_alias} | {item.bout_id} | {item.human_tally.player_1}:{item.human_tally.player_2} | "
            f"{item.human_tally.majority} | {item.model_tally.player_1}:{item.model_tally.player_2} | "
            f"{item.model_tally.majority} | {item.majority_agreement} |"
        )

    na = sorted({row["reason"] for row in _summary_rows(calculated, bootstrap, inference) if row["status"] == "not_applicable" and row["reason"]})
    lines += ["", "## Исключения и N/A", ""]
    lines.append(f"Зафиксировано исключений: {len(prepared.exclusions)}. Полный перечень — в `exclusions.csv`; диагностика кейсов — в `per_bout_collegium.csv`.")
    if na:
        lines.extend(f"- {reason}" for reason in na)

    lines += ["", "## Ограничения интерпретации", ""]
    if inputs.dataset == "pilot":
        lines.append("- Пилотная выборка не является независимой тестовой выборкой.")
    lines += [
        "- Решения людей используются как референс для согласованности, но не объявляются объективной истиной.",
        "- SHA-256 текущих транскрипций не доказывают, что в момент оценивания использовался идентичный текст.",
        "- Position bias необходимо интерпретировать вместе с repeatability: смена решения может отражать общую нестабильность.",
        "- Bootstrap-интервалы не устраняют возможную нерепрезентативность выборки.",
        "", "Подробные машинно-читаемые результаты и знаменатели находятся в `metrics.json` и CSV-файлах.", "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def _validate_staging(directory: Path) -> None:
    found = {path.name for path in directory.iterdir() if path.is_file()}
    if found != set(MANAGED_REPORT_FILES):
        raise ReportingError("staging directory does not contain exactly the six managed reports")
    for name in MANAGED_REPORT_FILES:
        if (directory / name).stat().st_size == 0:
            raise ReportingError(f"staged report is empty: {name}")
    with (directory / "metrics.json").open(encoding="utf-8") as stream:
        json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    for name, fields in (("summary.csv", SUMMARY_FIELDS), ("by_collegium.csv", BY_COLLEGIUM_FIELDS), ("per_bout_collegium.csv", PER_BOUT_FIELDS), ("exclusions.csv", EXCLUSION_FIELDS)):
        with (directory / name).open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream), None)
        if header != list(fields):
            raise ReportingError(f"unexpected CSV header in {name}")


def write_reports(
    inputs: AnalysisInputs,
    prepared: PreparedAnalysis,
    aggregated: AggregatedAnalysis,
    calculated: CalculatedMetrics,
    bootstrap: BootstrapAnalysis,
    inference: InferenceAnalysis,
    *,
    clock: Any = None,
) -> Path:
    """Stage, validate, and publish the six managed report files."""

    calculated_at = (clock or (lambda: datetime.now(timezone.utc)))()
    if calculated_at.tzinfo is None:
        raise ReportingError("calculation time must be timezone-aware")
    run_dir = inputs.run.run_dir
    staging = Path(tempfile.mkdtemp(prefix=".metrics-staging-", dir=run_dir))
    target = run_dir / "metrics"
    try:
        payload = build_metrics_payload(
            inputs,
            prepared,
            aggregated,
            calculated,
            bootstrap,
            inference,
            calculated_at=calculated_at,
        )
        (staging / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _write_csv(staging / "summary.csv", SUMMARY_FIELDS, _summary_rows(calculated, bootstrap, inference))
        _write_csv(staging / "by_collegium.csv", BY_COLLEGIUM_FIELDS, _by_collegium_rows(calculated))
        _write_csv(staging / "per_bout_collegium.csv", PER_BOUT_FIELDS, _per_bout_rows(inputs, prepared, aggregated))
        _write_csv(staging / "exclusions.csv", EXCLUSION_FIELDS, _exclusion_rows(prepared))
        (staging / "report.md").write_text(
            _report_markdown(inputs, prepared, aggregated, calculated, bootstrap, inference, calculated_at),
            encoding="utf-8",
        )
        _validate_staging(staging)
        if not target.exists():
            os.replace(staging, target)
        else:
            if not target.is_dir():
                raise ReportingError(f"metrics output path is not a directory: {target}")
            for name in MANAGED_REPORT_FILES:
                os.replace(staging / name, target / name)
    except ReportingError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ReportingError(f"Cannot create metrics reports: {error}") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return target


def console_summary(
    inputs: AnalysisInputs,
    prepared: PreparedAnalysis,
    calculated: CalculatedMetrics,
    output_dir: Path,
) -> str:
    """Return a deliberately short human-readable CLI summary."""

    lines = [
        f"Метрики рассчитаны: запуск {inputs.metadata.run_id}, выборка {inputs.dataset}.",
    ]
    individual = {item.model_alias: item for item in calculated.model_scopes if item.scope_id.startswith("model:")}
    for alias in prepared.model_aliases:
        item = individual[alias]
        lines.append(
            f"  {alias}: Accuracy {_display(item.agreement.accuracy, percent=True)}, "
            f"пригодно {item.agreement.accuracy.n_bouts}/{len(prepared.selected_bout_ids)} поединков."
        )
    lines.append(f"Отчёты: {output_dir}")
    return "\n".join(lines)


__all__ = [
    "MANAGED_REPORT_FILES", "ReportingError", "build_metrics_payload",
    "console_summary", "write_reports",
]
