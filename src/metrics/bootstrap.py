"""Clustered bootstrap intervals and paired model comparison.

The independent sampling unit is always a complete bout.  All collegium cases
belonging to a sampled bout travel together, and the paired comparison applies
the same sampled bout multiplicities to both models.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import fmean
from typing import Sequence

from src.experiment.config import ExperimentConfig, ModelConfigs
from src.metrics import (
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    PERCENTILE_BOUNDS,
)
from src.metrics.aggregation import AggregatedAnalysis, AggregatedCase
from src.metrics.contracts import ConfidenceIntervalResult, MetricResult
from src.metrics.preparation import PreparedAnalysis


INSUFFICIENT_BOUTS_REASON = (
    "A confidence interval requires at least two independent bouts"
)
MISSING_MODEL_REASON = "The configured model did not participate in this run"
NO_COMMON_BOUTS_REASON = "The participating models have no common eligible bouts"
SMALL_SAMPLE_BOUTS = 30
SMALL_SAMPLE_WARNING = (
    "The confidence interval is based on fewer than 30 independent bouts and "
    "may be unstable"
)


@dataclass(frozen=True, slots=True)
class AgreementBootstrap:
    model_alias: str
    scope_id: str
    bout_ids: tuple[str, ...]
    point_estimate: MetricResult
    confidence_interval: ConfidenceIntervalResult


@dataclass(frozen=True, slots=True)
class ModelComparison:
    decision_alias: str
    llm_alias: str
    scope_id: str
    bout_ids: tuple[str, ...]
    difference_percentage_points: MetricResult
    confidence_interval: ConfidenceIntervalResult


@dataclass(frozen=True, slots=True)
class BootstrapAnalysis:
    agreement_intervals: tuple[AgreementBootstrap, ...]
    comparison: ModelComparison


def _case_agrees(case: AggregatedCase) -> bool:
    if not case.original_complete or case.original_agreement is None:
        raise ValueError("bootstrap requires complete eligible original cases")
    return case.original_agreement


def _bout_scores(
    cases: Sequence[AggregatedCase], bout_ids: Sequence[str]
) -> dict[str, float]:
    requested = tuple(bout_ids)
    if len(set(requested)) != len(requested):
        raise ValueError("bootstrap bout_ids must be unique")
    grouped: dict[str, list[AggregatedCase]] = {bout_id: [] for bout_id in requested}
    for case in cases:
        if case.bout_id in grouped:
            grouped[case.bout_id].append(case)
    scores: dict[str, float] = {}
    for bout_id, bout_cases in grouped.items():
        if not bout_cases:
            raise ValueError(f"no eligible cases for bout {bout_id!r}")
        collegia = [case.collegium for case in bout_cases]
        if len(collegia) != 3 or len(set(collegia)) != 3:
            raise ValueError(
                f"bout {bout_id!r} must contain exactly three distinct collegia"
            )
        scores[bout_id] = fmean(_case_agrees(case) for case in bout_cases)
    return scores


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Hyndman-Fan type 7 percentile (linear interpolation)."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sample")
    position = (len(ordered) - 1) * percentile / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def _na_interval(
    metric: str,
    reason: str,
    *,
    n_bouts: int,
    n_cases: int,
    unit: str,
    replications: int,
    seed: int,
) -> ConfidenceIntervalResult:
    return ConfidenceIntervalResult(
        metric=metric,
        status="not_applicable",
        lower=None,
        upper=None,
        unit=unit,
        confidence_level=CONFIDENCE_LEVEL,
        replications=replications,
        valid_replications=0,
        seed=seed,
        n_bouts=n_bouts,
        n_cases=n_cases,
        reason=reason,
    )


def _interval(
    metric: str,
    samples: Sequence[float],
    *,
    n_bouts: int,
    n_cases: int,
    unit: str,
    replications: int,
    seed: int,
) -> ConfidenceIntervalResult:
    lower, upper = PERCENTILE_BOUNDS
    return ConfidenceIntervalResult(
        metric=metric,
        status="ok",
        lower=_percentile(samples, lower),
        upper=_percentile(samples, upper),
        unit=unit,
        confidence_level=CONFIDENCE_LEVEL,
        replications=replications,
        valid_replications=replications,
        seed=seed,
        n_bouts=n_bouts,
        n_cases=n_cases,
        warning=SMALL_SAMPLE_WARNING if n_bouts < SMALL_SAMPLE_BOUTS else None,
    )


def bootstrap_agreement_ci(
    cases: Sequence[AggregatedCase],
    bout_ids: Sequence[str],
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> ConfidenceIntervalResult:
    """Bootstrap Agreement Rate by resampling complete bout clusters."""

    if replications <= 0:
        raise ValueError("replications must be positive")
    n_bouts = len(bout_ids)
    n_cases = len(cases)
    if n_bouts < 2:
        return _na_interval(
            "agreement_rate",
            INSUFFICIENT_BOUTS_REASON,
            n_bouts=n_bouts,
            n_cases=n_cases,
            unit="proportion",
            replications=replications,
            seed=seed,
        )
    scores = _bout_scores(cases, bout_ids)
    ordered_ids = tuple(bout_ids)
    random = Random(seed)
    samples = tuple(
        fmean(scores[random.choice(ordered_ids)] for _ in range(n_bouts))
        for _ in range(replications)
    )
    return _interval(
        "agreement_rate",
        samples,
        n_bouts=n_bouts,
        n_cases=n_cases,
        unit="proportion",
        replications=replications,
        seed=seed,
    )


def paired_model_comparison(
    cases: Sequence[AggregatedCase],
    bout_ids: Sequence[str],
    *,
    decision_alias: str,
    llm_alias: str,
    participating_aliases: Sequence[str] | None = None,
    replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> ModelComparison:
    """Compare configured decision and LLM roles on paired common bouts."""

    if replications <= 0:
        raise ValueError("replications must be positive")
    common_ids = tuple(bout_ids)
    aliases = set(participating_aliases or (case.model_alias for case in cases))
    missing = [alias for alias in (decision_alias, llm_alias) if alias not in aliases]
    if missing:
        reason = f"{MISSING_MODEL_REASON}: {', '.join(missing)}"
        difference = MetricResult.not_applicable(
            "agreement_rate_difference_percentage_points", reason
        )
        interval = _na_interval(
            difference.metric,
            reason,
            n_bouts=0,
            n_cases=0,
            unit="percentage_points",
            replications=replications,
            seed=seed,
        )
        return ModelComparison(
            decision_alias, llm_alias, "common", (), difference, interval
        )
    if not common_ids:
        difference = MetricResult.not_applicable(
            "agreement_rate_difference_percentage_points", NO_COMMON_BOUTS_REASON
        )
        interval = _na_interval(
            difference.metric,
            NO_COMMON_BOUTS_REASON,
            n_bouts=0,
            n_cases=0,
            unit="percentage_points",
            replications=replications,
            seed=seed,
        )
        return ModelComparison(
            decision_alias, llm_alias, "common", (), difference, interval
        )

    common_set = set(common_ids)
    decision_cases = tuple(
        case
        for case in cases
        if case.model_alias == decision_alias and case.bout_id in common_set
    )
    llm_cases = tuple(
        case
        for case in cases
        if case.model_alias == llm_alias and case.bout_id in common_set
    )
    decision_scores = _bout_scores(decision_cases, common_ids)
    llm_scores = _bout_scores(llm_cases, common_ids)
    n_bouts = len(common_ids)
    n_cases = len(decision_cases) + len(llm_cases)
    point = 100 * fmean(
        decision_scores[bout_id] - llm_scores[bout_id] for bout_id in common_ids
    )
    difference = MetricResult.ok(
        "agreement_rate_difference_percentage_points",
        point,
        n_bouts=n_bouts,
        n_cases=n_cases,
    )
    if n_bouts < 2:
        interval = _na_interval(
            difference.metric,
            INSUFFICIENT_BOUTS_REASON,
            n_bouts=n_bouts,
            n_cases=n_cases,
            unit="percentage_points",
            replications=replications,
            seed=seed,
        )
    else:
        random = Random(seed)
        samples = []
        for _ in range(replications):
            sampled_ids = [random.choice(common_ids) for _ in range(n_bouts)]
            samples.append(
                100
                * fmean(
                    decision_scores[bout_id] - llm_scores[bout_id]
                    for bout_id in sampled_ids
                )
            )
        interval = _interval(
            difference.metric,
            samples,
            n_bouts=n_bouts,
            n_cases=n_cases,
            unit="percentage_points",
            replications=replications,
            seed=seed,
        )
    return ModelComparison(
        decision_alias,
        llm_alias,
        "common",
        common_ids,
        difference,
        interval,
    )


def calculate_bootstrap(
    prepared: PreparedAnalysis,
    aggregated: AggregatedAnalysis,
    config: ExperimentConfig | ModelConfigs,
    *,
    replications: int = BOOTSTRAP_REPLICATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapAnalysis:
    """Calculate model intervals and the configured role-based comparison."""

    results: list[AgreementBootstrap] = []
    for model_alias in prepared.model_aliases:
        for scope in (prepared.model_scope(model_alias), prepared.common_scope):
            scope_id = scope.scope_id if scope.kind == "model" else f"common:{model_alias}"
            cases = tuple(
                case
                for case in aggregated.original_cases
                if case.model_alias == model_alias and case.bout_id in scope.bout_ids
            )
            point = (
                MetricResult.ok(
                    "agreement_rate",
                    fmean(_case_agrees(case) for case in cases),
                    n_bouts=len(scope.bout_ids),
                    n_cases=len(cases),
                )
                if cases
                else MetricResult.not_applicable(
                    "agreement_rate",
                    "No eligible bout-collegium cases are available",
                )
            )
            results.append(
                AgreementBootstrap(
                    model_alias,
                    scope_id,
                    scope.bout_ids,
                    point,
                    bootstrap_agreement_ci(
                        cases,
                        scope.bout_ids,
                        replications=replications,
                        seed=seed,
                    ),
                )
            )

    models = config.models if isinstance(config, ExperimentConfig) else config
    comparison = paired_model_comparison(
        aggregated.original_cases,
        prepared.common_scope.bout_ids,
        decision_alias=models.decision.alias,
        llm_alias=models.llm.alias,
        participating_aliases=prepared.model_aliases,
        replications=replications,
        seed=seed,
    )
    return BootstrapAnalysis(tuple(results), comparison)


__all__ = [
    "AgreementBootstrap",
    "BootstrapAnalysis",
    "ModelComparison",
    "SMALL_SAMPLE_BOUTS",
    "bootstrap_agreement_ci",
    "calculate_bootstrap",
    "paired_model_comparison",
]
