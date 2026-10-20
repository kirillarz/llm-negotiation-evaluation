"""Pre-specified clustered inference for the confirmatory hypotheses H1-H3."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from statistics import fmean
from typing import Callable, Sequence

from src.experiment.config import ExperimentConfig, ModelConfigs
from src.metrics import (
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    EXACT_PERMUTATION_MAX_BOUTS,
    INFERENCE_ALPHA,
    MONTE_CARLO_PERMUTATIONS,
    PERCENTILE_BOUNDS,
)
from src.metrics.aggregation import AggregatedAnalysis, AggregatedCase
from src.metrics.calculations import (
    agreement_rate,
    cohens_kappa,
    position_bias,
    repeatability,
)
from src.metrics.contracts import (
    ConfidenceIntervalResult,
    HypothesisDecision,
    InferenceTestResult,
    MetricResult,
)
from src.metrics.preparation import PreparedAnalysis


INSUFFICIENT_BOUTS_REASON = "Inference requires at least two independent bouts"
MISSING_MODEL_REASON = "The configured model did not participate in this run"
NO_VALID_REPLICATES_REASON = "No bootstrap replication produced a defined metric"
SMALL_SAMPLE_BOUTS = 30
SMALL_SAMPLE_WARNING = (
    "The interval is based on fewer than 30 independent bouts and may be unstable"
)


@dataclass(frozen=True, slots=True)
class InferenceInterval:
    interval_id: str
    metric: str
    scope_id: str
    bout_ids: tuple[str, ...]
    point_estimate: MetricResult
    confidence_interval: ConfidenceIntervalResult
    model_alias: str | None = None
    comparison_id: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceAnalysis:
    alpha: float
    correction: str
    family_size: int
    exact_max_bouts: int
    monte_carlo_permutations: int
    seed: int
    intervals: tuple[InferenceInterval, ...]
    tests: tuple[InferenceTestResult, ...]
    decisions: tuple[HypothesisDecision, ...]


MetricFunction = Callable[[Sequence[AggregatedCase]], MetricResult]


def _models(config: ExperimentConfig | ModelConfigs):
    return config.models if isinstance(config, ExperimentConfig) else config


def _select(
    cases: Sequence[AggregatedCase], model_alias: str, bout_ids: Sequence[str]
) -> tuple[AggregatedCase, ...]:
    allowed = set(bout_ids)
    return tuple(
        case
        for case in cases
        if case.model_alias == model_alias and case.bout_id in allowed
    )


def _groups(
    cases: Sequence[AggregatedCase], bout_ids: Sequence[str]
) -> dict[str, tuple[AggregatedCase, ...]]:
    grouped = {
        bout_id: tuple(case for case in cases if case.bout_id == bout_id)
        for bout_id in bout_ids
    }
    for bout_id, bout_cases in grouped.items():
        if len(bout_cases) != 3 or len({case.collegium for case in bout_cases}) != 3:
            raise ValueError(
                f"bout {bout_id!r} must contain exactly three distinct collegia"
            )
    return grouped


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _na_interval(
    metric: str,
    reason: str,
    *,
    unit: str,
    n_bouts: int,
    n_cases: int,
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


def _bootstrap_interval(
    cases: Sequence[AggregatedCase],
    bout_ids: Sequence[str],
    metric_fn: MetricFunction,
    *,
    metric: str,
    unit: str,
    scale: float = 1.0,
    replications: int,
    seed: int,
) -> tuple[MetricResult, ConfidenceIntervalResult]:
    point = metric_fn(cases)
    n_bouts, n_cases = len(bout_ids), len(cases)
    if point.status != "ok":
        return point, _na_interval(
            metric,
            point.reason or "Metric is not applicable",
            unit=unit,
            n_bouts=n_bouts,
            n_cases=n_cases,
            replications=replications,
            seed=seed,
        )
    if n_bouts < 2:
        return point, _na_interval(
            metric,
            INSUFFICIENT_BOUTS_REASON,
            unit=unit,
            n_bouts=n_bouts,
            n_cases=n_cases,
            replications=replications,
            seed=seed,
        )
    grouped = _groups(cases, bout_ids)
    random = Random(seed)
    samples: list[float] = []
    for _ in range(replications):
        sampled = tuple(
            case
            for _index in range(n_bouts)
            for case in grouped[random.choice(tuple(bout_ids))]
        )
        value = metric_fn(sampled)
        if value.status == "ok":
            assert value.value is not None
            samples.append(scale * value.value)
    if not samples:
        return point, _na_interval(
            metric,
            NO_VALID_REPLICATES_REASON,
            unit=unit,
            n_bouts=n_bouts,
            n_cases=n_cases,
            replications=replications,
            seed=seed,
        )
    lower, upper = PERCENTILE_BOUNDS
    return point, ConfidenceIntervalResult(
        metric=metric,
        status="ok",
        lower=_percentile(samples, lower),
        upper=_percentile(samples, upper),
        unit=unit,
        confidence_level=CONFIDENCE_LEVEL,
        replications=replications,
        valid_replications=len(samples),
        seed=seed,
        n_bouts=n_bouts,
        n_cases=n_cases,
        warning=SMALL_SAMPLE_WARNING if n_bouts < SMALL_SAMPLE_BOUTS else None,
    )


def _paired_bootstrap_interval(
    left: Sequence[AggregatedCase],
    right: Sequence[AggregatedCase],
    bout_ids: Sequence[str],
    metric_fn: MetricFunction,
    *,
    metric: str,
    unit: str,
    scale: float,
    replications: int,
    seed: int,
) -> tuple[MetricResult, ConfidenceIntervalResult]:
    n_bouts = len(bout_ids)
    n_cases = len(left) + len(right)
    left_point, right_point = metric_fn(left), metric_fn(right)
    if left_point.status != "ok" or right_point.status != "ok":
        reason = left_point.reason or right_point.reason or "Metric is not applicable"
        point = MetricResult.not_applicable(
            metric, reason, n_bouts=n_bouts, n_cases=n_cases
        )
        return point, _na_interval(
            metric,
            reason,
            unit=unit,
            n_bouts=n_bouts,
            n_cases=n_cases,
            replications=replications,
            seed=seed,
        )
    assert left_point.value is not None and right_point.value is not None
    point = MetricResult.ok(
        metric,
        scale * (left_point.value - right_point.value),
        n_bouts=n_bouts,
        n_cases=n_cases,
    )
    if n_bouts < 2:
        return point, _na_interval(
            metric,
            INSUFFICIENT_BOUTS_REASON,
            unit=unit,
            n_bouts=n_bouts,
            n_cases=n_cases,
            replications=replications,
            seed=seed,
        )
    left_groups, right_groups = _groups(left, bout_ids), _groups(right, bout_ids)
    random = Random(seed)
    samples: list[float] = []
    ids = tuple(bout_ids)
    for _ in range(replications):
        sampled_ids = [random.choice(ids) for _index in range(n_bouts)]
        sampled_left = tuple(case for key in sampled_ids for case in left_groups[key])
        sampled_right = tuple(case for key in sampled_ids for case in right_groups[key])
        left_value, right_value = metric_fn(sampled_left), metric_fn(sampled_right)
        if left_value.status == "ok" and right_value.status == "ok":
            assert left_value.value is not None and right_value.value is not None
            samples.append(scale * (left_value.value - right_value.value))
    if not samples:
        return point, _na_interval(
            metric,
            NO_VALID_REPLICATES_REASON,
            unit=unit,
            n_bouts=n_bouts,
            n_cases=n_cases,
            replications=replications,
            seed=seed,
        )
    lower, upper = PERCENTILE_BOUNDS
    return point, ConfidenceIntervalResult(
        metric=metric,
        status="ok",
        lower=_percentile(samples, lower),
        upper=_percentile(samples, upper),
        unit=unit,
        confidence_level=CONFIDENCE_LEVEL,
        replications=replications,
        valid_replications=len(samples),
        seed=seed,
        n_bouts=n_bouts,
        n_cases=n_cases,
        warning=SMALL_SAMPLE_WARNING if n_bouts < SMALL_SAMPLE_BOUTS else None,
    )


def _pairwise_repeatability(cases: Sequence[AggregatedCase]) -> MetricResult:
    return repeatability(cases).pairwise_repeatability


def _label_flip_rate(cases: Sequence[AggregatedCase]) -> MetricResult:
    return position_bias(cases).label_flip_rate


def _label_kappa(labels: Sequence[tuple[str, str]]) -> float | None:
    if not labels:
        return None
    observed = fmean(model == human for model, human in labels)
    model_p1 = fmean(model == "player_1" for model, _human in labels)
    human_p1 = fmean(human == "player_1" for _model, human in labels)
    expected = model_p1 * human_p1 + (1 - model_p1) * (1 - human_p1)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def _assignments(n_bouts: int, permutations: int, seed: int):
    exact = n_bouts <= EXACT_PERMUTATION_MAX_BOUTS
    if exact:
        for mask in range(1 << n_bouts):
            yield tuple(bool(mask & (1 << index)) for index in range(n_bouts))
    else:
        random = Random(seed)
        for _ in range(permutations):
            yield tuple(bool(random.getrandbits(1)) for _index in range(n_bouts))


def _raw_p_value(extreme: int, valid: int, exact: bool) -> float:
    return extreme / valid if exact else (extreme + 1) / (valid + 1)


def _na_test(
    test_id: str,
    hypothesis_id: str,
    metric: str,
    scope_id: str,
    null_hypothesis: str,
    alternative: str,
    reason: str,
    *,
    n_bouts: int = 0,
    n_cases: int = 0,
    seed: int,
) -> InferenceTestResult:
    return InferenceTestResult(
        test_id=test_id,
        hypothesis_id=hypothesis_id,
        metric=metric,
        scope_id=scope_id,
        null_hypothesis=null_hypothesis,
        alternative=alternative,
        status="not_applicable",
        effect=None,
        raw_p_value=None,
        adjusted_p_value=None,
        alpha=INFERENCE_ALPHA,
        method="cluster_permutation_by_bout",
        exact=n_bouts <= EXACT_PERMUTATION_MAX_BOUTS,
        permutations=0,
        seed=seed,
        n_bouts=n_bouts,
        n_cases=n_cases,
        reason=reason,
    )


def _h1_test(
    cases: Sequence[AggregatedCase],
    bout_ids: Sequence[str],
    *,
    alias: str,
    metric: str,
    permutations: int,
    seed: int,
) -> InferenceTestResult:
    test_id = f"H1:{alias}:{metric}"
    null = "Accuracy = 0.5" if metric == "agreement_rate" else "Cohen's kappa = 0"
    n_bouts, n_cases = len(bout_ids), len(cases)
    if n_bouts < 2:
        return _na_test(
            test_id, "H1", metric, f"model:{alias}", null, "greater",
            INSUFFICIENT_BOUTS_REASON, n_bouts=n_bouts, n_cases=n_cases, seed=seed,
        )
    grouped = _groups(cases, bout_ids)
    labels_by_bout = {
        bout_id: tuple(
            (case.original_tally.majority, case.human_tally.majority)
            for case in grouped[bout_id]
        )
        for bout_id in bout_ids
    }
    labels = tuple(pair for bout_id in bout_ids for pair in labels_by_bout[bout_id])
    if metric == "agreement_rate":
        observed = fmean(model == human for model, human in labels)
    else:
        observed = _label_kappa(labels)
        if observed is None:
            return _na_test(
                test_id, "H1", metric, f"model:{alias}", null, "greater",
                "Cohen's kappa is undefined for the observed data",
                n_bouts=n_bouts, n_cases=n_cases, seed=seed,
            )
    exact = n_bouts <= EXACT_PERMUTATION_MAX_BOUTS
    extreme = valid = 0
    ids = tuple(bout_ids)
    for assignment in _assignments(n_bouts, permutations, seed):
        permuted = []
        for invert, bout_id in zip(assignment, ids, strict=True):
            for model, human in labels_by_bout[bout_id]:
                if invert:
                    model = "player_2" if model == "player_1" else "player_1"
                permuted.append((model, human))
        value = (
            fmean(model == human for model, human in permuted)
            if metric == "agreement_rate"
            else _label_kappa(permuted)
        )
        if value is None:
            continue
        valid += 1
        extreme += value >= observed - 1e-15
    if not valid:
        return _na_test(
            test_id, "H1", metric, f"model:{alias}", null, "greater",
            "No permutation produced a defined statistic",
            n_bouts=n_bouts, n_cases=n_cases, seed=seed,
        )
    raw = _raw_p_value(extreme, valid, exact)
    return InferenceTestResult(
        test_id=test_id,
        hypothesis_id="H1",
        metric=metric,
        scope_id=f"model:{alias}",
        null_hypothesis=null,
        alternative="greater",
        status="ok",
        effect=observed,
        raw_p_value=raw,
        adjusted_p_value=raw,
        alpha=INFERENCE_ALPHA,
        method="cluster_label_inversion_by_bout",
        exact=exact,
        permutations=valid,
        seed=seed,
        n_bouts=n_bouts,
        n_cases=n_cases,
    )


def _paired_test(
    left: Sequence[AggregatedCase],
    right: Sequence[AggregatedCase],
    bout_ids: Sequence[str],
    score: Callable[[Sequence[AggregatedCase]], MetricResult],
    *,
    test_id: str,
    hypothesis_id: str,
    metric: str,
    scope_id: str,
    permutations: int,
    seed: int,
) -> InferenceTestResult:
    n_bouts, n_cases = len(bout_ids), len(left) + len(right)
    if n_bouts < 2:
        return _na_test(
            test_id, hypothesis_id, metric, scope_id, "model difference = 0",
            "two_sided", INSUFFICIENT_BOUTS_REASON,
            n_bouts=n_bouts, n_cases=n_cases, seed=seed,
        )
    left_groups, right_groups = _groups(left, bout_ids), _groups(right, bout_ids)
    differences = []
    for bout_id in bout_ids:
        left_value, right_value = score(left_groups[bout_id]), score(right_groups[bout_id])
        if left_value.status != "ok" or right_value.status != "ok":
            return _na_test(
                test_id, hypothesis_id, metric, scope_id, "model difference = 0",
                "two_sided", left_value.reason or right_value.reason or "Metric is not applicable",
                n_bouts=n_bouts, n_cases=n_cases, seed=seed,
            )
        assert left_value.value is not None and right_value.value is not None
        differences.append(left_value.value - right_value.value)
    observed = 100 * fmean(differences)
    exact = n_bouts <= EXACT_PERMUTATION_MAX_BOUTS
    extreme = valid = 0
    for assignment in _assignments(n_bouts, permutations, seed):
        value = 100 * fmean(
            -difference if swap else difference
            for difference, swap in zip(differences, assignment, strict=True)
        )
        valid += 1
        extreme += abs(value) >= abs(observed) - 1e-12
    raw = _raw_p_value(extreme, valid, exact)
    return InferenceTestResult(
        test_id=test_id,
        hypothesis_id=hypothesis_id,
        metric=metric,
        scope_id=scope_id,
        null_hypothesis="model difference = 0",
        alternative="two_sided",
        status="ok",
        effect=observed,
        raw_p_value=raw,
        adjusted_p_value=raw,
        alpha=INFERENCE_ALPHA,
        method="paired_cluster_model_swap_by_bout",
        exact=exact,
        permutations=valid,
        seed=seed,
        n_bouts=n_bouts,
        n_cases=n_cases,
    )


def apply_holm(
    tests: Sequence[InferenceTestResult], *, family_size: int = 7
) -> tuple[InferenceTestResult, ...]:
    """Apply Holm while retaining unavailable planned tests in the family."""

    if len(tests) != family_size:
        raise ValueError("Holm family must contain exactly the pre-specified tests")
    ranked = sorted(
        enumerate(tests),
        key=lambda item: (
            item[1].raw_p_value if item[1].status == "ok" else 1.0,
            item[1].test_id,
        ),
    )
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, (index, test) in enumerate(ranked):
        effective = test.raw_p_value if test.status == "ok" else 1.0
        assert effective is not None
        running = max(running, min(1.0, (family_size - rank) * effective))
        if test.status == "ok":
            adjusted[index] = running
    return tuple(
        test.model_copy(update={"adjusted_p_value": adjusted[index]})
        if test.status == "ok"
        else test
        for index, test in enumerate(tests)
    )


def _decisions(
    tests: Sequence[InferenceTestResult], aliases: Sequence[str]
) -> tuple[HypothesisDecision, ...]:
    by_id = {test.test_id: test for test in tests}
    decisions: list[HypothesisDecision] = []
    for alias in aliases:
        selected = [
            by_id[f"H1:{alias}:agreement_rate"],
            by_id[f"H1:{alias}:cohens_kappa"],
        ]
        if any(test.status != "ok" for test in selected):
            status, conclusion = "not_evaluable", "Недостаточно данных для проверки H1."
        else:
            confirmed = all(
                test.effect is not None
                and test.effect > (0.5 if test.metric == "agreement_rate" else 0.0)
                and test.adjusted_p_value is not None
                and test.adjusted_p_value < INFERENCE_ALPHA
                for test in selected
            )
            status = "confirmed" if confirmed else "not_confirmed"
            conclusion = (
                "Accuracy выше 0,5 и Cohen's kappa выше 0 после коррекции Holm."
                if confirmed
                else "Совместный критерий Accuracy и Cohen's kappa не выполнен."
            )
        decisions.append(HypothesisDecision(
            decision_id=f"H1:{alias}", hypothesis_id="H1", model_alias=alias,
            status=status, test_ids=tuple(test.test_id for test in selected),
            conclusion=conclusion,
        ))

    h2 = by_id["H2:accuracy_difference"]
    h2_status = (
        "not_evaluable" if h2.status != "ok"
        else "confirmed" if h2.adjusted_p_value is not None and h2.adjusted_p_value < INFERENCE_ALPHA
        else "not_confirmed"
    )
    decisions.append(HypothesisDecision(
        decision_id="H2", hypothesis_id="H2", status=h2_status,
        test_ids=(h2.test_id,),
        conclusion=(
            "Разница Accuracy моделей статистически значима после коррекции Holm."
            if h2_status == "confirmed" else
            "Недостаточно данных для проверки H2." if h2_status == "not_evaluable" else
            "Статистически значимая разница Accuracy моделей не установлена."
        ),
    ))

    h3_tests = [by_id["H3:pairwise_repeatability_difference"], by_id["H3:label_flip_rate_difference"]]
    evaluable = [test for test in h3_tests if test.status == "ok"]
    if not evaluable:
        h3_status = "not_evaluable"
    elif any(test.adjusted_p_value is not None and test.adjusted_p_value < INFERENCE_ALPHA for test in evaluable):
        h3_status = "confirmed"
    elif len(evaluable) != len(h3_tests):
        h3_status = "not_evaluable"
    else:
        h3_status = "not_confirmed"
    decisions.append(HypothesisDecision(
        decision_id="H3", hypothesis_id="H3", status=h3_status,
        test_ids=tuple(test.test_id for test in h3_tests),
        conclusion=(
            "Хотя бы одно различие устойчивости значимо после коррекции Holm."
            if h3_status == "confirmed" else
            "Недостаточно данных для проверки H3." if h3_status == "not_evaluable" else
            "Значимые различия устойчивости моделей не установлены."
        ),
    ))
    return tuple(decisions)


def calculate_inference(
    prepared: PreparedAnalysis,
    aggregated: AggregatedAnalysis,
    config: ExperimentConfig | ModelConfigs,
    *,
    bootstrap_replications: int = BOOTSTRAP_REPLICATIONS,
    permutations: int = MONTE_CARLO_PERMUTATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> InferenceAnalysis:
    """Calculate all pre-specified intervals, tests, and H1-H3 decisions."""

    if bootstrap_replications <= 0 or permutations <= 0:
        raise ValueError("replication counts must be positive")
    models = _models(config)
    aliases = (models.llm.alias, models.decision.alias)
    comparison_id = f"{models.decision.alias}_minus_{models.llm.alias}"
    intervals: list[InferenceInterval] = []
    tests: list[InferenceTestResult] = []

    for alias in aliases:
        if alias not in prepared.model_aliases:
            reason = f"{MISSING_MODEL_REASON}: {alias}"
            for metric in ("agreement_rate", "cohens_kappa"):
                tests.append(_na_test(
                    f"H1:{alias}:{metric}", "H1", metric, f"model:{alias}",
                    "Accuracy = 0.5" if metric == "agreement_rate" else "Cohen's kappa = 0",
                    "greater", reason, seed=seed,
                ))
            continue
        scope = prepared.model_scope(alias)
        cases = _select(aggregated.original_cases, alias, scope.bout_ids)
        metric_specs = (
            ("agreement_rate", agreement_rate, "proportion"),
            ("cohens_kappa", cohens_kappa, "coefficient"),
            ("pairwise_repeatability", _pairwise_repeatability, "proportion"),
        )
        for metric, metric_fn, unit in metric_specs:
            point, interval = _bootstrap_interval(
                cases, scope.bout_ids, metric_fn, metric=metric, unit=unit,
                replications=bootstrap_replications, seed=seed,
            )
            intervals.append(InferenceInterval(
                interval_id=f"{alias}:{metric}", metric=metric,
                scope_id=scope.scope_id, bout_ids=scope.bout_ids,
                point_estimate=point, confidence_interval=interval,
                model_alias=alias,
            ))
        position_scope = prepared.position_scope(alias)
        position_cases = _select(
            aggregated.position_cases, alias, position_scope.bout_ids
        )
        point, interval = _bootstrap_interval(
            position_cases, position_scope.bout_ids, _label_flip_rate,
            metric="label_flip_rate", unit="proportion",
            replications=bootstrap_replications, seed=seed,
        )
        intervals.append(InferenceInterval(
            interval_id=f"{alias}:label_flip_rate", metric="label_flip_rate",
            scope_id=position_scope.scope_id, bout_ids=position_scope.bout_ids,
            point_estimate=point, confidence_interval=interval, model_alias=alias,
        ))
        tests.extend((
            _h1_test(cases, scope.bout_ids, alias=alias, metric="agreement_rate", permutations=permutations, seed=seed),
            _h1_test(cases, scope.bout_ids, alias=alias, metric="cohens_kappa", permutations=permutations, seed=seed),
        ))

    common_ids = prepared.common_scope.bout_ids
    decision_cases = _select(aggregated.original_cases, models.decision.alias, common_ids)
    llm_cases = _select(aggregated.original_cases, models.llm.alias, common_ids)
    paired_specs = (
        ("accuracy_difference", agreement_rate, "agreement_rate_difference_percentage_points", "percentage_points", 100.0),
        ("cohens_kappa_difference", cohens_kappa, "cohens_kappa_difference", "coefficient", 1.0),
        ("pairwise_repeatability_difference", _pairwise_repeatability, "pairwise_repeatability_difference_percentage_points", "percentage_points", 100.0),
    )
    for name, metric_fn, metric, unit, scale in paired_specs:
        point, interval = _paired_bootstrap_interval(
            decision_cases, llm_cases, common_ids, metric_fn, metric=metric,
            unit=unit, scale=scale, replications=bootstrap_replications, seed=seed,
        )
        intervals.append(InferenceInterval(
            interval_id=f"{comparison_id}:{name}", metric=metric,
            scope_id="common", bout_ids=common_ids, point_estimate=point,
            confidence_interval=interval, comparison_id=comparison_id,
        ))

    position_ids = prepared.position_common_scope.bout_ids
    decision_position = _select(aggregated.position_cases, models.decision.alias, position_ids)
    llm_position = _select(aggregated.position_cases, models.llm.alias, position_ids)
    point, interval = _paired_bootstrap_interval(
        decision_position, llm_position, position_ids, _label_flip_rate,
        metric="label_flip_rate_difference_percentage_points",
        unit="percentage_points", scale=100.0,
        replications=bootstrap_replications, seed=seed,
    )
    intervals.append(InferenceInterval(
        interval_id=f"{comparison_id}:label_flip_rate_difference",
        metric="label_flip_rate_difference_percentage_points",
        scope_id="position:common", bout_ids=position_ids,
        point_estimate=point, confidence_interval=interval,
        comparison_id=comparison_id,
    ))

    tests.extend((
        _paired_test(
            decision_cases, llm_cases, common_ids, agreement_rate,
            test_id="H2:accuracy_difference", hypothesis_id="H2",
            metric="agreement_rate_difference_percentage_points", scope_id="common",
            permutations=permutations, seed=seed,
        ),
        _paired_test(
            decision_cases, llm_cases, common_ids, _pairwise_repeatability,
            test_id="H3:pairwise_repeatability_difference", hypothesis_id="H3",
            metric="pairwise_repeatability_difference_percentage_points", scope_id="common",
            permutations=permutations, seed=seed,
        ),
        _paired_test(
            decision_position, llm_position, position_ids, _label_flip_rate,
            test_id="H3:label_flip_rate_difference", hypothesis_id="H3",
            metric="label_flip_rate_difference_percentage_points", scope_id="position:common",
            permutations=permutations, seed=seed,
        ),
    ))
    corrected = apply_holm(tests)
    return InferenceAnalysis(
        alpha=INFERENCE_ALPHA,
        correction="Holm",
        family_size=7,
        exact_max_bouts=EXACT_PERMUTATION_MAX_BOUTS,
        monte_carlo_permutations=permutations,
        seed=seed,
        intervals=tuple(intervals),
        tests=corrected,
        decisions=_decisions(corrected, aliases),
    )


__all__ = [
    "InferenceAnalysis", "InferenceInterval", "apply_holm", "calculate_inference",
]
