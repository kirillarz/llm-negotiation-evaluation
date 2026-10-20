from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.metrics.aggregation import AggregatedAnalysis
from src.metrics.bootstrap import (
    bootstrap_agreement_ci,
    calculate_bootstrap,
    paired_model_comparison,
)
from src.metrics.contracts import MetricScope
from tests.metrics.test_calculations import _case


def _bout(alias, bout_id, agreements):
    cases = []
    for collegium, agrees in zip(
        ("hiring", "negotiation", "ownership"), agreements, strict=True
    ):
        human = ["player_1"] * 3
        model = ["player_1"] * 3 if agrees else ["player_2"] * 3
        cases.append(_case(bout_id, collegium, human, model, model_alias=alias))
    return tuple(cases)


def test_cluster_bootstrap_keeps_three_collegia_together():
    cases = _bout("m", "perfect", (True, True, True)) + _bout(
        "m", "zero", (False, False, False)
    )

    result = bootstrap_agreement_ci(cases, ("perfect", "zero"), replications=500)

    assert result.status == "ok"
    assert result.n_bouts == 2 and result.n_cases == 6
    assert result.lower == 0.0 and result.upper == 1.0
    assert result.warning


def test_paired_comparison_uses_same_sampled_bouts_for_both_models():
    # Per-bout differences are always +1/3, so every correctly paired replicate
    # must have exactly the same 33.333... percentage-point difference.
    cases = (
        _bout("decision", "1", (True, True, False))
        + _bout("llm", "1", (True, False, False))
        + _bout("decision", "2", (True, False, True))
        + _bout("llm", "2", (False, False, True))
    )

    result = paired_model_comparison(
        cases,
        ("1", "2"),
        decision_alias="decision",
        llm_alias="llm",
        replications=500,
    )

    expected = 100 / 3
    assert result.difference_percentage_points.value == pytest.approx(expected)
    assert result.confidence_interval.lower == pytest.approx(expected)
    assert result.confidence_interval.upper == pytest.approx(expected)


def test_bootstrap_is_fully_reproducible_for_fixed_seed():
    cases = (
        _bout("m", "1", (True, True, True))
        + _bout("m", "2", (False, False, False))
        + _bout("m", "3", (True, False, False))
    )

    first = bootstrap_agreement_ci(cases, ("1", "2", "3"), replications=777)
    second = bootstrap_agreement_ci(cases, ("1", "2", "3"), replications=777)

    assert first == second


@pytest.mark.parametrize("n_bouts", [0, 1])
def test_ci_is_na_with_fewer_than_two_independent_bouts(n_bouts):
    bout_ids = tuple(str(index) for index in range(n_bouts))
    cases = tuple(case for bout_id in bout_ids for case in _bout("m", bout_id, (1, 1, 1)))

    result = bootstrap_agreement_ci(cases, bout_ids, replications=10)

    assert result.status == "not_applicable"
    assert result.lower is None and result.upper is None
    assert result.reason


def test_single_model_makes_comparison_na_without_losing_individual_result():
    cases = _bout("llm", "1", (True, True, True)) + _bout(
        "llm", "2", (True, False, True)
    )
    scope = MetricScope(
        scope_id="model:llm", kind="model", model_alias="llm", bout_ids=("1", "2")
    )
    prepared = SimpleNamespace(
        model_aliases=("llm",),
        common_scope=MetricScope(scope_id="common", kind="common", bout_ids=("1", "2")),
        model_scope=lambda alias: scope,
    )
    aggregated = AggregatedAnalysis(cases, cases, (), ())
    config = SimpleNamespace(
        llm=SimpleNamespace(alias="llm"), decision=SimpleNamespace(alias="decision")
    )

    result = calculate_bootstrap(prepared, aggregated, config, replications=100)

    assert result.agreement_intervals[0].point_estimate.status == "ok"
    assert result.agreement_intervals[0].confidence_interval.status == "ok"
    assert result.comparison.difference_percentage_points.status == "not_applicable"
    assert result.comparison.confidence_interval.status == "not_applicable"
