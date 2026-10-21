from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.metrics.aggregation import AggregatedAnalysis
from src.metrics.contracts import InferenceTestResult, MetricScope
from src.metrics.inference import apply_holm, calculate_inference
from tests.metrics.test_calculations import _case


COLLEGIA = ("hiring", "negotiation", "ownership")


def _bout(alias: str, bout_id: str, *, correct: bool, stable: bool = True, flip: bool = False):
    cases = []
    human_majorities = ("player_1", "player_2", "player_1")
    for collegium, human_majority in zip(COLLEGIA, human_majorities, strict=True):
        other = "player_2" if human_majority == "player_1" else "player_1"
        human = [human_majority] * 3
        if correct:
            model = [human_majority] * 3 if stable else [human_majority, human_majority, other]
        else:
            model = [other] * 3 if stable else [other, other, human_majority]
        model_majority = model[0]
        flipped = "player_2" if model_majority == "player_1" else "player_1"
        swapped = [flipped] * 3 if flip else model
        cases.append(
            _case(
                bout_id,
                collegium,
                human,
                model,
                model_alias=alias,
                swapped=swapped,
            )
        )
    return tuple(cases)


def _prepared_and_aggregated(n_bouts: int, *, one_class: bool = False):
    ids = tuple(str(index) for index in range(n_bouts))
    cases = []
    for bout_id in ids:
        if one_class:
            for alias in ("llm", "decision"):
                cases.extend(
                    _case(
                        bout_id,
                        collegium,
                        ["player_1"] * 3,
                        ["player_1"] * 3,
                        model_alias=alias,
                    )
                    for collegium in COLLEGIA
                )
        else:
            cases.extend(_bout("llm", bout_id, correct=False, stable=False, flip=True))
            cases.extend(_bout("decision", bout_id, correct=True, stable=True, flip=False))
    scopes = {
        alias: MetricScope(
            scope_id=f"model:{alias}", kind="model", model_alias=alias, bout_ids=ids
        )
        for alias in ("llm", "decision")
    }
    position_scopes = {
        alias: MetricScope(
            scope_id=f"position:{alias}", kind="position", model_alias=alias, bout_ids=ids
        )
        for alias in ("llm", "decision")
    }
    prepared = SimpleNamespace(
        model_aliases=("llm", "decision"),
        common_scope=MetricScope(scope_id="common", kind="common", bout_ids=ids),
        position_common_scope=MetricScope(
            scope_id="position:common", kind="position_common", bout_ids=ids
        ),
        model_scope=lambda alias: scopes[alias],
        position_scope=lambda alias: position_scopes[alias],
    )
    all_cases = tuple(cases)
    aggregated = AggregatedAnalysis(all_cases, all_cases, all_cases, ())
    config = SimpleNamespace(
        llm=SimpleNamespace(alias="llm"),
        decision=SimpleNamespace(alias="decision"),
    )
    return prepared, aggregated, config


def _test(test_id: str, raw: float | None):
    status = "ok" if raw is not None else "not_applicable"
    return InferenceTestResult(
        test_id=test_id,
        hypothesis_id="H1",
        metric="m",
        scope_id="s",
        null_hypothesis="null",
        alternative="greater",
        status=status,
        effect=1.0 if raw is not None else None,
        raw_p_value=raw,
        adjusted_p_value=raw,
        alpha=0.05,
        method="test",
        exact=True,
        permutations=8 if raw is not None else 0,
        seed=42,
        n_bouts=3,
        n_cases=9,
        reason=None if raw is not None else "unavailable",
    )


def test_exact_cluster_permutations_keep_collegia_together():
    prepared, aggregated, config = _prepared_and_aggregated(3)

    result = calculate_inference(
        prepared, aggregated, config, bootstrap_replications=50, permutations=10
    )

    accuracy = next(test for test in result.tests if test.test_id == "H1:decision:agreement_rate")
    assert accuracy.exact is True
    assert accuracy.permutations == 8
    assert accuracy.raw_p_value == pytest.approx(1 / 8)
    assert accuracy.n_bouts == 3 and accuracy.n_cases == 9
    assert len(result.tests) == 7
    assert {
        item.metric for item in result.intervals
    } >= {
        "cohens_kappa",
        "pairwise_repeatability",
        "label_flip_rate",
        "cohens_kappa_difference",
        "pairwise_repeatability_difference_percentage_points",
        "label_flip_rate_difference_percentage_points",
    }


def test_paired_h2_and_h3_use_common_bout_blocks():
    prepared, aggregated, config = _prepared_and_aggregated(3)

    result = calculate_inference(
        prepared, aggregated, config, bootstrap_replications=50, permutations=10
    )

    by_id = {test.test_id: test for test in result.tests}
    assert by_id["H2:accuracy_difference"].effect == pytest.approx(100.0)
    assert by_id["H3:pairwise_repeatability_difference"].effect == pytest.approx(200 / 3)
    assert by_id["H3:label_flip_rate_difference"].effect == pytest.approx(-100.0)
    assert all(by_id[test_id].permutations == 8 for test_id in (
        "H2:accuracy_difference",
        "H3:pairwise_repeatability_difference",
        "H3:label_flip_rate_difference",
    ))


def test_holm_uses_fixed_family_size_including_unavailable_test():
    tests = tuple(
        _test(f"test-{index}", raw)
        for index, raw in enumerate((0.001, 0.01, 0.02, 0.2, 0.4, 0.8, None))
    )

    corrected = apply_holm(tests)

    assert corrected[0].adjusted_p_value == pytest.approx(0.007)
    assert corrected[1].adjusted_p_value == pytest.approx(0.06)
    assert corrected[-1].adjusted_p_value is None


def test_undefined_observed_kappa_is_explicit_na():
    prepared, aggregated, config = _prepared_and_aggregated(2, one_class=True)

    result = calculate_inference(
        prepared, aggregated, config, bootstrap_replications=10, permutations=10
    )

    kappa = next(test for test in result.tests if test.test_id == "H1:llm:cohens_kappa")
    assert kappa.status == "not_applicable"
    assert kappa.raw_p_value is None and kappa.adjusted_p_value is None
    assert "undefined" in kappa.reason
    decision = next(item for item in result.decisions if item.decision_id == "H1:llm")
    assert decision.status == "not_evaluable"


def test_monte_carlo_mode_is_reproducible_above_exact_threshold():
    prepared, aggregated, config = _prepared_and_aggregated(21)

    first = calculate_inference(
        prepared, aggregated, config,
        bootstrap_replications=10, permutations=50, seed=17,
    )
    second = calculate_inference(
        prepared, aggregated, config,
        bootstrap_replications=10, permutations=50, seed=17,
    )

    assert first == second
    assert all(test.exact is False and test.permutations == 50 for test in first.tests)


def test_kappa_interval_records_valid_bootstrap_replications():
    prepared, aggregated, config = _prepared_and_aggregated(3)

    result = calculate_inference(
        prepared, aggregated, config, bootstrap_replications=40, permutations=10
    )

    interval = next(
        item.confidence_interval
        for item in result.intervals
        if item.interval_id == "decision:cohens_kappa"
    )
    assert interval.status == "ok"
    assert 0 < interval.valid_replications <= interval.replications == 40
