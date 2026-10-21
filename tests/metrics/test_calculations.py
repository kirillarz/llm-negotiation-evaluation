from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.metrics import COLLEGIA
from src.metrics.aggregation import AggregatedAnalysis, AggregatedCase, VoteTally
from src.metrics.calculations import (
    agreement_by_collegium,
    agreement_rate,
    calculate_metrics,
    cohens_kappa,
    human_human_agreement,
    individual_judge_agreement,
    macro_f1,
    position_bias,
    probability_metrics,
    repeatability,
)
from src.metrics.contracts import MetricScope, NormalizedVote


def _votes(
    source,
    bout_id,
    collegium,
    winners,
    *,
    model=None,
    variant=None,
    probabilities=None,
):
    probabilities = probabilities or [None] * len(winners)
    return tuple(
        NormalizedVote(
            source=source,
            bout_id=bout_id,
            collegium=collegium,
            ordinal=index,
            winner=winner,
            model_alias=model,
            variant=variant,
            probability_player_1=probability,
        )
        for index, (winner, probability) in enumerate(
            zip(winners, probabilities, strict=True), start=1
        )
    )


def _tally(winners):
    player_1 = winners.count("player_1")
    player_2 = len(winners) - player_1
    majority = "player_1" if player_1 > player_2 else "player_2"
    return VoteTally(player_1, player_2, majority)


def _case(
    bout_id,
    collegium,
    human,
    model,
    *,
    model_alias="model-a",
    swapped=None,
    probabilities=None,
):
    swapped = model if swapped is None else swapped
    human_votes = _votes("human", bout_id, collegium, human)
    original_votes = _votes(
        "model",
        bout_id,
        collegium,
        model,
        model=model_alias,
        variant="original",
        probabilities=probabilities,
    )
    swapped_votes = _votes(
        "model",
        bout_id,
        collegium,
        swapped,
        model=model_alias,
        variant="swapped",
    )
    human_tally = _tally(human)
    original_tally = _tally(model)
    swapped_tally = _tally(swapped)
    return AggregatedCase(
        model_alias=model_alias,
        bout_id=bout_id,
        collegium=collegium,
        human_votes=human_votes,
        human_tally=human_tally,
        original_complete=True,
        original_votes=original_votes,
        original_tally=original_tally,
        original_agreement=original_tally.majority == human_tally.majority,
        swapped_complete=True,
        swapped_votes=swapped_votes,
        swapped_tally=swapped_tally,
        position_eligible=True,
        position_winner_changed=original_tally.majority != swapped_tally.majority,
        position_vote_share_change=abs(
            original_tally.player_1_share - swapped_tally.player_1_share
        ),
    )


def _six_cases():
    specifications = (
        ("1", "hiring", ["player_1"] * 3, ["player_1"] * 3),
        ("1", "negotiation", ["player_1"] * 2 + ["player_2"], ["player_2"] * 3),
        ("1", "ownership", ["player_2"] * 3, ["player_2"] * 2 + ["player_1"]),
        ("2", "hiring", ["player_2"] * 3, ["player_1"] * 2 + ["player_2"]),
        ("2", "negotiation", ["player_1"] * 3, ["player_1"] * 2 + ["player_2"]),
        ("2", "ownership", ["player_2"] * 2 + ["player_1"], ["player_2"] * 3),
    )
    return tuple(_case(*specification) for specification in specifications)


def test_agreement_uses_six_cases_not_eighteen_votes_and_splits_collegia():
    cases = _six_cases()

    result = agreement_rate(cases)
    by_collegium = agreement_by_collegium(cases)

    assert result.value == pytest.approx(4 / 6)
    assert (
        result.n_bouts,
        result.n_cases,
        result.n_model_votes,
        result.n_human_votes,
        result.n_comparisons,
    ) == (2, 6, 18, 18, 6)
    assert {key: value.n_cases for key, value in by_collegium.items()} == {
        "hiring": 2,
        "negotiation": 2,
        "ownership": 2,
    }
    assert by_collegium["hiring"].value == pytest.approx(1 / 2)


def test_cohen_kappa_and_macro_f1_use_fixed_player_classes():
    cases = _six_cases()

    assert cohens_kappa(cases).value == pytest.approx(1 / 3)
    assert macro_f1(cases).value == pytest.approx(2 / 3)


def test_individual_judge_metrics_use_cartesian_pairs_not_ordinals():
    case = _case(
        "1",
        "hiring",
        ["player_1", "player_1", "player_2"],
        ["player_1", "player_1", "player_2"],
    )

    result = individual_judge_agreement((case,))

    assert result.majority_to_individual.value == pytest.approx(2 / 3)
    assert (
        result.majority_to_individual.n_model_votes,
        result.majority_to_individual.n_human_votes,
        result.majority_to_individual.n_comparisons,
    ) == (3, 3, 3)
    assert result.all_pairs.value == pytest.approx(5 / 9)
    assert (
        result.all_pairs.n_model_votes,
        result.all_pairs.n_human_votes,
        result.all_pairs.n_comparisons,
    ) == (3, 3, 9)


@pytest.mark.parametrize(
    ("votes", "unanimity", "pairwise"),
    [
        (["player_1"] * 3, 1.0, 1.0),
        (["player_1", "player_1", "player_2"], 0.0, 1 / 3),
    ],
)
def test_repeatability_reference_patterns(votes, unanimity, pairwise):
    result = repeatability((_case("1", "hiring", votes, votes),))

    assert result.unanimity_rate.value == unanimity
    assert result.pairwise_repeatability.value == pytest.approx(pairwise)
    assert (
        result.unanimity_rate.n_model_votes,
        result.unanimity_rate.n_comparisons,
    ) == (3, None)
    assert (
        result.pairwise_repeatability.n_model_votes,
        result.pairwise_repeatability.n_comparisons,
    ) == (3, 3)


def test_human_pairwise_reference_patterns_and_fleiss_formula():
    cases = (
        _case("1", "hiring", ["player_1"] * 3, ["player_1"] * 3),
        _case(
            "2",
            "hiring",
            ["player_1", "player_1", "player_2"],
            ["player_1"] * 3,
        ),
    )

    result = human_human_agreement(cases)

    assert result.pairwise_agreement.value == pytest.approx(2 / 3)
    assert result.fleiss_kappa.value == pytest.approx(-0.2)
    assert (
        result.fleiss_kappa.n_human_votes,
        result.fleiss_kappa.n_comparisons,
    ) == (6, 6)


def test_position_metrics_use_mapped_back_aggregates_and_collegium_denominators():
    cases = (
        _case(
            "1",
            "hiring",
            ["player_1"] * 3,
            ["player_1"] * 2 + ["player_2"],
            swapped=["player_1"] * 3,
        ),
        _case(
            "1",
            "negotiation",
            ["player_2"] * 3,
            ["player_1"] * 3,
            swapped=["player_2"] * 2 + ["player_1"],
        ),
    )

    result = position_bias(cases)

    assert result.label_flip_rate.value == pytest.approx(1 / 2)
    assert result.vote_share_change.value == pytest.approx(1 / 2)
    assert result.by_collegium["hiring"][0].value == 0
    assert result.by_collegium["ownership"][0].status == "not_applicable"
    assert (
        result.label_flip_rate.n_model_votes,
        result.label_flip_rate.n_human_votes,
        result.label_flip_rate.n_comparisons,
    ) == (12, None, 2)


def test_probability_reference_values_and_coverage_exclude_incomplete_triples():
    complete = _case(
        "1",
        "hiring",
        ["player_1", "player_1", "player_2"],
        ["player_1"] * 3,
        probabilities=[0.8, 0.6, 0.7],
    )
    incomplete = _case(
        "1",
        "ownership",
        ["player_2"] * 3,
        ["player_2"] * 3,
        probabilities=[0.2, None, 0.3],
    )

    result = probability_metrics((complete, incomplete))

    assert result.brier_score.value == pytest.approx(0.09)
    assert result.brier_score_individual.value == pytest.approx(0.0966666666667)
    assert result.vote_share_mae.value == pytest.approx(1 / 30)
    assert result.vote_share_mae_individual.value == pytest.approx(7 / 90)
    assert result.brier_score.n_cases == 1
    assert (
        result.brier_score.n_model_votes,
        result.brier_score.n_human_votes,
        result.brier_score.n_comparisons,
    ) == (3, 3, 1)
    assert result.brier_score_individual.n_comparisons == 3
    assert result.coverage.value == pytest.approx(1 / 2)
    assert result.coverage.n_cases == 2
    assert (
        result.coverage.n_model_votes,
        result.coverage.n_human_votes,
        result.coverage.n_comparisons,
    ) == (3, None, None)


def test_empty_and_degenerate_metrics_are_explicit_na_without_nonfinite_values():
    empty = agreement_rate(())
    one_class = tuple(
        _case("1", collegium, ["player_1"] * 3, ["player_1"] * 3)
        for collegium in COLLEGIA
    )

    assert empty.status == "not_applicable" and empty.value is None and empty.reason
    assert (
        empty.n_model_votes,
        empty.n_human_votes,
        empty.n_comparisons,
    ) == (0, 0, 0)
    assert cohens_kappa(one_class).status == "not_applicable"
    assert macro_f1(one_class).status == "not_applicable"
    assert human_human_agreement(one_class).fleiss_kappa.status == "not_applicable"
    missing = probability_metrics(one_class)
    assert missing.brier_score.status == "not_applicable"
    assert missing.coverage.value == 0


def test_calculate_metrics_uses_individual_common_and_human_scope_denominators():
    cases_a = _six_cases()
    cases_b = tuple(
        _case(
            case.bout_id,
            case.collegium,
            [vote.winner for vote in case.human_votes],
            [vote.winner for vote in case.original_votes],
            model_alias="model-b",
        )
        for case in cases_a
        if case.bout_id == "1"
    )
    all_cases = cases_a + cases_b
    scopes = {
        "model-a": MetricScope(
            scope_id="model:model-a",
            kind="model",
            model_alias="model-a",
            bout_ids=("1", "2"),
        ),
        "model-b": MetricScope(
            scope_id="model:model-b",
            kind="model",
            model_alias="model-b",
            bout_ids=("1",),
        ),
    }
    position_scopes = {
        alias: MetricScope(
            scope_id=f"position:{alias}",
            kind="position",
            model_alias=alias,
            bout_ids=scope.bout_ids,
        )
        for alias, scope in scopes.items()
    }
    prepared = SimpleNamespace(
        model_aliases=("model-a", "model-b"),
        selected_bout_ids=("1", "2"),
        common_scope=MetricScope(scope_id="common", kind="common", bout_ids=("1",)),
        position_common_scope=MetricScope(
            scope_id="position:common", kind="position_common", bout_ids=("1",)
        ),
        model_scope=lambda alias: scopes[alias],
        position_scope=lambda alias: position_scopes[alias],
    )
    aggregated = AggregatedAnalysis(
        case_diagnostics=all_cases,
        original_cases=all_cases,
        position_cases=all_cases,
        bout_summaries=(),
    )

    result = calculate_metrics(prepared, aggregated)

    denominators = {
        item.scope_id: item.agreement.accuracy.n_bouts for item in result.model_scopes
    }
    assert denominators == {
        "model:model-a": 2,
        "common:model-a": 1,
        "model:model-b": 1,
        "common:model-b": 1,
    }
    human_denominators = {
        item.scope_id: item.metrics.pairwise_agreement.n_bouts
        for item in result.human_scopes
    }
    assert human_denominators == {
        "all_humans": 2,
        "humans:model:model-a": 2,
        "humans:model:model-b": 1,
        "humans:common": 1,
    }
