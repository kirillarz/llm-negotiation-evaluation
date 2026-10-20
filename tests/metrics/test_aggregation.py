from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.metrics import COLLEGIA
from src.metrics.aggregation import aggregate_analysis_data
from src.metrics.contracts import NormalizedVote
from src.metrics.preparation import PreparedBlock


def _human(bout_id, collegium, winners):
    return tuple(
        NormalizedVote(
            source="human",
            bout_id=bout_id,
            collegium=collegium,
            ordinal=ordinal,
            winner=winner,
        )
        for ordinal, winner in enumerate(winners, start=1)
    )


def _model(model, bout_id, variant, collegium, winners, probabilities=None):
    if probabilities is None:
        probabilities = [None] * len(winners)
    return tuple(
        NormalizedVote(
            source="model",
            model_alias=model,
            bout_id=bout_id,
            variant=variant,
            collegium=collegium,
            ordinal=ordinal,
            winner=winner,
            probability_player_1=probability,
        )
        for ordinal, (winner, probability) in enumerate(
            zip(winners, probabilities, strict=True), start=1
        )
    )


def _prepared():
    model = "model-a"
    human_patterns = {
        "hiring": ("player_1", "player_1", "player_2"),
        "negotiation": ("player_2", "player_2", "player_1"),
        "ownership": ("player_1", "player_1", "player_1"),
    }
    original_patterns = {
        "hiring": ("player_1", "player_2", "player_2"),
        "negotiation": ("player_2", "player_2", "player_2"),
        "ownership": ("player_1", "player_1", "player_2"),
    }
    swapped_patterns = {
        "hiring": ("player_2", "player_2", "player_1"),
        "negotiation": ("player_1", "player_1", "player_2"),
        "ownership": ("player_1", "player_1", "player_1"),
    }
    humans = tuple(
        vote
        for bout_id in ("1", "2")
        for collegium in COLLEGIA
        for vote in _human(bout_id, collegium, human_patterns[collegium])
    )
    blocks = []
    for bout_id in ("1", "2"):
        for variant, patterns in (
            ("original", original_patterns),
            ("swapped", swapped_patterns),
        ):
            votes = tuple(
                vote
                for collegium in COLLEGIA
                for vote in _model(
                    model,
                    bout_id,
                    variant,
                    collegium,
                    patterns[collegium],
                    (0.8, None, 0.2),
                )
            )
            if bout_id == "2" and variant == "original":
                votes = votes[:-1]
            complete = len(votes) == 9
            blocks.append(
                PreparedBlock(
                    model_alias=model,
                    bout_id=bout_id,
                    variant=variant,
                    planned_count=9,
                    successful_count=len(votes),
                    required_count=9,
                    complete=complete,
                    votes=votes,
                    cases=(),
                )
            )
    block_index = {
        (block.model_alias, block.bout_id, block.variant): block for block in blocks
    }
    return SimpleNamespace(
        selected_bout_ids=("1", "2"),
        model_aliases=(model,),
        human_votes=humans,
        block=lambda model_alias, bout_id, variant: block_index[
            (model_alias, bout_id, variant)
        ],
    )


def _case(result, bout_id, collegium):
    return next(
        case
        for case in result.case_diagnostics
        if case.bout_id == bout_id and case.collegium == collegium
    )


def test_complete_case_keeps_votes_scores_majorities_and_probabilities():
    result = aggregate_analysis_data(_prepared())
    case = _case(result, "1", "hiring")

    assert [vote.winner for vote in case.human_votes] == [
        "player_1",
        "player_1",
        "player_2",
    ]
    assert [vote.winner for vote in case.original_votes] == [
        "player_1",
        "player_2",
        "player_2",
    ]
    assert (case.human_tally.player_1, case.human_tally.player_2) == (2, 1)
    assert (case.original_tally.player_1, case.original_tally.player_2) == (1, 2)
    assert case.human_tally.majority == "player_1"
    assert case.original_tally.majority == "player_2"
    assert case.original_agreement is False
    assert case.original_probabilities == (0.8, None, 0.2)


def test_position_aggregation_uses_mapped_back_votes_and_vote_share():
    result = aggregate_analysis_data(_prepared())
    negotiation = _case(result, "1", "negotiation")
    ownership = _case(result, "1", "ownership")

    assert negotiation.position_eligible
    assert negotiation.position_winner_changed is True
    assert negotiation.position_vote_share_change == pytest.approx(2 / 3)
    assert ownership.position_winner_changed is False
    assert ownership.position_vote_share_change == pytest.approx(1 / 3)


def test_nine_vote_summary_is_descriptive_and_separate_from_case_accuracy():
    result = aggregate_analysis_data(_prepared())

    assert len(result.original_cases) == 3
    assert [case.original_agreement for case in result.original_cases] == [
        False,
        True,
        True,
    ]
    assert len(result.bout_summaries) == 1
    summary = result.bout_summaries[0]
    assert (summary.human_tally.player_1, summary.human_tally.player_2) == (6, 3)
    assert (summary.model_tally.player_1, summary.model_tally.player_2) == (3, 6)
    assert summary.human_tally.majority == "player_1"
    assert summary.model_tally.majority == "player_2"
    assert summary.majority_agreement is False


def test_incomplete_block_is_diagnostic_only_even_if_local_triples_are_complete():
    result = aggregate_analysis_data(_prepared())
    incomplete_hiring = _case(result, "2", "hiring")
    incomplete_ownership = _case(result, "2", "ownership")

    assert len(result.case_diagnostics) == 6
    assert len(incomplete_hiring.original_votes) == 3
    assert incomplete_hiring.original_tally.majority == "player_2"
    assert incomplete_hiring.original_complete is False
    assert incomplete_hiring.original_agreement is None
    assert len(incomplete_ownership.original_votes) == 2
    assert incomplete_ownership.original_tally.majority is None
    assert not any(case.bout_id == "2" for case in result.original_cases)
    assert not any(case.bout_id == "2" for case in result.position_cases)
    assert not any(summary.bout_id == "2" for summary in result.bout_summaries)
