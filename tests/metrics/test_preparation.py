from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.experiment.schemas import (
    CallRecord,
    JudgeResponse,
    PlayerProbabilities,
    TaskKey,
    load_transcript,
)
from src.experiment.storage import PlannedTaskIdentity
from src.metrics import COLLEGIA
from src.metrics.loading import SelectedTranscript
from src.metrics.preparation import prepare_analysis_data


MODELS = ("model-a", "model-b")
BOUTS = ("1", "2")
VARIANTS = ("original", "swapped")


def _inputs():
    planned = tuple(
        PlannedTaskIdentity(
            model_alias=model,
            bout_id=bout,
            variant=variant,
            collegium=collegium,
            repeat_index=repeat,
        )
        for model in MODELS
        for bout in BOUTS
        for variant in VARIANTS
        for collegium in COLLEGIA
        for repeat in (1, 2, 3)
    )
    omitted = {
        ("model-a", "2", "original", "ownership", 3),
        ("model-b", "1", "swapped", "negotiation", 2),
    }
    records = []
    for task in planned:
        identity = (
            task.model_alias,
            task.bout_id,
            task.variant,
            task.collegium,
            task.repeat_index,
        )
        if identity in omitted:
            continue
        probabilities = PlayerProbabilities(player_1=0.8, player_2=0.2)
        if identity == ("model-a", "1", "original", "hiring", 2):
            probabilities = None
        records.append(
            CallRecord.model_construct(
                task=TaskKey(run_id="run-1", **task.model_dump()),
                status="success",
                response=JudgeResponse(
                    winner="player_1",
                    probabilities=probabilities,
                    raw_response={},
                ),
            )
        )

    transcripts = tuple(
        SelectedTranscript(
            path=path,
            transcript=load_transcript(path),
        )
        for path in (
            Path("data/pilot_dataset/fub_video_transcript.json"),
            Path("data/pilot_dataset/fub_video_transcript2.json"),
        )
    )
    metadata = SimpleNamespace(
        selected_bout_ids=BOUTS,
        models=tuple(SimpleNamespace(alias=alias) for alias in MODELS),
        planned_tasks=planned,
    )
    return SimpleNamespace(
        metadata=metadata,
        latest_records=tuple(records),
        transcripts=transcripts,
    )


def test_whole_original_bout_completeness_is_independent_for_each_model():
    prepared = prepare_analysis_data(_inputs())

    assert prepared.model_scope("model-a").bout_ids == ("1",)
    assert prepared.model_scope("model-b").bout_ids == ("1", "2")
    assert prepared.common_scope.bout_ids == ("1",)

    incomplete = prepared.block("model-a", "2", "original")
    assert incomplete.successful_count == 8
    assert incomplete.required_count == 9
    assert not incomplete.complete
    # A complete collegium triple must not leak out of an incomplete whole bout.
    assert incomplete.cases == ()
    exclusion = next(
        item
        for item in prepared.exclusions
        if item.scope_id == "model:model-a" and item.bout_id == "2"
    )
    assert (exclusion.actual_count, exclusion.required_count) == (8, 9)


def test_incomplete_swapped_only_changes_position_scopes():
    prepared = prepare_analysis_data(_inputs())

    assert prepared.model_scope("model-b").bout_ids == ("1", "2")
    assert prepared.position_scope("model-a").bout_ids == ("1",)
    assert prepared.position_scope("model-b").bout_ids == ("2",)
    assert prepared.position_common_scope.bout_ids == ()
    assert any(
        item.scope_id == "position:model-b"
        and item.bout_id == "1"
        and item.reason_code == "incomplete_swapped"
        for item in prepared.exclusions
    )


def test_swapped_votes_and_probabilities_are_mapped_back_to_real_players():
    prepared = prepare_analysis_data(_inputs())
    vote = prepared.block("model-a", "1", "swapped").cases[0].model_votes[0]

    assert vote.winner == "player_2"
    assert vote.probability_player_1 == 0.2


def test_probability_scopes_require_all_three_real_probabilities_per_case():
    prepared = prepare_analysis_data(_inputs())
    individual = prepared.probability_scope("model-a")
    common = prepared.probability_scope("model-a", common=True)

    assert individual.base_scope_id == "model:model-a"
    assert individual.n_bouts == 1
    assert individual.n_cases == 2
    assert common.n_cases == 2
    assert ("1", "hiring") not in {
        (case.bout_id, case.collegium) for case in individual.cases
    }
    probability_exclusion = next(
        item
        for item in prepared.exclusions
        if item.scope_id == "probability:individual:model-a"
        and item.collegium == "hiring"
    )
    assert (
        probability_exclusion.actual_count,
        probability_exclusion.required_count,
    ) == (2, 3)


def test_normalized_human_votes_are_local_ordered_triples():
    prepared = prepare_analysis_data(_inputs())

    hiring = tuple(
        vote
        for vote in prepared.human_votes
        if vote.bout_id == "1" and vote.collegium == "hiring"
    )
    assert [vote.ordinal for vote in hiring] == [1, 2, 3]
    assert all(vote.source == "human" for vote in hiring)


def test_common_intersection_explains_peer_caused_exclusions_for_each_model():
    prepared = prepare_analysis_data(_inputs())

    assert any(
        item.scope_id == "common"
        and item.model_alias == "model-b"
        and item.bout_id == "2"
        and item.reason_code == "outside_common_scope"
        for item in prepared.exclusions
    )
    assert any(
        item.scope_id == "position:common"
        and item.model_alias == "model-a"
        and item.bout_id == "1"
        and item.reason_code == "outside_position_common_scope"
        for item in prepared.exclusions
    )
