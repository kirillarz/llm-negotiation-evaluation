"""Normalize saved votes and build auditable eligibility scopes.

This layer deliberately performs no metric aggregation.  Its output is the
single source of truth for which whole bouts and collegium cases later metric
functions may use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.experiment.schemas import CallRecord, Collegium, Player, Variant
from src.experiment.storage import PlannedTaskIdentity
from src.metrics import COLLEGIA
from src.metrics.contracts import (
    BoutCollegiumCase,
    Exclusion,
    MetricScope,
    NormalizedVote,
    ProbabilityCase,
    ProbabilityScope,
)
from src.metrics.loading import AnalysisInputs


HUMAN_COLLEGIUM_KEYS: Mapping[str, Collegium] = {
    "Нанимающаяся на работу": "hiring",
    "Отправляющая на переговоры": "negotiation",
    "Доверяющая собственность": "ownership",
}
VARIANTS: tuple[Variant, ...] = ("original", "swapped")
TaskIdentity = tuple[str, str, Variant, Collegium, int]
BlockIdentity = tuple[str, str, Variant]


@dataclass(frozen=True, slots=True)
class PreparedBlock:
    """Completeness and normalized successful votes for one model block."""

    model_alias: str
    bout_id: str
    variant: Variant
    planned_count: int
    successful_count: int
    required_count: int
    complete: bool
    votes: tuple[NormalizedVote, ...]
    cases: tuple[BoutCollegiumCase, ...]


@dataclass(frozen=True, slots=True)
class PreparedAnalysis:
    """Normalized data and every eligibility denominator required downstream."""

    selected_bout_ids: tuple[str, ...]
    model_aliases: tuple[str, ...]
    human_votes: tuple[NormalizedVote, ...]
    blocks: tuple[PreparedBlock, ...]
    model_scopes: tuple[MetricScope, ...]
    common_scope: MetricScope
    position_scopes: tuple[MetricScope, ...]
    position_common_scope: MetricScope
    probability_scopes: tuple[ProbabilityScope, ...]
    exclusions: tuple[Exclusion, ...]

    def block(
        self, model_alias: str, bout_id: str, variant: Variant
    ) -> PreparedBlock:
        for block in self.blocks:
            if (
                block.model_alias == model_alias
                and block.bout_id == bout_id
                and block.variant == variant
            ):
                return block
        raise KeyError((model_alias, bout_id, variant))

    def model_scope(self, model_alias: str) -> MetricScope:
        return _find_model_item(self.model_scopes, model_alias)

    def position_scope(self, model_alias: str) -> MetricScope:
        return _find_model_item(self.position_scopes, model_alias)

    def probability_scope(
        self, model_alias: str, *, common: bool = False
    ) -> ProbabilityScope:
        base = "common" if common else f"model:{model_alias}"
        for scope in self.probability_scopes:
            if scope.model_alias == model_alias and scope.base_scope_id == base:
                return scope
        raise KeyError((model_alias, base))


def _find_model_item(items, model_alias: str):
    for item in items:
        if item.model_alias == model_alias:
            return item
    raise KeyError(model_alias)


def _task_identity(task: PlannedTaskIdentity) -> TaskIdentity:
    return (
        task.model_alias,
        task.bout_id,
        task.variant,
        task.collegium,
        task.repeat_index,
    )


def _record_identity(record: CallRecord) -> TaskIdentity:
    task = record.task
    return (
        task.model_alias,
        task.bout_id,
        task.variant,
        task.collegium,
        task.repeat_index,
    )


def _original_player(player: Player, variant: Variant) -> Player:
    if variant == "original":
        return player
    return "player_2" if player == "player_1" else "player_1"


def _human_votes(inputs: AnalysisInputs) -> tuple[NormalizedVote, ...]:
    normalized: list[NormalizedVote] = []
    for selected in inputs.transcripts:
        grouped: dict[Collegium, list] = {collegium: [] for collegium in COLLEGIA}
        for vote in selected.transcript.human_votes:
            grouped[HUMAN_COLLEGIUM_KEYS[vote.collegia]].append(vote)
        for collegium in COLLEGIA:
            for ordinal, vote in enumerate(
                sorted(grouped[collegium], key=lambda item: item.judge_id), start=1
            ):
                normalized.append(
                    NormalizedVote(
                        source="human",
                        bout_id=selected.transcript.id,
                        collegium=collegium,
                        ordinal=ordinal,
                        winner=vote.decision,
                    )
                )
    return tuple(normalized)


def _model_vote(record: CallRecord) -> NormalizedVote:
    # Only callers that selected a successful latest revision may enter here.
    assert record.response is not None
    task = record.task
    probability = record.response.probabilities
    probability_player_1 = None
    if probability is not None:
        probability_player_1 = (
            probability.player_1
            if task.variant == "original"
            else probability.player_2
        )
    return NormalizedVote(
        source="model",
        bout_id=task.bout_id,
        collegium=task.collegium,
        ordinal=task.repeat_index,
        winner=_original_player(record.response.winner, task.variant),
        model_alias=task.model_alias,
        variant=task.variant,
        probability_player_1=probability_player_1,
    )


def _scope_bouts(
    selected_bout_ids: tuple[str, ...], complete: Mapping[str, set[str]]
) -> tuple[str, ...]:
    if not complete:
        return ()
    intersection = set.intersection(*(set(ids) for ids in complete.values()))
    return tuple(bout_id for bout_id in selected_bout_ids if bout_id in intersection)


def prepare_analysis_data(inputs: AnalysisInputs) -> PreparedAnalysis:
    """Normalize latest revisions and construct all stage-three scopes."""

    metadata = inputs.metadata
    selected_bout_ids = metadata.selected_bout_ids
    model_aliases = tuple(model.alias for model in metadata.models)
    planned = {_task_identity(task) for task in metadata.planned_tasks}
    repeat_indices = tuple(
        sorted({task.repeat_index for task in metadata.planned_tasks})
    )
    if repeat_indices not in {(1,), (1, 2, 3)}:
        raise ValueError(
            "planned model votes must use repeat indices (1,) or (1, 2, 3)"
        )
    required_votes_per_block = len(COLLEGIA) * len(repeat_indices)
    latest = {_record_identity(record): record for record in inputs.latest_records}
    humans = _human_votes(inputs)
    humans_by_case = {
        (bout_id, collegium): tuple(
            vote
            for vote in humans
            if vote.bout_id == bout_id and vote.collegium == collegium
        )
        for bout_id in selected_bout_ids
        for collegium in COLLEGIA
    }

    blocks: list[PreparedBlock] = []
    block_index: dict[BlockIdentity, PreparedBlock] = {}
    for model_alias in model_aliases:
        for bout_id in selected_bout_ids:
            for variant in VARIANTS:
                expected = {
                    (model_alias, bout_id, variant, collegium, repeat_index)
                    for collegium in COLLEGIA
                    for repeat_index in repeat_indices
                }
                planned_count = len(expected & planned)
                records = [
                    latest[identity]
                    for identity in expected
                    if identity in latest and latest[identity].status == "success"
                ]
                votes = tuple(
                    sorted(
                        (_model_vote(record) for record in records),
                        key=lambda vote: (COLLEGIA.index(vote.collegium), vote.ordinal),
                    )
                )
                complete = (
                    planned_count == required_votes_per_block
                    and len(votes) == required_votes_per_block
                )
                cases: tuple[BoutCollegiumCase, ...] = ()
                if complete:
                    cases = tuple(
                        BoutCollegiumCase(
                            bout_id=bout_id,
                            collegium=collegium,
                            model_alias=model_alias,
                            variant=variant,
                            model_votes=tuple(
                                vote for vote in votes if vote.collegium == collegium
                            ),
                            human_votes=humans_by_case[(bout_id, collegium)],
                        )
                        for collegium in COLLEGIA
                    )
                block = PreparedBlock(
                    model_alias=model_alias,
                    bout_id=bout_id,
                    variant=variant,
                    planned_count=planned_count,
                    successful_count=len(votes),
                    required_count=required_votes_per_block,
                    complete=complete,
                    votes=votes,
                    cases=cases,
                )
                blocks.append(block)
                block_index[(model_alias, bout_id, variant)] = block

    complete_original: dict[str, set[str]] = {}
    complete_position: dict[str, set[str]] = {}
    model_scopes: list[MetricScope] = []
    position_scopes: list[MetricScope] = []
    exclusions: list[Exclusion] = []
    for model_alias in model_aliases:
        originals = {
            bout_id
            for bout_id in selected_bout_ids
            if block_index[(model_alias, bout_id, "original")].complete
        }
        positions = {
            bout_id
            for bout_id in originals
            if block_index[(model_alias, bout_id, "swapped")].complete
        }
        complete_original[model_alias] = originals
        complete_position[model_alias] = positions
        model_scopes.append(
            MetricScope(
                scope_id=f"model:{model_alias}",
                kind="model",
                model_alias=model_alias,
                bout_ids=tuple(b for b in selected_bout_ids if b in originals),
            )
        )
        position_scopes.append(
            MetricScope(
                scope_id=f"position:{model_alias}",
                kind="position",
                model_alias=model_alias,
                bout_ids=tuple(b for b in selected_bout_ids if b in positions),
            )
        )
        for bout_id in selected_bout_ids:
            original = block_index[(model_alias, bout_id, "original")]
            if not original.complete:
                exclusions.append(
                    _block_exclusion(
                        "model:" + model_alias,
                        original,
                        "incomplete_original",
                    )
                )
            if bout_id not in positions:
                original_ok = original.complete
                deficient = block_index[
                    (model_alias, bout_id, "swapped" if original_ok else "original")
                ]
                code = "incomplete_swapped" if original_ok else "incomplete_original"
                exclusions.append(
                    _block_exclusion("position:" + model_alias, deficient, code)
                )

    common_bouts = _scope_bouts(selected_bout_ids, complete_original)
    position_common_bouts = _scope_bouts(selected_bout_ids, complete_position)
    common_scope = MetricScope(
        scope_id="common", kind="common", bout_ids=common_bouts
    )
    position_common_scope = MetricScope(
        scope_id="position:common",
        kind="position_common",
        bout_ids=position_common_bouts,
    )

    for model_alias in model_aliases:
        for bout_id in selected_bout_ids:
            if bout_id not in common_bouts:
                original = block_index[(model_alias, bout_id, "original")]
                if original.complete:
                    exclusions.append(
                        Exclusion(
                            scope_id="common",
                            bout_id=bout_id,
                            model_alias=model_alias,
                            variant="original",
                            reason_code="outside_common_scope",
                            reason=(
                                "At least one other participating model has an "
                                "incomplete original block"
                            ),
                            actual_count=original.successful_count,
                            required_count=original.required_count,
                        )
                    )
                else:
                    exclusions.append(
                        _block_exclusion("common", original, "incomplete_original")
                    )
            if bout_id not in position_common_bouts:
                original = block_index[(model_alias, bout_id, "original")]
                swapped = block_index[(model_alias, bout_id, "swapped")]
                if original.complete and swapped.complete:
                    exclusions.append(
                        Exclusion(
                            scope_id="position:common",
                            bout_id=bout_id,
                            model_alias=model_alias,
                            reason_code="outside_position_common_scope",
                            reason=(
                                "At least one other participating model has an "
                                "incomplete original or swapped block"
                            ),
                            actual_count=(
                                original.successful_count + swapped.successful_count
                            ),
                            required_count=(
                                original.required_count + swapped.required_count
                            ),
                        )
                    )
                else:
                    deficient = swapped if original.complete else original
                    code = (
                        "incomplete_swapped"
                        if original.complete
                        else "incomplete_original"
                    )
                    exclusions.append(
                        _block_exclusion("position:common", deficient, code)
                    )

    probability_scopes: list[ProbabilityScope] = []
    for model_alias in model_aliases:
        for base_scope_id, bout_ids in (
            (
                f"model:{model_alias}",
                tuple(
                    bout_id
                    for bout_id in selected_bout_ids
                    if bout_id in complete_original[model_alias]
                ),
            ),
            ("common", common_bouts),
        ):
            eligible_cases: list[ProbabilityCase] = []
            scope_label = "common" if base_scope_id == "common" else "individual"
            probability_scope_id = f"probability:{scope_label}:{model_alias}"
            for bout_id in bout_ids:
                original = block_index[(model_alias, bout_id, "original")]
                for case in original.cases:
                    actual = sum(
                        vote.probability_player_1 is not None
                        for vote in case.model_votes
                    )
                    required = len(case.model_votes)
                    if actual == required:
                        eligible_cases.append(
                            ProbabilityCase(bout_id=bout_id, collegium=case.collegium)
                        )
                    else:
                        exclusions.append(
                            Exclusion(
                                scope_id=probability_scope_id,
                                bout_id=bout_id,
                                model_alias=model_alias,
                                variant="original",
                                collegium=case.collegium,
                                metric="probabilistic_metrics",
                                reason_code="incomplete_probabilities",
                                reason=(
                                    f"Only {actual} of {required} original votes contain "
                                    "provider probabilities"
                                ),
                                actual_count=actual,
                                required_count=required,
                            )
                        )
            probability_scopes.append(
                ProbabilityScope(
                    scope_id=probability_scope_id,
                    base_scope_id=base_scope_id,
                    model_alias=model_alias,
                    cases=tuple(eligible_cases),
                )
            )

    return PreparedAnalysis(
        selected_bout_ids=selected_bout_ids,
        model_aliases=model_aliases,
        human_votes=humans,
        blocks=tuple(blocks),
        model_scopes=tuple(model_scopes),
        common_scope=common_scope,
        position_scopes=tuple(position_scopes),
        position_common_scope=position_common_scope,
        probability_scopes=tuple(probability_scopes),
        exclusions=tuple(exclusions),
    )


def _block_exclusion(
    scope_id: str, block: PreparedBlock, reason_code: str
) -> Exclusion:
    label = "original" if block.variant == "original" else "swapped"
    return Exclusion(
        scope_id=scope_id,
        bout_id=block.bout_id,
        model_alias=block.model_alias,
        variant=block.variant,
        reason_code=reason_code,
        reason=(
            f"Only {block.successful_count} of {block.required_count} required "
            f"{label} votes succeeded ({block.planned_count} planned)"
        ),
        actual_count=block.successful_count,
        required_count=block.required_count,
    )


# Short alias for pipeline callers.
prepare_inputs = prepare_analysis_data


__all__ = [
    "HUMAN_COLLEGIUM_KEYS",
    "PreparedAnalysis",
    "PreparedBlock",
    "REQUIRED_VOTES_PER_BLOCK",
    "prepare_analysis_data",
    "prepare_inputs",
]
