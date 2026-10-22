"""Aggregate normalized votes without changing stage-three eligibility.

The module deliberately keeps diagnostic rows for every selected
``model x bout x collegium`` combination.  Separate eligible collections make
it difficult for later metric functions to accidentally use a locally complete
collegium from a globally incomplete planned block.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.experiment.schemas import Collegium, Player
from src.metrics import COLLEGIA
from src.metrics.contracts import NormalizedVote
from src.metrics.preparation import PreparedAnalysis, PreparedBlock


@dataclass(frozen=True, slots=True)
class VoteTally:
    """Observed votes and their majority, if the observed count is odd."""

    player_1: int
    player_2: int
    majority: Player | None

    @property
    def total(self) -> int:
        return self.player_1 + self.player_2

    @property
    def player_1_share(self) -> float | None:
        return self.player_1 / self.total if self.total else None


@dataclass(frozen=True, slots=True)
class AggregatedCase:
    """One diagnostic row for a ``model x bout x collegium`` combination."""

    model_alias: str
    bout_id: str
    collegium: Collegium
    human_votes: tuple[NormalizedVote, ...]
    human_tally: VoteTally
    original_complete: bool
    original_votes: tuple[NormalizedVote, ...]
    original_tally: VoteTally
    original_agreement: bool | None
    swapped_complete: bool
    swapped_votes: tuple[NormalizedVote, ...]
    swapped_tally: VoteTally
    position_eligible: bool
    position_winner_changed: bool | None
    position_vote_share_change: float | None

    @property
    def original_probabilities(self) -> tuple[float | None, ...]:
        """Provider probabilities in repeat order; missing values stay missing."""

        return tuple(vote.probability_player_1 for vote in self.original_votes)

    @property
    def swapped_probabilities(self) -> tuple[float | None, ...]:
        """Mapped-back provider probabilities in repeat order."""

        return tuple(vote.probability_player_1 for vote in self.swapped_votes)


@dataclass(frozen=True, slots=True)
class AggregatedBout:
    """Descriptive all-vote summary for an eligible original bout."""

    model_alias: str
    bout_id: str
    human_votes: tuple[NormalizedVote, ...]
    human_tally: VoteTally
    model_votes: tuple[NormalizedVote, ...]
    model_tally: VoteTally
    majority_agreement: bool


@dataclass(frozen=True, slots=True)
class AggregatedAnalysis:
    """Stage-four output, split into metric-safe and diagnostic collections."""

    case_diagnostics: tuple[AggregatedCase, ...]
    original_cases: tuple[AggregatedCase, ...]
    position_cases: tuple[AggregatedCase, ...]
    bout_summaries: tuple[AggregatedBout, ...]


def _tally(votes: tuple[NormalizedVote, ...]) -> VoteTally:
    player_1 = sum(vote.winner == "player_1" for vote in votes)
    player_2 = len(votes) - player_1
    majority: Player | None = None
    if len(votes) % 2 == 1:
        if player_1 > player_2:
            majority = "player_1"
        elif player_2 > player_1:
            majority = "player_2"
    return VoteTally(player_1=player_1, player_2=player_2, majority=majority)


def _collegium_votes(
    block: PreparedBlock, collegium: Collegium
) -> tuple[NormalizedVote, ...]:
    return tuple(vote for vote in block.votes if vote.collegium == collegium)


def aggregate_analysis_data(prepared: PreparedAnalysis) -> AggregatedAnalysis:
    """Build case and bout aggregates from normalized stage-three data."""

    humans_by_case = {
        (bout_id, collegium): tuple(
            vote
            for vote in prepared.human_votes
            if vote.bout_id == bout_id and vote.collegium == collegium
        )
        for bout_id in prepared.selected_bout_ids
        for collegium in COLLEGIA
    }

    diagnostics: list[AggregatedCase] = []
    original_cases: list[AggregatedCase] = []
    position_cases: list[AggregatedCase] = []
    bout_summaries: list[AggregatedBout] = []

    for model_alias in prepared.model_aliases:
        for bout_id in prepared.selected_bout_ids:
            original = prepared.block(model_alias, bout_id, "original")
            swapped = prepared.block(model_alias, bout_id, "swapped")
            bout_human_votes: list[NormalizedVote] = []

            for collegium in COLLEGIA:
                human_votes = humans_by_case[(bout_id, collegium)]
                original_votes = _collegium_votes(original, collegium)
                swapped_votes = _collegium_votes(swapped, collegium)
                human_tally = _tally(human_votes)
                original_tally = _tally(original_votes)
                swapped_tally = _tally(swapped_votes)
                position_eligible = original.complete and swapped.complete

                case = AggregatedCase(
                    model_alias=model_alias,
                    bout_id=bout_id,
                    collegium=collegium,
                    human_votes=human_votes,
                    human_tally=human_tally,
                    original_complete=original.complete,
                    original_votes=original_votes,
                    original_tally=original_tally,
                    original_agreement=(
                        original_tally.majority == human_tally.majority
                        if original.complete
                        else None
                    ),
                    swapped_complete=swapped.complete,
                    swapped_votes=swapped_votes,
                    swapped_tally=swapped_tally,
                    position_eligible=position_eligible,
                    position_winner_changed=(
                        original_tally.majority != swapped_tally.majority
                        if position_eligible
                        else None
                    ),
                    position_vote_share_change=(
                        abs(
                            original_tally.player_1_share
                            - swapped_tally.player_1_share
                        )
                        if position_eligible
                        else None
                    ),
                )
                diagnostics.append(case)
                bout_human_votes.extend(human_votes)
                if original.complete:
                    original_cases.append(case)
                if position_eligible:
                    position_cases.append(case)

            if original.complete:
                human_votes = tuple(bout_human_votes)
                human_tally = _tally(human_votes)
                model_tally = _tally(original.votes)
                # Complete blocks contain an odd number of votes, so both majorities exist.
                assert human_tally.majority is not None
                assert model_tally.majority is not None
                bout_summaries.append(
                    AggregatedBout(
                        model_alias=model_alias,
                        bout_id=bout_id,
                        human_votes=human_votes,
                        human_tally=human_tally,
                        model_votes=original.votes,
                        model_tally=model_tally,
                        majority_agreement=(
                            model_tally.majority == human_tally.majority
                        ),
                    )
                )

    return AggregatedAnalysis(
        case_diagnostics=tuple(diagnostics),
        original_cases=tuple(original_cases),
        position_cases=tuple(position_cases),
        bout_summaries=tuple(bout_summaries),
    )


# Short alias for pipeline callers.
aggregate_votes = aggregate_analysis_data


__all__ = [
    "AggregatedAnalysis",
    "AggregatedBout",
    "AggregatedCase",
    "VoteTally",
    "aggregate_analysis_data",
    "aggregate_votes",
]
