"""Strict internal contracts shared by metrics pipeline layers."""

from __future__ import annotations

from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.experiment.schemas import Collegium, Player, Probability, Variant


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveOrdinal = Annotated[int, Field(strict=True, ge=1, le=3)]
NonNegativeCount = Annotated[int, Field(strict=True, ge=0)]
VoteSource = Literal["human", "model"]
MetricStatus = Literal["ok", "not_applicable"]
ScopeKind = Literal[
    "model",
    "common",
    "position",
    "position_common",
    "probability",
]


class FrozenStrictModel(BaseModel):
    """Immutable model that rejects coercion and unknown fields."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class NormalizedVote(FrozenStrictModel):
    """One human or model vote normalized to the original player labels."""

    source: VoteSource
    bout_id: NonEmptyString
    collegium: Collegium
    ordinal: PositiveOrdinal
    winner: Player
    model_alias: NonEmptyString | None = None
    variant: Variant | None = None
    probability_player_1: Probability | None = None

    @model_validator(mode="after")
    def check_source_fields(self) -> Self:
        if self.source == "human":
            if self.model_alias is not None or self.variant is not None:
                raise ValueError("human votes cannot have model_alias or variant")
            if self.probability_player_1 is not None:
                raise ValueError("human votes cannot have provider probabilities")
        elif self.model_alias is None or self.variant is None:
            raise ValueError("model votes require model_alias and variant")
        return self


class BoutCollegiumCase(FrozenStrictModel):
    """A complete ``bout x collegium`` comparison for one model and variant."""

    bout_id: NonEmptyString
    collegium: Collegium
    model_alias: NonEmptyString
    variant: Variant
    model_votes: Annotated[tuple[NormalizedVote, ...], Field(min_length=3, max_length=3)]
    human_votes: Annotated[tuple[NormalizedVote, ...], Field(min_length=3, max_length=3)]

    @model_validator(mode="after")
    def check_votes_match_case(self) -> Self:
        expected_ordinals = [1, 2, 3]
        for source, votes in (("model", self.model_votes), ("human", self.human_votes)):
            if sorted(vote.ordinal for vote in votes) != expected_ordinals:
                raise ValueError(f"{source}_votes must contain ordinals 1, 2, 3")
            for vote in votes:
                if (
                    vote.source != source
                    or vote.bout_id != self.bout_id
                    or vote.collegium != self.collegium
                ):
                    raise ValueError(f"{source}_votes do not match the case identity")
                if source == "model" and (
                    vote.model_alias != self.model_alias or vote.variant != self.variant
                ):
                    raise ValueError("model_votes do not match the model or variant")
        return self


class MetricScope(FrozenStrictModel):
    """Named, auditable set of bouts used as a metric denominator."""

    scope_id: NonEmptyString
    kind: ScopeKind
    bout_ids: tuple[NonEmptyString, ...]
    model_alias: NonEmptyString | None = None

    @model_validator(mode="after")
    def check_scope(self) -> Self:
        if len(set(self.bout_ids)) != len(self.bout_ids):
            raise ValueError("scope bout_ids must be unique")
        if self.kind in {"model", "position", "probability"}:
            if self.model_alias is None:
                raise ValueError(f"{self.kind} scope requires model_alias")
        elif self.model_alias is not None:
            raise ValueError(f"{self.kind} scope cannot belong to one model")
        return self

    @property
    def n_bouts(self) -> int:
        return len(self.bout_ids)


class Exclusion(FrozenStrictModel):
    """Reason why a bout or collegium case is absent from a calculation."""

    scope_id: NonEmptyString
    bout_id: NonEmptyString
    reason_code: NonEmptyString
    reason: NonEmptyString
    model_alias: NonEmptyString | None = None
    variant: Variant | None = None
    collegium: Collegium | None = None
    metric: NonEmptyString | None = None
    actual_count: NonNegativeCount | None = None
    required_count: NonNegativeCount | None = None


class ProbabilityCase(FrozenStrictModel):
    """One complete ``bout x collegium`` case with three real probabilities."""

    bout_id: NonEmptyString
    collegium: Collegium


class ProbabilityScope(FrozenStrictModel):
    """Auditable case-level denominator for probabilistic metrics."""

    scope_id: NonEmptyString
    base_scope_id: NonEmptyString
    model_alias: NonEmptyString
    cases: tuple[ProbabilityCase, ...]

    @model_validator(mode="after")
    def check_cases(self) -> Self:
        identities = [(case.bout_id, case.collegium) for case in self.cases]
        if len(set(identities)) != len(identities):
            raise ValueError("probability scope cases must be unique")
        return self

    @property
    def bout_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(case.bout_id for case in self.cases))

    @property
    def n_bouts(self) -> int:
        return len(self.bout_ids)

    @property
    def n_cases(self) -> int:
        return len(self.cases)


class MetricResult(FrozenStrictModel):
    """One finite metric value or an explicit, explained N/A result."""

    metric: NonEmptyString
    status: MetricStatus
    value: float | None
    n_bouts: NonNegativeCount
    n_cases: NonNegativeCount
    n_votes: NonNegativeCount | None = None
    reason: NonEmptyString | None = None

    @model_validator(mode="after")
    def check_result(self) -> Self:
        if self.status == "ok":
            if self.value is None or not isfinite(self.value):
                raise ValueError("ok metric result requires a finite value")
            if self.reason is not None:
                raise ValueError("ok metric result cannot have an N/A reason")
        else:
            if self.value is not None:
                raise ValueError("not_applicable metric result must have value=null")
            if self.reason is None:
                raise ValueError("not_applicable metric result requires a reason")
        return self

    @classmethod
    def ok(
        cls,
        metric: str,
        value: float,
        *,
        n_bouts: int,
        n_cases: int,
        n_votes: int | None = None,
    ) -> Self:
        return cls(
            metric=metric,
            status="ok",
            value=value,
            n_bouts=n_bouts,
            n_cases=n_cases,
            n_votes=n_votes,
        )

    @classmethod
    def not_applicable(
        cls,
        metric: str,
        reason: str,
        *,
        n_bouts: int = 0,
        n_cases: int = 0,
        n_votes: int | None = None,
    ) -> Self:
        return cls(
            metric=metric,
            status="not_applicable",
            value=None,
            reason=reason,
            n_bouts=n_bouts,
            n_cases=n_cases,
            n_votes=n_votes,
        )
