"""Shared data contracts for source transcripts and experiment execution."""

from __future__ import annotations

from collections import Counter
from datetime import date
from math import isclose
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    JsonValue,
    StringConstraints,
    model_validator,
)


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeStrictInt = Annotated[int, Field(strict=True, ge=0)]
Player = Literal["player_1", "player_2"]
Collegium = Literal["hiring", "negotiation", "ownership"]
Variant = Literal["original", "swapped"]
AttemptStatus = Literal["success", "failed"]
CallStatus = Literal["success", "failed"]
Collegia = Literal[
    "Нанимающаяся на работу",
    "Отправляющая на переговоры",
    "Доверяющая собственность",
]

EXPECTED_COLLEGIA_COUNTS: dict[str, int] = {
    "Нанимающаяся на работу": 3,
    "Отправляющая на переговоры": 3,
    "Доверяющая собственность": 3,
}


class StrictModel(BaseModel):
    """Base model that rejects coercion and fields outside the schema."""

    model_config = ConfigDict(extra="forbid", strict=True)


class FrozenStrictModel(StrictModel):
    """Strict immutable model for data passed between pipeline layers."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Roles(StrictModel):
    player_1: NonEmptyString
    player_2: NonEmptyString


class Turn(StrictModel):
    turn_id: PositiveStrictInt
    speaker: Player
    text: NonEmptyString


class Round(StrictModel):
    round_id: PositiveStrictInt
    roles: Roles
    turns: Annotated[list[Turn], Field(min_length=1)]

    @model_validator(mode="after")
    def check_turn_ids(self) -> Self:
        turn_ids = [turn.turn_id for turn in self.turns]
        expected = list(range(1, len(self.turns) + 1))
        if turn_ids != expected:
            raise ValueError(
                f"turn_id must be unique and consecutive from 1; got {turn_ids}"
            )
        return self


class Vote(StrictModel):
    judge_id: PositiveStrictInt
    collegia: Collegia
    decision: Player
    comment: NonEmptyString


class Score(StrictModel):
    player_1: NonNegativeStrictInt
    player_2: NonNegativeStrictInt


class ResultDef(StrictModel):
    score: Score


class MetadataDef(StrictModel):
    date: date
    source_file: HttpUrl


class TranscriptSchema(StrictModel):
    id: NonEmptyString
    situation: NonEmptyString
    rounds: Annotated[list[Round], Field(min_length=1)]
    human_votes: Annotated[list[Vote], Field(min_length=9, max_length=9)]
    result: ResultDef
    metadata: MetadataDef
    schema_version: NonEmptyString

    @model_validator(mode="after")
    def check_corpus_rules(self) -> Self:
        round_ids = [round_.round_id for round_ in self.rounds]
        expected_round_ids = list(range(1, len(self.rounds) + 1))
        if round_ids != expected_round_ids:
            raise ValueError(
                f"round_id must be unique and consecutive from 1; got {round_ids}"
            )

        judge_ids = [vote.judge_id for vote in self.human_votes]
        if judge_ids != list(range(1, 10)):
            raise ValueError(
                f"judge_id must be unique and consecutive from 1 to 9; got {judge_ids}"
            )

        collegia_counts = Counter(vote.collegia for vote in self.human_votes)
        if collegia_counts != EXPECTED_COLLEGIA_COUNTS:
            raise ValueError(
                "each of the three collegia must have exactly three judges; "
                f"got {dict(collegia_counts)}"
            )

        vote_counts = Counter(vote.decision for vote in self.human_votes)
        score = self.result.score
        if (
            score.player_1 != vote_counts["player_1"]
            or score.player_2 != vote_counts["player_2"]
        ):
            raise ValueError(
                "result.score must equal the actual vote counts; "
                f"votes are player_1={vote_counts['player_1']}, "
                f"player_2={vote_counts['player_2']}"
            )
        if score.player_1 + score.player_2 != 9:
            raise ValueError("result.score must total nine votes")

        return self


class PreparedRoles(FrozenStrictModel):
    """Player-to-role mapping that is safe to expose to an adapter."""

    player_1: NonEmptyString
    player_2: NonEmptyString


class PreparedTurn(FrozenStrictModel):
    """One transcript turn containing no human judging information."""

    turn_id: PositiveStrictInt
    speaker: Player
    text: NonEmptyString


class PreparedRound(FrozenStrictModel):
    """One complete round in the adapter-facing representation."""

    round_id: PositiveStrictInt
    roles: PreparedRoles
    turns: Annotated[tuple[PreparedTurn, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def check_turn_ids(self) -> Self:
        turn_ids = [turn.turn_id for turn in self.turns]
        expected = list(range(1, len(self.turns) + 1))
        if turn_ids != expected:
            raise ValueError(
                f"turn_id must be unique and consecutive from 1; got {turn_ids}"
            )
        return self


class PreparedBout(FrozenStrictModel):
    """Whitelist-only bout data supplied to model adapters."""

    situation: NonEmptyString
    rounds: Annotated[tuple[PreparedRound, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def check_round_ids(self) -> Self:
        round_ids = [round_.round_id for round_ in self.rounds]
        expected = list(range(1, len(self.rounds) + 1))
        if round_ids != expected:
            raise ValueError(
                f"round_id must be unique and consecutive from 1; got {round_ids}"
            )
        return self


Probability = Annotated[
    float,
    Field(strict=True, ge=0.0, le=1.0, allow_inf_nan=False),
]


class PlayerProbabilities(FrozenStrictModel):
    """Provider-supplied probability distribution over both allowed players."""

    player_1: Probability
    player_2: Probability

    @model_validator(mode="after")
    def check_total(self) -> Self:
        total = self.player_1 + self.player_2
        if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"player probabilities must sum to 1; got {total}")
        return self


class JudgeResponse(FrozenStrictModel):
    """Normalized successful response returned by every judge adapter."""

    winner: Player
    probabilities: PlayerProbabilities | None = None
    raw_response: JsonValue
    usage: dict[str, JsonValue] | None = None


RepeatIndex = Annotated[int, Field(strict=True, ge=1, le=3)]
Revision = Annotated[int, Field(strict=True, gt=0)]
AttemptIndex = Annotated[int, Field(strict=True, ge=1, le=2)]


class TaskKey(FrozenStrictModel):
    """Stable identity of one planned substantive vote."""

    run_id: NonEmptyString
    model_alias: NonEmptyString
    bout_id: NonEmptyString
    variant: Variant
    collegium: Collegium
    repeat_index: RepeatIndex


class AttemptError(FrozenStrictModel):
    """Serializable adapter error details without exception internals or secrets."""

    error_type: NonEmptyString
    message: NonEmptyString
    retryable: bool
    abort_run: bool = False


class TechnicalAttempt(FrozenStrictModel):
    """Outcome of one technical attempt for a planned vote."""

    attempt_index: AttemptIndex
    status: AttemptStatus
    started_at: AwareDatetime
    finished_at: AwareDatetime
    raw_response: JsonValue | None = None
    error: AttemptError | None = None

    @model_validator(mode="after")
    def check_outcome(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be earlier than started_at")
        if self.status == "success" and self.error is not None:
            raise ValueError("a successful attempt cannot contain an error")
        if self.status == "failed" and self.error is None:
            raise ValueError("a failed attempt must contain an error")
        return self


class CallRecord(FrozenStrictModel):
    """Final, storage-ready outcome of one planned vote revision."""

    task: TaskKey
    record_revision: Revision = 1
    previous_revision: Revision | None = None
    manual_retry: bool = False
    status: CallStatus
    provider: NonEmptyString
    api_model_id: NonEmptyString
    model_version: NonEmptyString | None = None
    prompt_path: NonEmptyString
    effective_parameters: dict[str, JsonValue]
    started_at: AwareDatetime
    finished_at: AwareDatetime
    attempts: Annotated[tuple[TechnicalAttempt, ...], Field(min_length=1, max_length=2)]
    response: JudgeResponse | None = None
    error: AttemptError | None = None

    @model_validator(mode="after")
    def check_outcome(self) -> Self:
        if self.record_revision == 1:
            if self.previous_revision is not None or self.manual_retry:
                raise ValueError(
                    "the first record revision cannot reference or be a manual retry"
                )
        elif (
            not self.manual_retry
            or self.previous_revision != self.record_revision - 1
        ):
            raise ValueError(
                "a later record revision must be a manual retry referencing the "
                "immediately previous revision"
            )

        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be earlier than started_at")

        attempt_indices = [attempt.attempt_index for attempt in self.attempts]
        expected = list(range(1, len(self.attempts) + 1))
        if attempt_indices != expected:
            raise ValueError(
                "attempt_index must be unique and consecutive from 1; "
                f"got {attempt_indices}"
            )

        if self.status == "success":
            if self.response is None or self.error is not None:
                raise ValueError(
                    "a successful call must contain response and must not contain error"
                )
            if self.attempts[-1].status != "success":
                raise ValueError("the final attempt of a successful call must succeed")
        else:
            if self.response is not None or self.error is None:
                raise ValueError(
                    "a failed call must contain error and must not contain response"
                )
            if any(attempt.status == "success" for attempt in self.attempts):
                raise ValueError("a failed call cannot contain a successful attempt")
        return self

    @property
    def revision(self) -> int:
        """Backward-compatible shorthand for the explicit persisted field name."""

        return self.record_revision


def load_transcript(path: Path) -> TranscriptSchema:
    """Read and strictly validate one UTF-8 transcript file."""

    return TranscriptSchema.model_validate_json(path.read_text(encoding="utf-8"))
