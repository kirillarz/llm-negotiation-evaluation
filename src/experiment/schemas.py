"""Production data contracts for negotiation transcripts."""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StringConstraints,
    model_validator,
)


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeStrictInt = Annotated[int, Field(strict=True, ge=0)]
Player = Literal["player_1", "player_2"]
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


def load_transcript(path: Path) -> TranscriptSchema:
    """Read and strictly validate one UTF-8 transcript file."""

    return TranscriptSchema.model_validate_json(path.read_text(encoding="utf-8"))
