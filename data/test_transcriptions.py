from __future__ import annotations

import copy
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, ValidationError, model_validator


DATASET_DIR = Path(__file__).parent / "dataset"
TRANSCRIPT_FILES = sorted(DATASET_DIR.glob("*.json"))

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeStrictInt = Annotated[int, Field(strict=True, ge=0)]
Player = Literal["player_1", "player_2"]
Collegia = Literal[
    "Нанимающаяся на работу",
    "Отправляющая на переговоры",
    "Доверяющая собственность",
]


class StrictModel(BaseModel):
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
    def check_turn_ids(self):
        turn_ids = [turn.turn_id for turn in self.turns]
        expected = list(range(1, len(self.turns) + 1))
        if turn_ids != expected:
            raise ValueError(f"turn_id must be unique and consecutive from 1; got {turn_ids}")
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
    def check_corpus_rules(self):
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
        expected_collegia_counts = {
            "Нанимающаяся на работу": 3,
            "Отправляющая на переговоры": 3,
            "Доверяющая собственность": 3,
        }
        if collegia_counts != expected_collegia_counts:
            raise ValueError(
                "each of the three collegia must have exactly three judges; "
                f"got {dict(collegia_counts)}"
            )

        vote_counts = Counter(vote.decision for vote in self.human_votes)
        score = self.result.score
        if score.player_1 != vote_counts["player_1"] or score.player_2 != vote_counts["player_2"]:
            raise ValueError(
                "result.score must equal the actual vote counts; "
                f"votes are player_1={vote_counts['player_1']}, "
                f"player_2={vote_counts['player_2']}"
            )
        if score.player_1 + score.player_2 != 9:
            raise ValueError("result.score must total nine votes")

        return self


def load_transcript(path: Path) -> TranscriptSchema:
    return TranscriptSchema.model_validate_json(path.read_text(encoding="utf-8"))


def test_dataset_is_not_empty():
    assert TRANSCRIPT_FILES, f"no JSON transcripts found in {DATASET_DIR}"


@pytest.mark.parametrize("path", TRANSCRIPT_FILES, ids=lambda path: path.name)
def test_transcription_is_valid(path: Path):
    load_transcript(path)


def test_corpus_ids_and_filenames_are_consistent():
    seen_ids: set[str] = set()

    for path in TRANSCRIPT_FILES:
        if path.name == "fub_video_transcript.json":
            expected_id = "1"
        else:
            match = re.fullmatch(r"fub_video_transcript([1-9]\d*)\.json", path.name)
            assert match is not None, f"unexpected transcript filename: {path.name}"
            expected_id = match.group(1)
            assert expected_id != "1", (
                "id 1 must use the special filename fub_video_transcript.json"
            )

        transcript = load_transcript(path)
        assert transcript.id == expected_id, (
            f"{path.name} must contain id={expected_id!r}, got {transcript.id!r}"
        )
        assert transcript.id not in seen_ids, f"duplicate transcript id: {transcript.id}"
        seen_ids.add(transcript.id)


def _valid_transcript_data() -> dict[str, Any]:
    return {
        "id": "1",
        "situation": "Ситуация",
        "rounds": [
            {
                "round_id": 1,
                "roles": {"player_1": "Роль 1", "player_2": "Роль 2"},
                "turns": [
                    {"turn_id": 1, "speaker": "player_1", "text": "Реплика"}
                ],
            }
        ],
        "human_votes": [
            {
                "judge_id": judge_id,
                "collegia": collegia,
                "decision": "player_1" if judge_id <= 5 else "player_2",
                "comment": "Комментарий",
            }
            for collegia, judge_ids in (
                ("Нанимающаяся на работу", range(1, 4)),
                ("Отправляющая на переговоры", range(4, 7)),
                ("Доверяющая собственность", range(7, 10)),
            )
            for judge_id in judge_ids
        ],
        "result": {"score": {"player_1": 5, "player_2": 4}},
        "metadata": {
            "date": "2026-10-07",
            "source_file": "https://example.com/video/1",
        },
        "schema_version": "1.0",
    }


def _validate_data(data: dict[str, Any]) -> TranscriptSchema:
    return TranscriptSchema.model_validate_json(json.dumps(data, ensure_ascii=False))


def _add_extra_field(data: dict[str, Any]) -> None:
    data["unexpected"] = True


def _make_string_empty(data: dict[str, Any]) -> None:
    data["rounds"][0]["turns"][0]["text"] = "   "


def _make_integer_non_strict(data: dict[str, Any]) -> None:
    data["rounds"][0]["round_id"] = "1"


def _use_unknown_player(data: dict[str, Any]) -> None:
    data["rounds"][0]["turns"][0]["speaker"] = "player_3"


def _use_unknown_collegia(data: dict[str, Any]) -> None:
    data["human_votes"][0]["collegia"] = "Неизвестная коллегия"


def _use_invalid_date(data: dict[str, Any]) -> None:
    data["metadata"]["date"] = "07.10.2026"


def _use_invalid_url(data: dict[str, Any]) -> None:
    data["metadata"]["source_file"] = "not a URL"


def _repeat_round_id(data: dict[str, Any]) -> None:
    second_round = copy.deepcopy(data["rounds"][0])
    data["rounds"].append(second_round)


def _skip_turn_id(data: dict[str, Any]) -> None:
    second_turn = copy.deepcopy(data["rounds"][0]["turns"][0])
    second_turn["turn_id"] = 3
    data["rounds"][0]["turns"].append(second_turn)


def _repeat_judge_id(data: dict[str, Any]) -> None:
    data["human_votes"][1]["judge_id"] = 1


def _break_collegia_composition(data: dict[str, Any]) -> None:
    data["human_votes"][2]["collegia"] = "Отправляющая на переговоры"


def _mismatch_score(data: dict[str, Any]) -> None:
    data["result"]["score"] = {"player_1": 4, "player_2": 5}


@pytest.mark.parametrize(
    "mutate",
    [
        _add_extra_field,
        _make_string_empty,
        _make_integer_non_strict,
        _use_unknown_player,
        _use_unknown_collegia,
        _use_invalid_date,
        _use_invalid_url,
        _repeat_round_id,
        _skip_turn_id,
        _repeat_judge_id,
        _break_collegia_composition,
        _mismatch_score,
    ],
    ids=lambda mutate: mutate.__name__.removeprefix("_"),
)
def test_invalid_transcript_is_rejected(mutate):
    data = _valid_transcript_data()
    mutate(data)

    with pytest.raises(ValidationError):
        _validate_data(data)
