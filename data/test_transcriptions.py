from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from src.experiment.schemas import TranscriptSchema, load_transcript


DATA_DIR = Path(__file__).parent
DATASET_DIRS = (DATA_DIR / "pilot_dataset", DATA_DIR / "dataset")
TRANSCRIPT_FILES = tuple(
    path for directory in DATASET_DIRS for path in sorted(directory.glob("*.json"))
)


@pytest.mark.parametrize("dataset_dir", DATASET_DIRS, ids=lambda path: path.name)
def test_dataset_is_not_empty(dataset_dir: Path):
    assert list(dataset_dir.glob("*.json")), f"no JSON transcripts found in {dataset_dir}"


def test_pilot_dataset_contains_fixed_ids():
    pilot_dir = DATA_DIR / "pilot_dataset"
    actual_ids = {load_transcript(path).id for path in pilot_dir.glob("*.json")}

    assert actual_ids == {"1", "2", "3", "4", "5"}


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


def _remove_required_field(data: dict[str, Any]) -> None:
    del data["situation"]


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
        _remove_required_field,
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
