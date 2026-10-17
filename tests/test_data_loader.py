from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiment import data_loader
from src.experiment.config import load_config
from src.experiment.data_loader import (
    DatasetLoadError,
    load_dataset,
    load_prepared_bouts,
    prepare_bout,
    prepared_bout_payload,
    serialize_prepared_bout,
    swap_players,
)
from src.experiment.schemas import load_transcript


def _transcript_data(bout_id: str) -> dict:
    return {
        "id": bout_id,
        "situation": "Ситуация упоминает result.score, но это часть исходного текста.",
        "rounds": [
            {
                "round_id": 1,
                "roles": {"player_1": "Директор", "player_2": "Сотрудник"},
                "turns": [
                    {
                        "turn_id": 1,
                        "speaker": "player_1",
                        "text": "Первая реплика",
                    },
                    {
                        "turn_id": 2,
                        "speaker": "player_2",
                        "text": "Слово human_votes здесь является частью реплики",
                    },
                ],
            },
            {
                "round_id": 2,
                "roles": {"player_1": "Клиент", "player_2": "Подрядчик"},
                "turns": [
                    {
                        "turn_id": 1,
                        "speaker": "player_2",
                        "text": "Финальная реплика",
                    }
                ],
            },
        ],
        "human_votes": [
            {
                "judge_id": judge_id,
                "collegia": collegia,
                "decision": "player_1" if judge_id <= 5 else "player_2",
                "comment": f"Комментарий {judge_id}",
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
            "source_file": f"https://example.com/video/{bout_id}",
        },
        "schema_version": "1.0",
    }


def _write_transcript(directory: Path, filename: str, bout_id: str) -> Path:
    path = directory / filename
    path.write_text(
        json.dumps(_transcript_data(bout_id), ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_configured_pilot_dataset_is_loaded_in_numeric_id_order():
    bouts = load_dataset(load_config(), "pilot")

    assert [bout.bout_id for bout in bouts] == ["1", "2", "3", "4", "5"]
    assert bouts[0].get("original") is bouts[0].original
    assert bouts[0].get("swapped") is bouts[0].swapped


def test_unknown_dataset_is_rejected():
    with pytest.raises(DatasetLoadError, match="Unknown dataset"):
        load_dataset(load_config(), "archive")  # type: ignore[arg-type]


def test_bout_id_selects_exactly_one_after_dataset_validation(tmp_path: Path):
    _write_transcript(tmp_path, "ten.json", "10")
    _write_transcript(tmp_path, "two.json", "2")

    bouts = load_prepared_bouts(tmp_path, bout_id="2")

    assert len(bouts) == 1
    assert bouts[0].bout_id == "2"


def test_any_invalid_file_prevents_all_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_transcript(tmp_path, "valid.json", "1")
    (tmp_path / "invalid.json").write_text("{not valid json", encoding="utf-8")
    prepared_ids: list[str] = []
    real_prepare = data_loader.prepare_bout

    def observe_prepare(transcript):
        prepared_ids.append(transcript.id)
        return real_prepare(transcript)

    monkeypatch.setattr(data_loader, "prepare_bout", observe_prepare)

    with pytest.raises(DatasetLoadError, match=r"invalid\.json"):
        load_prepared_bouts(tmp_path, bout_id="1")
    assert prepared_ids == []


def test_duplicate_ids_are_rejected_before_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _write_transcript(tmp_path, "first.json", "1")
    _write_transcript(tmp_path, "second.json", "1")
    prepared_ids: list[str] = []

    monkeypatch.setattr(
        data_loader,
        "prepare_bout",
        lambda transcript: prepared_ids.append(transcript.id),
    )

    with pytest.raises(DatasetLoadError, match=r"Duplicate bout IDs.*first\.json.*second\.json"):
        load_prepared_bouts(tmp_path, bout_id="1")
    assert prepared_ids == []


def test_missing_bout_id_reports_available_ids(tmp_path: Path):
    _write_transcript(tmp_path, "two.json", "2")
    _write_transcript(tmp_path, "ten.json", "10")

    with pytest.raises(
        DatasetLoadError,
        match=r"Bout ID '3' was not found; available IDs: 2, 10",
    ):
        load_prepared_bouts(tmp_path, bout_id="3")


def test_prepared_bout_contains_full_whitelist_and_no_source_metadata(tmp_path: Path):
    source_path = _write_transcript(tmp_path, "bout.json", "1")
    source = load_transcript(source_path)

    prepared = prepare_bout(source)
    payload = prepared_bout_payload(prepared)

    assert set(payload) == {"situation", "rounds"}
    assert [round_["round_id"] for round_ in payload["rounds"]] == [1, 2]
    assert [turn["turn_id"] for turn in payload["rounds"][0]["turns"]] == [1, 2]
    assert payload["rounds"][0]["turns"][1]["text"] == source.rounds[0].turns[1].text
    assert payload["situation"] == source.situation

    serialized = json.dumps(payload, ensure_ascii=False)
    for forbidden_key in (
        '"id"',
        '"human_votes"',
        '"comment"',
        '"result"',
        '"metadata"',
        '"schema_version"',
        '"source_file"',
    ):
        assert forbidden_key not in serialized


def test_free_text_is_preserved_without_result_heuristics(tmp_path: Path):
    source = load_transcript(_write_transcript(tmp_path, "bout.json", "1"))

    prepared = prepare_bout(source)

    assert "result.score" in prepared.situation
    assert "human_votes" in prepared.rounds[0].turns[1].text


def test_swap_changes_only_player_labels_and_does_not_mutate_original(tmp_path: Path):
    source = load_transcript(_write_transcript(tmp_path, "bout.json", "1"))
    original = prepare_bout(source)
    before = original.model_dump(mode="json")

    swapped = swap_players(original)

    assert original.model_dump(mode="json") == before
    assert swapped is not original
    assert swapped.situation == original.situation
    assert [round_.round_id for round_ in swapped.rounds] == [1, 2]
    assert swapped.rounds[0].roles.player_1 == original.rounds[0].roles.player_2
    assert swapped.rounds[0].roles.player_2 == original.rounds[0].roles.player_1
    assert [turn.speaker for turn in swapped.rounds[0].turns] == [
        "player_2",
        "player_1",
    ]
    assert [turn.text for turn in swapped.rounds[0].turns] == [
        turn.text for turn in original.rounds[0].turns
    ]
    assert [turn.turn_id for turn in swapped.rounds[0].turns] == [1, 2]


def test_canonical_serialization_is_stable_and_complete(tmp_path: Path):
    source = load_transcript(_write_transcript(tmp_path, "bout.json", "1"))
    first = prepare_bout(source)
    second = prepare_bout(source)

    serialized = serialize_prepared_bout(first)

    assert serialized == serialize_prepared_bout(first)
    assert serialized == serialize_prepared_bout(second)
    assert json.loads(serialized) == prepared_bout_payload(first)
    assert len(json.loads(serialized)["rounds"]) == len(source.rounds)
