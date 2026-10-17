"""Atomic dataset loading and whitelist-only bout preparation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, cast

from pydantic import JsonValue, ValidationError

from .config import ExperimentConfig
from .schemas import (
    Player,
    PreparedBout,
    PreparedRoles,
    PreparedRound,
    PreparedTurn,
    TranscriptSchema,
    Variant,
    load_transcript,
)


DatasetName = Literal["pilot", "main"]


class DatasetLoadError(ValueError):
    """The selected dataset cannot be validated or selected unambiguously."""


@dataclass(frozen=True, slots=True)
class PreparedBoutVariants:
    """Bookkeeping ID plus isolated original and label-swapped adapter inputs."""

    bout_id: str
    original: PreparedBout
    swapped: PreparedBout

    def get(self, variant: Variant) -> PreparedBout:
        if variant == "original":
            return self.original
        if variant == "swapped":
            return self.swapped
        raise ValueError(f"Unknown bout variant: {variant!r}")


def _json_files(directory: Path) -> tuple[Path, ...]:
    try:
        files = tuple(
            sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() == ".json"
                ),
                key=lambda path: (path.name.casefold(), path.name),
            )
        )
    except OSError as error:
        raise DatasetLoadError(f"Cannot scan dataset directory {directory}: {error}") from error

    if not files:
        raise DatasetLoadError(f"Dataset directory contains no JSON files: {directory}")
    return files


def _load_all(files: tuple[Path, ...]) -> tuple[tuple[Path, TranscriptSchema], ...]:
    loaded: list[tuple[Path, TranscriptSchema]] = []
    failures: list[str] = []

    for path in files:
        try:
            transcript = load_transcript(path)
        except (OSError, UnicodeError, ValidationError) as error:
            failures.append(f"{path.name}: {type(error).__name__}: {error}")
        else:
            loaded.append((path, transcript))

    if failures:
        details = "\n".join(f"- {failure}" for failure in failures)
        raise DatasetLoadError(
            "Dataset validation failed; no bouts were prepared:\n" + details
        )
    return tuple(loaded)


def _check_unique_ids(loaded: tuple[tuple[Path, TranscriptSchema], ...]) -> None:
    paths_by_id: dict[str, list[Path]] = {}
    for path, transcript in loaded:
        paths_by_id.setdefault(transcript.id, []).append(path)

    duplicates = {
        bout_id: paths for bout_id, paths in paths_by_id.items() if len(paths) > 1
    }
    if not duplicates:
        return

    details = "; ".join(
        f"id {bout_id!r}: {', '.join(path.name for path in paths)}"
        for bout_id, paths in sorted(duplicates.items())
    )
    raise DatasetLoadError(f"Duplicate bout IDs in selected dataset: {details}")


def _bout_sort_key(item: tuple[Path, TranscriptSchema]) -> tuple[object, ...]:
    path, transcript = item
    if transcript.id.isdecimal():
        return (0, int(transcript.id), transcript.id, path.name.casefold(), path.name)
    return (1, transcript.id.casefold(), transcript.id, path.name.casefold(), path.name)


def _select_bouts(
    loaded: tuple[tuple[Path, TranscriptSchema], ...],
    bout_id: str | None,
) -> tuple[TranscriptSchema, ...]:
    ordered = tuple(sorted(loaded, key=_bout_sort_key))
    if bout_id is None:
        return tuple(transcript for _, transcript in ordered)

    if not isinstance(bout_id, str) or not bout_id.strip():
        raise DatasetLoadError("bout_id must be a non-empty string")
    requested_id = bout_id.strip()
    matches = tuple(
        transcript for _, transcript in ordered if transcript.id == requested_id
    )
    if not matches:
        available = ", ".join(transcript.id for _, transcript in ordered)
        raise DatasetLoadError(
            f"Bout ID {requested_id!r} was not found; available IDs: {available}"
        )
    if len(matches) != 1:
        # Duplicate IDs are checked before selection; retain a defensive boundary here.
        raise DatasetLoadError(f"Bout ID {requested_id!r} is ambiguous")
    return matches


def prepare_bout(transcript: TranscriptSchema) -> PreparedBout:
    """Copy only adapter-approved transcript fields into an isolated object."""

    return PreparedBout(
        situation=transcript.situation,
        rounds=tuple(
            PreparedRound(
                round_id=round_.round_id,
                roles=PreparedRoles(
                    player_1=round_.roles.player_1,
                    player_2=round_.roles.player_2,
                ),
                turns=tuple(
                    PreparedTurn(
                        turn_id=turn.turn_id,
                        speaker=turn.speaker,
                        text=turn.text,
                    )
                    for turn in round_.turns
                ),
            )
            for round_ in transcript.rounds
        ),
    )


def _swap_player(player: Player) -> Player:
    if player == "player_1":
        return "player_2"
    return "player_1"


def swap_players(bout: PreparedBout) -> PreparedBout:
    """Create a new bout with player labels swapped everywhere they identify a player."""

    return PreparedBout(
        situation=bout.situation,
        rounds=tuple(
            PreparedRound(
                round_id=round_.round_id,
                roles=PreparedRoles(
                    player_1=round_.roles.player_2,
                    player_2=round_.roles.player_1,
                ),
                turns=tuple(
                    PreparedTurn(
                        turn_id=turn.turn_id,
                        speaker=_swap_player(turn.speaker),
                        text=turn.text,
                    )
                    for turn in round_.turns
                ),
            )
            for round_ in bout.rounds
        ),
    )


def prepared_bout_payload(bout: PreparedBout) -> dict[str, JsonValue]:
    """Return the common JSON-compatible content supplied to either adapter."""

    return cast(dict[str, JsonValue], bout.model_dump(mode="json"))


def serialize_prepared_bout(bout: PreparedBout) -> str:
    """Serialize the complete adapter input to stable, canonical UTF-8 JSON text."""

    return json.dumps(
        prepared_bout_payload(bout),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_prepared_bouts(
    directory: Path,
    *,
    bout_id: str | None = None,
) -> tuple[PreparedBoutVariants, ...]:
    """Validate a whole dataset atomically, then prepare all or one selected bout."""

    files = _json_files(directory)
    loaded = _load_all(files)
    _check_unique_ids(loaded)
    selected = _select_bouts(loaded, bout_id)

    result = []
    for transcript in selected:
        original = prepare_bout(transcript)
        result.append(
            PreparedBoutVariants(
                bout_id=transcript.id,
                original=original,
                swapped=swap_players(original),
            )
        )
    return tuple(result)


def load_dataset(
    config: ExperimentConfig,
    dataset: DatasetName,
    *,
    bout_id: str | None = None,
) -> tuple[PreparedBoutVariants, ...]:
    """Select the configured pilot/main directory and load it without API access."""

    if dataset == "pilot":
        directory = config.datasets.pilot
    elif dataset == "main":
        directory = config.datasets.main
    else:
        raise DatasetLoadError(
            f"Unknown dataset {dataset!r}; expected 'pilot' or 'main'"
        )
    return load_prepared_bouts(directory, bout_id=bout_id)
