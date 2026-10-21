"""Read-only loading and provenance capture for one metrics calculation run."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from src.experiment.config import ExperimentConfig
from src.experiment.schemas import CallRecord, TranscriptSchema, load_transcript
from src.experiment.storage import RunMetadata, RunStorage, StorageError


DatasetName = Literal["pilot", "main"]


class AnalysisInputError(RuntimeError):
    """A selected local input cannot be read or violates its persisted contract."""


@dataclass(frozen=True, slots=True)
class InputFile:
    """A file actually used by the analysis and its content digest."""

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class SelectedTranscript:
    """One selected, fully validated transcript together with its source file."""

    path: Path
    transcript: TranscriptSchema


@dataclass(frozen=True, slots=True)
class AnalysisInputs:
    """Validated scientific inputs for exactly one explicitly selected run."""

    dataset: DatasetName
    run: RunStorage
    latest_records: tuple[CallRecord, ...]
    transcripts: tuple[SelectedTranscript, ...]
    input_files: tuple[InputFile, ...]

    @property
    def metadata(self) -> RunMetadata:
        return self.run.metadata


def file_sha256(path: Path) -> str:
    """Hash the exact bytes of one input file."""

    digest = sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise AnalysisInputError(f"Cannot hash input file {path}: {error}") from error
    return digest.hexdigest()


def _configured_paths(
    config: ExperimentConfig,
    dataset: DatasetName,
) -> tuple[Path, Path]:
    dataset_dir = getattr(config.datasets, dataset)
    results_root = getattr(config.results, dataset)
    if not dataset_dir.is_dir():
        raise AnalysisInputError(
            f"Selected dataset directory does not exist: {dataset_dir}"
        )
    if not results_root.is_dir():
        raise AnalysisInputError(
            f"Selected results directory does not exist: {results_root}"
        )
    return dataset_dir, results_root


def _json_files(directory: Path) -> tuple[Path, ...]:
    try:
        return tuple(
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
        raise AnalysisInputError(
            f"Cannot scan selected dataset directory {directory}: {error}"
        ) from error


def _read_root_id(path: Path) -> str:
    """Read only enough structure to index a transcript candidate by root ``id``."""

    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnalysisInputError(f"Cannot index transcript {path}: {error}") from error
    if not isinstance(payload, dict):
        raise AnalysisInputError(f"Cannot index transcript {path}: root must be an object")
    bout_id = payload.get("id")
    if not isinstance(bout_id, str) or not bout_id.strip():
        raise AnalysisInputError(
            f"Cannot index transcript {path}: root id must be a non-empty string"
        )
    return bout_id.strip()


def load_selected_transcripts(
    directory: Path,
    selected_bout_ids: tuple[str, ...],
) -> tuple[SelectedTranscript, ...]:
    """Index files by root ID, then fully validate only selected transcripts."""

    selected = set(selected_bout_ids)
    paths_by_id: dict[str, list[Path]] = {bout_id: [] for bout_id in selected_bout_ids}
    for path in _json_files(directory):
        bout_id = _read_root_id(path)
        if bout_id in selected:
            paths_by_id[bout_id].append(path)

    problems = []
    for bout_id in selected_bout_ids:
        count = len(paths_by_id[bout_id])
        if count != 1:
            problems.append(f"id {bout_id!r}: expected exactly one file, found {count}")
    if problems:
        raise AnalysisInputError(
            "Selected transcripts cannot be matched by root id: " + "; ".join(problems)
        )

    loaded = []
    for bout_id in selected_bout_ids:
        path = paths_by_id[bout_id][0]
        try:
            transcript = load_transcript(path)
        except (OSError, UnicodeError, ValidationError) as error:
            raise AnalysisInputError(
                f"Invalid selected transcript {path}: {error}"
            ) from error
        if transcript.id != bout_id:  # defensive: the same schema field built the index
            raise AnalysisInputError(
                f"Selected transcript id changed while reading {path}"
            )
        loaded.append(SelectedTranscript(path=path, transcript=transcript))
    return tuple(loaded)


def load_analysis_inputs(
    config: ExperimentConfig,
    dataset: DatasetName,
    run_id: str,
) -> AnalysisInputs:
    """Load exactly one run and its selected current transcripts without writes."""

    dataset_dir, results_root = _configured_paths(config, dataset)
    try:
        run = RunStorage.open_read_only(results_root, run_id)
    except StorageError as error:
        raise AnalysisInputError(f"Cannot open selected run: {error}") from error

    if run.metadata.mode != dataset or run.metadata.dataset != dataset:
        raise AnalysisInputError(
            "Selected run metadata does not match the configured dataset directory: "
            f"requested {dataset!r}, metadata mode={run.metadata.mode!r}, "
            f"dataset={run.metadata.dataset!r}"
        )

    transcripts = load_selected_transcripts(
        dataset_dir,
        run.metadata.selected_bout_ids,
    )
    source_paths = (
        run.metadata_path,
        run.calls_path,
        *(item.path for item in transcripts),
    )
    input_files = tuple(
        InputFile(path=path, sha256=file_sha256(path)) for path in source_paths
    )
    return AnalysisInputs(
        dataset=dataset,
        run=run,
        latest_records=run.latest_records,
        transcripts=transcripts,
        input_files=input_files,
    )


# Concise alias for callers that already operate inside the metrics package.
load_inputs = load_analysis_inputs


__all__ = [
    "AnalysisInputError",
    "AnalysisInputs",
    "InputFile",
    "SelectedTranscript",
    "file_sha256",
    "load_analysis_inputs",
    "load_inputs",
    "load_selected_transcripts",
]
