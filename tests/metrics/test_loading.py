from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
import yaml

from src.experiment.config import DEFAULT_CONFIG_PATH, load_config
from src.experiment.storage import (
    PlannedTaskIdentity,
    RunModelSpec,
    RunSpec,
    RunStorage,
)
from src.metrics.loading import AnalysisInputError, load_analysis_inputs


def _config(tmp_path: Path, dataset_dir: Path, results_dir: Path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["datasets"]["pilot"] = str(dataset_dir)
    raw["datasets"]["main"] = str(tmp_path / "unused-dataset")
    raw["results"]["pilot"] = str(results_dir)
    raw["results"]["main"] = str(tmp_path / "unused-results")
    for name in raw["prompts"]:
        raw["prompts"][name] = str(tmp_path / f"missing-{name}.txt")
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return load_config(path, validate_input_paths=False)


def _spec(run_id: str, bout_id: str = "1") -> RunSpec:
    task = PlannedTaskIdentity(
        model_alias="model-a",
        bout_id=bout_id,
        variant="original",
        collegium="hiring",
        repeat_index=1,
    )
    return RunSpec(
        mode="pilot",
        dataset="pilot",
        selected_bout_ids=(bout_id,),
        models=(
            RunModelSpec(
                alias="model-a",
                provider="test",
                api_model_id="test-model",
                effective_parameters={},
            ),
        ),
        prompt_paths={
            "hiring": "missing-hiring.txt",
            "negotiation": "missing-negotiation.txt",
            "ownership": "missing-ownership.txt",
        },
        planned_tasks=(task,),
    )


def _copy_transcript(dataset_dir: Path, *, target_name: str = "not-the-id.json") -> Path:
    dataset_dir.mkdir(parents=True)
    source = Path("data/pilot_dataset/fub_video_transcript.json").resolve()
    target = dataset_dir / target_name
    shutil.copyfile(source, target)
    return target


def test_loads_one_explicit_run_by_transcript_root_id_and_hashes_used_files(
    tmp_path: Path,
):
    dataset_dir = tmp_path / "pilot-data"
    transcript_path = _copy_transcript(dataset_dir)
    # This file has a valid root id but violates TranscriptSchema. It is unselected.
    (dataset_dir / "unselected.json").write_text(
        json.dumps({"id": "unselected", "anything": "invalid"}),
        encoding="utf-8",
    )
    results_dir = tmp_path / "results" / "pilot"
    storage = RunStorage.create(results_dir, run_id="007", spec=_spec("007"))
    storage.summary_path.write_bytes(b'{"stale":true}\n')
    before = storage.summary_path.read_bytes()
    config = _config(tmp_path, dataset_dir, results_dir)

    loaded = load_analysis_inputs(config, "pilot", "007")

    assert loaded.metadata.run_id == "007"
    assert loaded.transcripts[0].path == transcript_path
    assert loaded.transcripts[0].transcript.id == "1"
    assert loaded.latest_records == ()
    assert storage.summary_path.read_bytes() == before
    assert [item.path for item in loaded.input_files] == [
        storage.metadata_path,
        storage.calls_path,
        transcript_path,
    ]
    assert all(len(item.sha256) == 64 for item in loaded.input_files)


def test_rejects_run_from_a_different_result_mode(tmp_path: Path):
    dataset_dir = tmp_path / "pilot-data"
    _copy_transcript(dataset_dir)
    results_dir = tmp_path / "results" / "pilot"
    mismatched = _spec("008").model_copy(update={"mode": "main", "dataset": "main"})
    RunStorage.create(results_dir, run_id="008", spec=mismatched)
    config = _config(tmp_path, dataset_dir, results_dir)

    with pytest.raises(AnalysisInputError, match="does not match"):
        load_analysis_inputs(config, "pilot", "008")


def test_requires_exactly_one_selected_transcript(tmp_path: Path):
    dataset_dir = tmp_path / "pilot-data"
    first = _copy_transcript(dataset_dir, target_name="first.json")
    shutil.copyfile(first, dataset_dir / "duplicate.json")
    results_dir = tmp_path / "results" / "pilot"
    RunStorage.create(results_dir, run_id="009", spec=_spec("009"))
    config = _config(tmp_path, dataset_dir, results_dir)

    with pytest.raises(AnalysisInputError, match="expected exactly one file, found 2"):
        load_analysis_inputs(config, "pilot", "009")


def test_rejects_corrupt_selected_run_journal(tmp_path: Path):
    dataset_dir = tmp_path / "pilot-data"
    _copy_transcript(dataset_dir)
    results_dir = tmp_path / "results" / "pilot"
    storage = RunStorage.create(results_dir, run_id="010", spec=_spec("010"))
    storage.calls_path.write_bytes(b'{"truncated":')
    config = _config(tmp_path, dataset_dir, results_dir)

    with pytest.raises(AnalysisInputError, match=r"calls\.jsonl line 1"):
        load_analysis_inputs(config, "pilot", "010")
