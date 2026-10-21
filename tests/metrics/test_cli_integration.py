from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
import shutil

import pytest
import yaml

from src.calculate_metrics import EXIT_SUCCESS, EXIT_USAGE_ERROR, run
from src.experiment.config import DEFAULT_CONFIG_PATH
from src.experiment.schemas import (
    CallRecord,
    JudgeResponse,
    PlayerProbabilities,
    TaskKey,
    TechnicalAttempt,
)
from src.experiment.storage import (
    PlannedTaskIdentity,
    RunModelSpec,
    RunSpec,
    RunStorage,
)
from src.metrics import COLLEGIA
from src.metrics.reporting import MANAGED_REPORT_FILES


NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def _write_config(tmp_path: Path, dataset_dir: Path, results_dir: Path) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["datasets"]["pilot"] = str(dataset_dir)
    raw["datasets"]["main"] = str(tmp_path / "unused-main-data")
    raw["results"]["smoke"] = str(tmp_path / "unused-smoke-results")
    raw["results"]["pilot"] = str(results_dir)
    raw["results"]["main"] = str(tmp_path / "unused-main-results")
    for name in raw["prompts"]:
        raw["prompts"][name] = str(tmp_path / f"unused-{name}.txt")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _create_complete_run(tmp_path: Path) -> tuple[Path, RunStorage, Path]:
    dataset_dir = tmp_path / "pilot-data"
    dataset_dir.mkdir()
    transcript_path = dataset_dir / "selected.json"
    shutil.copyfile(
        Path("data/pilot_dataset/fub_video_transcript.json"), transcript_path
    )
    results_dir = tmp_path / "results" / "pilot"
    tasks = tuple(
        PlannedTaskIdentity(
            model_alias="gigachat",
            bout_id="1",
            variant="original",
            collegium=collegium,
            repeat_index=repeat_index,
        )
        for collegium in COLLEGIA
        for repeat_index in (1, 2, 3)
    )
    spec = RunSpec(
        mode="pilot",
        dataset="pilot",
        selected_bout_ids=("1",),
        models=(
            RunModelSpec(
                alias="gigachat",
                provider="synthetic",
                api_model_id="offline-model",
                effective_parameters={},
            ),
        ),
        prompt_paths={collegium: f"unused-{collegium}.txt" for collegium in COLLEGIA},
        planned_tasks=tasks,
    )
    storage = RunStorage.create(
        results_dir, run_id="integration-001", spec=spec, started_at=NOW
    )
    for planned in tasks:
        task = TaskKey(run_id="integration-001", **planned.model_dump())
        response = JudgeResponse(
            winner="player_1",
            probabilities=PlayerProbabilities(player_1=0.75, player_2=0.25),
            raw_response={"winner": "player_1"},
        )
        attempt = TechnicalAttempt(
            attempt_index=1,
            status="success",
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=1),
            raw_response=response.raw_response,
        )
        storage.append_record(
            CallRecord(
                task=task,
                status="success",
                provider="synthetic",
                api_model_id="offline-model",
                prompt_path=storage.metadata.prompt_paths[task.collegium],
                effective_parameters={},
                started_at=NOW,
                finished_at=NOW + timedelta(seconds=1),
                attempts=(attempt,),
                response=response,
            )
        )
    config_path = _write_config(tmp_path, dataset_dir, results_dir)
    return config_path, storage, transcript_path


def _invoke(config_path: Path) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    code = run(
        [
            "--dataset",
            "pilot",
            "--run-id",
            "integration-001",
            "--config",
            str(config_path),
        ],
        output=stdout,
        error_output=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def test_cli_runs_offline_republishes_and_never_changes_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path, storage, transcript_path = _create_complete_run(tmp_path)
    monkeypatch.delenv("GIGACHAT_CREDENTIALS", raising=False)
    monkeypatch.delenv("AITUNNEL_API_KEY", raising=False)
    input_paths = (
        config_path,
        storage.metadata_path,
        storage.calls_path,
        storage.summary_path,
        transcript_path,
    )
    before = {path: path.read_bytes() for path in input_paths}

    first_code, first_stdout, first_stderr = _invoke(config_path)

    output_dir = storage.run_dir / "metrics"
    assert first_code == EXIT_SUCCESS
    assert "Метрики рассчитаны" in first_stdout
    assert first_stderr == ""
    assert {path.name for path in output_dir.iterdir()} == set(MANAGED_REPORT_FILES)
    assert {path: path.read_bytes() for path in input_paths} == before

    extra = output_dir / "team-notes.txt"
    extra.write_text("keep", encoding="utf-8")
    (output_dir / "report.md").write_text("stale report", encoding="utf-8")

    second_code, _, second_stderr = _invoke(config_path)

    assert second_code == EXIT_SUCCESS
    assert second_stderr == ""
    assert (output_dir / "report.md").read_text(encoding="utf-8") != "stale report"
    assert extra.read_text(encoding="utf-8") == "keep"
    assert {path: path.read_bytes() for path in input_paths} == before


@pytest.mark.parametrize(
    ("damage", "expected_message"),
    [
        ("missing_metadata", "Cannot open selected run"),
        ("missing_calls", "Cannot open selected run"),
        ("corrupt_metadata", "Cannot open selected run"),
        ("corrupt_calls", "calls.jsonl line 1"),
        ("missing_transcript", "expected exactly one file, found 0"),
        ("corrupt_transcript", "Cannot index transcript"),
    ],
)
def test_cli_reports_missing_and_corrupt_inputs(
    tmp_path: Path, damage: str, expected_message: str
):
    config_path, storage, transcript_path = _create_complete_run(tmp_path)
    targets = {
        "missing_metadata": storage.metadata_path,
        "missing_calls": storage.calls_path,
        "corrupt_metadata": storage.metadata_path,
        "corrupt_calls": storage.calls_path,
        "missing_transcript": transcript_path,
        "corrupt_transcript": transcript_path,
    }
    target = targets[damage]
    if damage.startswith("missing_"):
        target.unlink()
    else:
        target.write_bytes(b'{"truncated":')

    code, _, stderr = _invoke(config_path)

    assert code == EXIT_USAGE_ERROR
    assert expected_message in stderr
    assert not (storage.run_dir / "metrics").exists()
