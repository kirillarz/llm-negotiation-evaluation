from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from src.experiment.config import load_config
from src.experiment.data_loader import load_dataset
from src.experiment.prompts import load_prompts
from src.experiment.runner import build_task_plan
from src.experiment.schemas import (
    AttemptError,
    CallRecord,
    Collegium,
    JudgeResponse,
    PreparedBout,
    TaskKey,
    TechnicalAttempt,
)
from src.experiment.storage import (
    JournalCorruptionError,
    RecordRevisionError,
    RunCompatibilityError,
    RunSpec,
    RunStorage,
    build_run_spec,
)
from src.judges.base import BaseJudge


NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


class StorageJudge(BaseJudge):
    SUPPORTED_MODEL_IDS = frozenset({"GigaChat-2-Pro"})

    def _evaluate_once(
        self,
        bout: PreparedBout,
        collegium: Collegium,
    ) -> JudgeResponse:
        return JudgeResponse(
            winner="player_1",
            probabilities=None,
            raw_response={"winner": "player_1"},
        )


def _plan(run_id: str, *, full: bool = False):
    config = load_config()
    prompts = load_prompts(config.prompts)
    judge = StorageJudge(
        config.models.llm,
        prompts,
        environ={"GIGACHAT_CREDENTIALS": "never-persist-this-secret"},
    )
    bouts = load_dataset(config, "pilot", bout_id="1")
    plan = build_task_plan(
        run_id=run_id,
        bouts=bouts,
        judges=(judge,),
        smoke=not full,
        bout_id=None if full else "1",
        collegium=None if full else "hiring",
    )
    return plan, prompts


def _spec(
    run_id: str,
    *,
    full: bool = False,
    mode: str = "smoke",
    dataset: str = "pilot",
) -> RunSpec:
    plan, prompts = _plan(run_id, full=full)
    return build_run_spec(
        mode=mode,  # type: ignore[arg-type]
        dataset=dataset,  # type: ignore[arg-type]
        tasks=plan,
        prompt_paths=prompts.metadata_paths(),
        model_versions={"gigachat": "provider-version-1"},
    )


def _record(
    storage: RunStorage,
    task: TaskKey,
    *,
    success: bool,
    retry_failed: bool = False,
    raw_response=None,
) -> CallRecord:
    revision = storage.revision_fields(task, retry_failed=retry_failed)
    model = storage.metadata.models[0]
    if success:
        attempt = TechnicalAttempt(
            attempt_index=1,
            status="success",
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=1),
            raw_response=raw_response or {"winner": "player_1"},
        )
        response = JudgeResponse(
            winner="player_1",
            probabilities=None,
            raw_response=raw_response or {"winner": "player_1"},
            usage={"input_tokens": 10},
        )
        return CallRecord(
            task=task,
            **revision,
            status="success",
            provider=model.provider,
            api_model_id=model.api_model_id,
            model_version=model.model_version,
            prompt_path=storage.metadata.prompt_paths[task.collegium],
            effective_parameters=model.effective_parameters,
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=1),
            attempts=(attempt,),
            response=response,
        )

    error = AttemptError(
        error_type="TransientJudgeError",
        message="provider timeout",
        retryable=True,
    )
    attempts = (
        TechnicalAttempt(
            attempt_index=1,
            status="failed",
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=1),
            error=error,
        ),
        TechnicalAttempt(
            attempt_index=2,
            status="failed",
            started_at=NOW + timedelta(seconds=1),
            finished_at=NOW + timedelta(seconds=2),
            error=error,
        ),
    )
    return CallRecord(
        task=task,
        **revision,
        status="failed",
        provider=model.provider,
        api_model_id=model.api_model_id,
        model_version=model.model_version,
        prompt_path=storage.metadata.prompt_paths[task.collegium],
        effective_parameters=model.effective_parameters,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=2),
        attempts=attempts,
        error=error,
    )


@pytest.mark.parametrize("result_kind", ["smoke", "pilot", "main"])
def test_create_keeps_result_kinds_in_the_supplied_root(
    tmp_path: Path,
    result_kind: str,
):
    run_id = f"run-{result_kind}"
    spec = _spec(
        run_id,
        mode=result_kind,
        dataset="main" if result_kind == "main" else "pilot",
    )
    root = tmp_path / "results" / result_kind

    storage = RunStorage.create(root, run_id=run_id, spec=spec, started_at=NOW)

    assert storage.run_dir == root / run_id
    assert {path.name for path in storage.run_dir.iterdir()} == {
        "metadata.json",
        "calls.jsonl",
        "summary.json",
    }
    metadata_text = storage.metadata_path.read_text(encoding="utf-8")
    assert "never-persist-this-secret" not in metadata_text
    assert "human_votes" not in metadata_text
    assert '"situation"' not in metadata_text
    assert storage.calls_path.read_bytes() == b""


def test_append_is_immediate_jsonl_and_summary_is_technical_only(tmp_path: Path):
    run_id = "run-append"
    storage = RunStorage.create(
        tmp_path,
        run_id=run_id,
        spec=_spec(run_id),
        started_at=NOW,
        clock=lambda: NOW,
    )
    task = storage.tasks_to_run()[0]

    storage.append_record(_record(storage, task, success=True))

    journal = storage.calls_path.read_bytes()
    assert journal.endswith(b"\n")
    persisted = json.loads(journal)
    assert persisted["record_revision"] == 1
    assert persisted["response"]["winner"] == "player_1"
    assert persisted["response"]["usage"] == {"input_tokens": 10}
    assert len(persisted["attempts"]) == 1

    summary_text = storage.summary_path.read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    assert (summary["planned"], summary["success"], summary["failed"], summary["pending"]) == (
        1,
        1,
        0,
        0,
    )
    assert summary["blocks"][0]["status"] == "completed"
    assert "winner" not in summary_text
    assert "player_1" not in summary_text


def test_provider_raw_data_is_preserved_but_secrets_and_request_are_redacted(
    tmp_path: Path,
):
    run_id = "run-redaction"
    storage = RunStorage.create(
        tmp_path,
        run_id=run_id,
        spec=_spec(run_id),
        started_at=NOW,
    )
    task = storage.tasks_to_run()[0]
    raw = {
        "decision_id": "abc",
        "authorization": "Bearer secret",
        "request": {"situation": "full transcript must not be stored"},
    }

    persisted = storage.append_record(
        _record(storage, task, success=True, raw_response=raw)
    )

    assert persisted.response is not None
    assert persisted.response.raw_response == {
        "decision_id": "abc",
        "authorization": "[REDACTED]",
        "request": "[REDACTED]",
    }
    assert "full transcript must not be stored" not in storage.calls_path.read_text(
        encoding="utf-8"
    )


def test_resume_skips_finished_tasks_and_retry_failed_adds_linked_revision(
    tmp_path: Path,
):
    run_id = "run-retry"
    spec = _spec(run_id, full=True)
    storage = RunStorage.create(tmp_path, run_id=run_id, spec=spec, started_at=NOW)
    successful_task, failed_task = storage.tasks_to_run()[:2]
    storage.append_record(_record(storage, successful_task, success=True))
    storage.append_record(_record(storage, failed_task, success=False))

    resumed = RunStorage.resume(tmp_path, run_id, expected=spec)

    default_tasks = resumed.tasks_to_run()
    retry_tasks = resumed.tasks_to_run(retry_failed=True)
    assert successful_task not in default_tasks
    assert failed_task not in default_tasks
    assert successful_task not in retry_tasks
    assert failed_task in retry_tasks
    assert resumed.revision_fields(failed_task, retry_failed=True) == {
        "record_revision": 2,
        "previous_revision": 1,
        "manual_retry": True,
    }

    resumed.append_record(
        _record(resumed, failed_task, success=True, retry_failed=True)
    )
    reopened = RunStorage.resume(tmp_path, run_id)
    assert len(reopened.records) == 3
    assert reopened.records[-1].record_revision == 2
    assert reopened.records[-1].previous_revision == 1
    assert reopened.records[-1].manual_retry is True
    assert reopened.summary().success == 2
    assert reopened.summary().failed == 0


def test_successful_task_cannot_be_retried(tmp_path: Path):
    run_id = "run-no-success-retry"
    storage = RunStorage.create(
        tmp_path,
        run_id=run_id,
        spec=_spec(run_id),
        started_at=NOW,
    )
    task = storage.tasks_to_run()[0]
    storage.append_record(_record(storage, task, success=True))

    with pytest.raises(RecordRevisionError, match="successful task cannot be retried"):
        storage.revision_fields(task, retry_failed=True)


def test_resume_rejects_changed_effective_parameters(tmp_path: Path):
    run_id = "run-compatibility"
    spec = _spec(run_id)
    RunStorage.create(tmp_path, run_id=run_id, spec=spec, started_at=NOW)
    changed_model = spec.models[0].model_copy(
        update={"effective_parameters": {"temperature": 0.9}}
    )
    changed = spec.model_copy(update={"models": (changed_model,)})

    with pytest.raises(RunCompatibilityError, match="do not match"):
        RunStorage.resume(tmp_path, run_id, expected=changed)


@pytest.mark.parametrize(
    "damaged_line",
    [
        b'{"task":',
        b'{"not":"a call record"}\n',
        b"\n",
    ],
)
def test_resume_fails_loudly_on_incomplete_or_invalid_jsonl(
    tmp_path: Path,
    damaged_line: bytes,
):
    run_id = "run-corrupt"
    storage = RunStorage.create(
        tmp_path,
        run_id=run_id,
        spec=_spec(run_id),
        started_at=NOW,
    )
    storage.calls_path.write_bytes(damaged_line)

    with pytest.raises(JournalCorruptionError, match=r"calls\.jsonl line 1"):
        RunStorage.resume(tmp_path, run_id)


def test_resume_rebuilds_stale_summary_from_append_only_journal(tmp_path: Path):
    run_id = "run-rebuild-summary"
    storage = RunStorage.create(
        tmp_path,
        run_id=run_id,
        spec=_spec(run_id),
        started_at=NOW,
        clock=lambda: NOW,
    )
    task = storage.tasks_to_run()[0]
    storage.append_record(_record(storage, task, success=False))
    storage.summary_path.write_text("{}\n", encoding="utf-8")

    resumed = RunStorage.resume(tmp_path, run_id, clock=lambda: NOW)

    summary = json.loads(resumed.summary_path.read_text(encoding="utf-8"))
    assert summary["failed"] == 1
    assert summary["pending"] == 0
    assert summary["error_counts"] == {"TransientJudgeError": 1}


def test_open_read_only_validates_run_without_rewriting_summary(tmp_path: Path):
    run_id = "run-read-only"
    storage = RunStorage.create(
        tmp_path,
        run_id=run_id,
        spec=_spec(run_id),
        started_at=NOW,
    )
    task = storage.tasks_to_run()[0]
    storage.append_record(_record(storage, task, success=False))
    stale_summary = b'{"intentionally":"stale"}\n'
    storage.summary_path.write_bytes(stale_summary)

    reopened = RunStorage.open_read_only(tmp_path, run_id)

    assert reopened.records == storage.records
    assert reopened.latest_records == storage.records
    assert reopened.summary_path.read_bytes() == stale_summary

