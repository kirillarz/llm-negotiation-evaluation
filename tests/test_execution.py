from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.experiment.config import load_config
from src.experiment.data_loader import load_dataset
from src.experiment.prompts import load_prompts
from src.experiment.runner import build_task_plan, execute_task_plan
from src.experiment.schemas import Collegium, JudgeResponse, PreparedBout
from src.experiment.storage import RunStorage, build_run_spec
from src.judges.base import BaseJudge
from src.judges.errors import (
    JudgeConfigurationError,
    JudgeResponseFormatError,
    PermanentJudgeTransportError,
    TransientJudgeError,
)


NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


class SequenceJudge(BaseJudge):
    SUPPORTED_MODEL_IDS = frozenset({"qwen3.7-flash"})

    def __init__(self, *args, outcomes, events, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcomes = deque(outcomes)
        self.events = events

    def _evaluate_once(
        self, bout: PreparedBout, collegium: Collegium
    ) -> JudgeResponse:
        self.events.append(collegium)
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return JudgeResponse(
            winner=outcome,
            probabilities=None,
            raw_response={"winner": outcome},
        )


def _storage_and_plan(tmp_path: Path, outcomes):
    config = load_config()
    prompts = load_prompts(config.prompts)
    events = []
    judge = SequenceJudge(
        config.models.llm,
        prompts,
        environ={"AITUNNEL_API_KEY": "secret"},
        outcomes=outcomes,
        events=events,
    )
    bouts = load_dataset(config, "pilot", bout_id="1")
    plan = build_task_plan(
        run_id="execution-test",
        bouts=bouts,
        judges=(judge,),
        smoke=True,
        bout_id="1",
        collegium="hiring",
    )
    spec = build_run_spec(
        mode="smoke",
        dataset="pilot",
        tasks=plan,
        prompt_paths=prompts.metadata_paths(),
    )
    ticks = iter(NOW + timedelta(seconds=index) for index in range(100))
    storage = RunStorage.create(
        tmp_path,
        run_id="execution-test",
        spec=spec,
        started_at=NOW,
        clock=lambda: next(ticks),
    )
    return plan, storage, events, ticks


def test_retryable_failure_uses_two_attempts_for_the_same_task(tmp_path: Path):
    plan, storage, events, ticks = _storage_and_plan(
        tmp_path,
        [TransientJudgeError("timeout"), "player_2"],
    )

    result = execute_task_plan(plan, storage, clock=lambda: next(ticks))

    assert not result.aborted
    assert (result.summary.success, result.summary.failed) == (1, 0)
    assert events == ["hiring", "hiring"]
    record = storage.records[0]
    assert [attempt.status for attempt in record.attempts] == ["failed", "success"]
    assert record.response is not None and record.response.winner == "player_2"


def test_response_format_error_retries_but_permanent_error_does_not(tmp_path: Path):
    plan, storage, events, ticks = _storage_and_plan(
        tmp_path,
        [JudgeResponseFormatError("bad JSON"), PermanentJudgeTransportError("400")],
    )

    result = execute_task_plan(plan, storage, clock=lambda: next(ticks))

    assert not result.aborted
    assert (result.summary.success, result.summary.failed) == (0, 1)
    assert len(events) == 2
    assert storage.records[0].error is not None
    assert storage.records[0].error.error_type == "PermanentJudgeTransportError"


def test_configuration_error_is_recorded_without_retry_and_aborts_run(tmp_path: Path):
    config = load_config()
    prompts = load_prompts(config.prompts)
    events = []
    judge = SequenceJudge(
        config.models.llm,
        prompts,
        environ={"AITUNNEL_API_KEY": "secret"},
        outcomes=[JudgeConfigurationError("invalid model"), "player_1"],
        events=events,
    )
    bouts = load_dataset(config, "pilot", bout_id="1")
    plan = build_task_plan(
        run_id="abort-test", bouts=bouts, judges=(judge,)
    )
    spec = build_run_spec(
        mode="pilot",
        dataset="pilot",
        tasks=plan,
        prompt_paths=prompts.metadata_paths(),
    )
    ticks = iter(NOW + timedelta(seconds=index) for index in range(100))
    storage = RunStorage.create(
        tmp_path,
        run_id="abort-test",
        spec=spec,
        started_at=NOW,
        clock=lambda: next(ticks),
    )

    result = execute_task_plan(plan, storage, clock=lambda: next(ticks))

    assert result.aborted
    assert (result.summary.failed, result.summary.pending) == (1, 17)
    assert len(events) == 1
    assert len(storage.calls_path.read_text(encoding="utf-8").splitlines()) == 1


def test_final_local_failure_is_saved_and_later_tasks_continue(tmp_path: Path):
    config = load_config()
    prompts = load_prompts(config.prompts)
    events = []
    judge = SequenceJudge(
        config.models.llm,
        prompts,
        environ={"AITUNNEL_API_KEY": "secret"},
        outcomes=[PermanentJudgeTransportError("bad request"), *(["player_1"] * 17)],
        events=events,
    )
    bouts = load_dataset(config, "pilot", bout_id="1")
    plan = build_task_plan(run_id="continue-test", bouts=bouts, judges=(judge,))
    spec = build_run_spec(
        mode="pilot",
        dataset="pilot",
        tasks=plan,
        prompt_paths=prompts.metadata_paths(),
    )
    ticks = iter(NOW + timedelta(seconds=index) for index in range(200))
    storage = RunStorage.create(
        tmp_path,
        run_id="continue-test",
        spec=spec,
        started_at=NOW,
        clock=lambda: next(ticks),
    )

    result = execute_task_plan(plan, storage, clock=lambda: next(ticks))

    assert not result.aborted
    assert (result.summary.success, result.summary.failed, result.summary.pending) == (
        17,
        1,
        0,
    )
    assert len(events) == 18
    assert len(storage.calls_path.read_text(encoding="utf-8").splitlines()) == 18


def test_tasks_run_sequentially_and_each_result_is_durable_before_the_next(
    tmp_path: Path,
):
    config = load_config()
    prompts = load_prompts(config.prompts)
    observed_journal_lengths: list[int] = []

    class DurabilityJudge(BaseJudge):
        SUPPORTED_MODEL_IDS = frozenset({"qwen3.7-flash"})

        def _evaluate_once(
            self, bout: PreparedBout, collegium: Collegium
        ) -> JudgeResponse:
            observed_journal_lengths.append(
                len(storage.calls_path.read_text(encoding="utf-8").splitlines())
            )
            return JudgeResponse(
                winner="player_1",
                probabilities=None,
                raw_response={"winner": "player_1"},
            )

    judge = DurabilityJudge(
        config.models.llm,
        prompts,
        environ={"AITUNNEL_API_KEY": "secret"},
    )
    bouts = load_dataset(config, "pilot", bout_id="1")
    tasks = build_task_plan(
        run_id="sequential-test", bouts=bouts, judges=(judge,)
    )[:3]
    spec = build_run_spec(
        mode="pilot",
        dataset="pilot",
        tasks=tasks,
        prompt_paths=prompts.metadata_paths(),
    )
    ticks = iter(NOW + timedelta(seconds=index) for index in range(100))
    storage = RunStorage.create(
        tmp_path,
        run_id="sequential-test",
        spec=spec,
        started_at=NOW,
        clock=lambda: next(ticks),
    )

    result = execute_task_plan(tasks, storage, clock=lambda: next(ticks))

    assert not result.aborted
    assert observed_journal_lengths == [0, 1, 2]
    assert [record.task.repeat_index for record in storage.records] == [1, 2, 3]
