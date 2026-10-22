"""Deterministic planning and sequential execution for AI-judge votes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from collections.abc import Callable
from itertools import groupby
from time import monotonic
from typing import TYPE_CHECKING

from pydantic import JsonValue

from .data_loader import PreparedBoutVariants
from .progress import ExecutionObserver, event, format_duration
from .prompts import COLLEGIA, Prompt
from .schemas import (
    AttemptError,
    CallRecord,
    Collegium,
    JudgeResponse,
    PreparedBout,
    TaskKey,
    TechnicalAttempt,
    Variant,
)
from .storage import RunStorage, RunSummary, task_key_tuple, utc_now
from src.judges.errors import JudgeError

if TYPE_CHECKING:
    from src.judges.base import BaseJudge


VARIANTS: tuple[Variant, ...] = ("original", "swapped")
REPEAT_INDICES: tuple[int, ...] = (1,)


class TaskPlanError(ValueError):
    """A requested experiment plan is empty, ambiguous, or inconsistent."""


@dataclass(frozen=True, slots=True)
class PlannedTask:
    """One planned vote bound to immutable input and one configured adapter."""

    key: TaskKey
    bout: PreparedBout
    judge: BaseJudge

    @property
    def prompt(self) -> Prompt:
        """Resolve the shared prompt solely from the stable collegium key."""

        return self.judge.prompt_for(self.key.collegium)

    @property
    def effective_parameters(self) -> dict[str, JsonValue]:
        """Return the adapter's fixed, secret-free parameter snapshot."""

        return self.judge.effective_parameters


@dataclass(frozen=True, slots=True)
class RunExecutionResult:
    """Technical outcome of executing all currently eligible tasks."""

    summary: RunSummary
    abort_error: AttemptError | None = None

    @property
    def aborted(self) -> bool:
        return self.abort_error is not None


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id.strip():
        raise TaskPlanError("run_id must be a non-empty string")
    return run_id.strip()


def _validate_bouts(
    bouts: tuple[PreparedBoutVariants, ...],
) -> tuple[PreparedBoutVariants, ...]:
    if not bouts:
        raise TaskPlanError("Cannot build a task plan without prepared bouts")
    ids = [bout.bout_id for bout in bouts]
    duplicates = sorted({bout_id for bout_id in ids if ids.count(bout_id) > 1})
    if duplicates:
        raise TaskPlanError(
            "Prepared bout IDs must be unique; duplicates: " + ", ".join(duplicates)
        )
    return bouts


def _validate_judges(judges: tuple[BaseJudge, ...]) -> tuple[BaseJudge, ...]:
    if not judges:
        raise TaskPlanError("Cannot build a task plan without judge adapters")
    aliases = [judge.model_alias for judge in judges]
    duplicates = sorted({alias for alias in aliases if aliases.count(alias) > 1})
    if duplicates:
        raise TaskPlanError(
            "Judge model aliases must be unique; duplicates: " + ", ".join(duplicates)
        )
    return judges


def _select_bouts(
    bouts: tuple[PreparedBoutVariants, ...],
    bout_id: str | None,
) -> tuple[PreparedBoutVariants, ...]:
    if bout_id is None:
        return bouts
    if not isinstance(bout_id, str) or not bout_id.strip():
        raise TaskPlanError("bout_id must be a non-empty string when provided")
    requested = bout_id.strip()
    matches = tuple(bout for bout in bouts if bout.bout_id == requested)
    if not matches:
        available = ", ".join(bout.bout_id for bout in bouts)
        raise TaskPlanError(
            f"Bout ID {requested!r} is not available for planning; "
            f"available IDs: {available}"
        )
    return matches


def _smoke_dimensions(
    *,
    smoke: bool,
    bout_id: str | None,
    collegium: Collegium | None,
) -> tuple[tuple[Variant, ...], tuple[Collegium, ...], tuple[int, ...]]:
    if smoke:
        if bout_id is None or not isinstance(bout_id, str) or not bout_id.strip():
            raise TaskPlanError("Smoke mode requires an explicit bout_id")
        if collegium is None:
            raise TaskPlanError("Smoke mode requires an explicit collegium")
        if collegium not in COLLEGIA:
            raise TaskPlanError(
                f"Unknown collegium {collegium!r}; expected one of: "
                + ", ".join(COLLEGIA)
            )
        return (("original",), (collegium,), (1,))

    if collegium is not None:
        raise TaskPlanError("collegium can only restrict a smoke plan")
    return (VARIANTS, COLLEGIA, REPEAT_INDICES)


def build_task_plan(
    *,
    run_id: str,
    bouts: tuple[PreparedBoutVariants, ...],
    judges: tuple[BaseJudge, ...],
    smoke: bool = False,
    bout_id: str | None = None,
    collegium: Collegium | None = None,
) -> tuple[PlannedTask, ...]:
    """Build tasks in stable bout/model/variant/collegium/repeat order.

    The caller supplies ordered tuples produced by dataset and model selection.
    Repeats bind to the same frozen ``PreparedBout`` and adapter instance, so their
    prompt and effective parameters are identical and no response/history is part
    of a planned task.
    """

    validated_run_id = _validate_run_id(run_id)
    validated_bouts = _validate_bouts(bouts)
    validated_judges = _validate_judges(judges)
    variants, collegia, repeat_indices = _smoke_dimensions(
        smoke=smoke,
        bout_id=bout_id,
        collegium=collegium,
    )
    selected_bouts = _select_bouts(validated_bouts, bout_id)

    tasks: list[PlannedTask] = []
    for bout_variants in selected_bouts:
        for judge in validated_judges:
            for variant in variants:
                prepared_bout = bout_variants.get(variant)
                for selected_collegium in collegia:
                    for repeat_index in repeat_indices:
                        key = TaskKey(
                            run_id=validated_run_id,
                            model_alias=judge.model_alias,
                            bout_id=bout_variants.bout_id,
                            variant=variant,
                            collegium=selected_collegium,
                            repeat_index=repeat_index,
                        )
                        tasks.append(
                            PlannedTask(key=key, bout=prepared_bout, judge=judge)
                        )
    return tuple(tasks)


def _attempt_error(error: Exception) -> AttemptError:
    """Normalize an adapter failure without persisting exception internals."""

    if isinstance(error, JudgeError):
        retryable = error.retryable
        abort_run = error.abort_run
    else:
        # Untyped adapter failures are local and non-retryable. Provider adapters
        # must deliberately translate retryable and run-wide failures.
        retryable = False
        abort_run = False
    message = str(error).strip() or type(error).__name__
    return AttemptError(
        error_type=type(error).__name__,
        message=message,
        retryable=retryable,
        abort_run=abort_run,
    )


def _operational_error_message(error: AttemptError) -> str:
    """Describe a failure without copying arbitrary provider or input text."""

    messages = {
        "JudgeConfigurationError": "judge configuration rejected during execution",
        "TransientJudgeError": "temporary provider failure",
        "JudgeResponseFormatError": "invalid provider response",
        "PermanentJudgeTransportError": "provider rejected the request",
    }
    return messages.get(error.error_type, "unexpected adapter failure")


def _task_log_fields(task: PlannedTask) -> dict[str, str | int]:
    return {
        "model": task.key.model_alias,
        "bout": task.key.bout_id,
        "variant": task.key.variant,
        "collegium": task.key.collegium,
        "repeat": task.key.repeat_index,
    }


def execute_task(
    task: PlannedTask,
    storage: RunStorage,
    *,
    max_attempts: int = 2,
    retry_failed: bool = False,
    model_version: str | None = None,
    clock: Callable[[], datetime] = utc_now,
    observer: ExecutionObserver | None = None,
) -> CallRecord:
    """Execute and durably record one planned vote.

    The same immutable task is reused for a retry. ``KeyboardInterrupt`` is only
    observed and re-raised, so a partially started vote remains pending and can
    be resumed.
    """

    if max_attempts != 2:
        raise ValueError("max_attempts must be exactly 2")

    started_at = clock()
    attempts: list[TechnicalAttempt] = []
    response: JudgeResponse | None = None
    final_error: AttemptError | None = None

    for attempt_index in range(1, max_attempts + 1):
        attempt_started_at = clock()
        try:
            response = task.judge.evaluate(
                bout=task.bout,
                collegium=task.key.collegium,
            )
        except KeyboardInterrupt:
            if observer is not None:
                observer.emit(
                    event(
                        "task.interrupted",
                        "WARNING",
                        **_task_log_fields(task),
                        attempt=attempt_index,
                        max_attempts=max_attempts,
                    )
                )
            raise
        except Exception as error:
            attempt_finished_at = clock()
            final_error = _attempt_error(error)
            attempts.append(
                TechnicalAttempt(
                    attempt_index=attempt_index,
                    status="failed",
                    started_at=attempt_started_at,
                    finished_at=attempt_finished_at,
                    error=final_error,
                )
            )
            will_retry = final_error.retryable and not final_error.abort_run and (
                attempt_index < max_attempts
            )
            if observer is not None:
                observer.emit(
                    event(
                        "attempt.failed",
                        "WARNING",
                        **_task_log_fields(task),
                        attempt=attempt_index,
                        max_attempts=max_attempts,
                        error_type=final_error.error_type,
                        message=_operational_error_message(final_error),
                        retry=will_retry,
                    )
                )
                if will_retry:
                    observer.emit(
                        event(
                            "attempt.retry",
                            "WARNING",
                            **_task_log_fields(task),
                            next_attempt=attempt_index + 1,
                            max_attempts=max_attempts,
                        )
                    )
            if not will_retry:
                break
        else:
            attempt_finished_at = clock()
            attempts.append(
                TechnicalAttempt(
                    attempt_index=attempt_index,
                    status="success",
                    started_at=attempt_started_at,
                    finished_at=attempt_finished_at,
                    raw_response=response.raw_response,
                )
            )
            final_error = None
            break

    finished_at = clock()
    common = {
        "task": task.key,
        **storage.revision_fields(task.key, retry_failed=retry_failed),
        "provider": task.judge.provider,
        "api_model_id": task.judge.api_model_id,
        "model_version": model_version,
        "prompt_path": str(task.prompt.path),
        "effective_parameters": task.effective_parameters,
        "started_at": started_at,
        "finished_at": finished_at,
        "attempts": tuple(attempts),
    }
    if response is not None:
        record = CallRecord(status="success", response=response, **common)
    else:
        if final_error is None:  # defensive: the loop always sets one on failure
            raise RuntimeError("task execution ended without a response or error")
        record = CallRecord(status="failed", error=final_error, **common)
    persisted = storage.append_record(record)
    if persisted.error is not None and observer is not None:
        observer.emit(
            event(
                "task.failed",
                "ERROR",
                **_task_log_fields(task),
                attempts=len(persisted.attempts),
                error_type=persisted.error.error_type,
                message=_operational_error_message(persisted.error),
                abort_run=persisted.error.abort_run,
            )
        )
    return persisted


def execute_task_plan(
    tasks: tuple[PlannedTask, ...],
    storage: RunStorage,
    *,
    max_attempts: int = 2,
    retry_failed: bool = False,
    model_versions: dict[str, str | None] | None = None,
    clock: Callable[[], datetime] = utc_now,
    timer: Callable[[], float] = monotonic,
    observer: ExecutionObserver | None = None,
) -> RunExecutionResult:
    """Sequentially execute pending tasks and stop on run-wide configuration errors."""

    versions = {} if model_versions is None else model_versions
    by_key = {task_key_tuple(task.key): task for task in tasks}
    selected = storage.tasks_to_run(retry_failed=retry_failed)

    selected_tasks: list[PlannedTask] = []
    for key in selected:
        task = by_key.get(task_key_tuple(key))
        if task is None:
            raise TaskPlanError("storage contains a task absent from the current plan")
        selected_tasks.append(task)

    block_groups = tuple(
        (block_key, tuple(group))
        for block_key, group in groupby(
            selected_tasks,
            key=lambda task: (
                task.key.bout_id,
                task.key.model_alias,
                task.key.variant,
            ),
        )
    )
    session_started = timer()
    session_processed = 0

    for block_index, ((bout_id, model_alias, variant), block_tasks) in enumerate(
        block_groups, start=1
    ):
        block_started = timer()
        manual_retries = sum(
            bool(
                storage.revision_fields(
                    task.key, retry_failed=retry_failed
                )["manual_retry"]
            )
            for task in block_tasks
        )
        if observer is not None:
            observer.emit(
                event(
                    "block.start",
                    block=f"{block_index}/{len(block_groups)}",
                    bout=bout_id,
                    model=model_alias,
                    variant=variant,
                    tasks=len(block_tasks),
                    manual_retries=manual_retries,
                    request_timeout=format_duration(
                        float(block_tasks[0].judge.request_timeout_seconds)
                    ),
                    max_attempts=max_attempts,
                )
            )

        block_success = 0
        block_failed = 0
        abort_error: AttemptError | None = None
        for task in block_tasks:
            record = execute_task(
                task,
                storage,
                max_attempts=max_attempts,
                retry_failed=retry_failed,
                model_version=versions.get(task.key.model_alias),
                clock=clock,
                observer=observer,
            )
            session_processed += 1
            if record.status == "success":
                block_success += 1
            else:
                block_failed += 1
            if record.error is not None and record.error.abort_run:
                abort_error = record.error
                break

        block_finished = timer()
        elapsed = max(0.0, block_finished - session_started)
        remaining = len(selected_tasks) - session_processed
        eta = (
            elapsed / session_processed * remaining
            if session_processed and abort_error is None
            else None
        )
        summary = storage.summary()
        if observer is not None:
            observer.emit(
                event(
                    "block.finish",
                    "WARNING" if block_failed else "INFO",
                    block=f"{block_index}/{len(block_groups)}",
                    bout=bout_id,
                    model=model_alias,
                    variant=variant,
                    processed=block_success + block_failed,
                    success=block_success,
                    failed=block_failed,
                    manual_retries=manual_retries,
                    progress=f"{session_processed}/{len(selected_tasks)}",
                    percent=round(
                        100.0 * session_processed / len(selected_tasks), 1
                    ),
                    run_success=summary.success,
                    run_failed=summary.failed,
                    run_pending=summary.pending,
                    duration=format_duration(
                        max(0.0, block_finished - block_started)
                    ),
                    elapsed=format_duration(elapsed),
                    eta=format_duration(eta),
                )
            )
        if abort_error is not None:
            return RunExecutionResult(summary=summary, abort_error=abort_error)

    return RunExecutionResult(summary=storage.summary())


__all__ = [
    "PlannedTask",
    "REPEAT_INDICES",
    "RunExecutionResult",
    "TaskPlanError",
    "VARIANTS",
    "build_task_plan",
    "execute_task",
    "execute_task_plan",
]
