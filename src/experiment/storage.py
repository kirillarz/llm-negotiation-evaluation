"""Crash-resistant, append-only storage for AI-judge experiment runs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import AwareDatetime, Field, JsonValue, ValidationError, model_validator

from .prompts import COLLEGIA
from .schemas import (
    CallRecord,
    Collegium,
    FrozenStrictModel,
    NonEmptyString,
    RepeatIndex,
    TaskKey,
    Variant,
)


ResultKind = Literal["smoke", "pilot", "main"]
DatasetName = Literal["pilot", "main"]
BlockStatus = Literal["pending", "completed", "partial", "failed"]
NonNegativeCount = Annotated[int, Field(strict=True, ge=0)]
PositiveCount = Annotated[int, Field(strict=True, gt=0)]

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REDACTED = "[REDACTED]"
_SENSITIVE_PROVIDER_KEYS = {
    "authorization",
    "proxyauthorization",
    "apikey",
    "accesstoken",
    "refreshtoken",
    "token",
    "credentials",
    "secret",
    "password",
    "cookie",
    "setcookie",
    "request",
    "requestbody",
    "requestpayload",
    "messages",
    "transcript",
    "humanvotes",
}


class StorageError(RuntimeError):
    """A run cannot be created, read, validated, or updated safely."""


class RunCompatibilityError(StorageError):
    """The requested resume configuration differs from persisted metadata."""


class JournalCorruptionError(StorageError):
    """The append-only call journal is truncated or internally inconsistent."""


class RecordRevisionError(StorageError):
    """A record would overwrite or ambiguously revise an existing task."""


class PlannedTaskIdentity(FrozenStrictModel):
    """Secret-free task dimensions persisted in metadata, without a run ID."""

    model_alias: NonEmptyString
    bout_id: NonEmptyString
    variant: Variant
    collegium: Collegium
    repeat_index: RepeatIndex

    @classmethod
    def from_key(cls, key: TaskKey) -> Self:
        return cls(
            model_alias=key.model_alias,
            bout_id=key.bout_id,
            variant=key.variant,
            collegium=key.collegium,
            repeat_index=key.repeat_index,
        )

    def to_key(self, run_id: str) -> TaskKey:
        return TaskKey(run_id=run_id, **self.model_dump())


class RunModelSpec(FrozenStrictModel):
    """Public, reproducibility-relevant model configuration."""

    alias: NonEmptyString
    provider: NonEmptyString
    api_model_id: NonEmptyString
    model_version: NonEmptyString | None = None
    effective_parameters: dict[str, JsonValue]


class RunSpec(FrozenStrictModel):
    """Configuration that must remain identical when a run is resumed."""

    mode: ResultKind
    dataset: DatasetName
    selected_bout_ids: Annotated[tuple[NonEmptyString, ...], Field(min_length=1)]
    models: Annotated[tuple[RunModelSpec, ...], Field(min_length=1)]
    prompt_paths: dict[Collegium, NonEmptyString]
    planned_tasks: Annotated[tuple[PlannedTaskIdentity, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        if self.mode in ("pilot", "main") and self.mode != self.dataset:
            raise ValueError("pilot/main result mode must match the selected dataset")

        if len(set(self.selected_bout_ids)) != len(self.selected_bout_ids):
            raise ValueError("selected_bout_ids must be unique")

        model_aliases = [model.alias for model in self.models]
        if len(set(model_aliases)) != len(model_aliases):
            raise ValueError("model aliases in run metadata must be unique")

        if set(self.prompt_paths) != set(COLLEGIA):
            raise ValueError(
                "prompt_paths must contain exactly: " + ", ".join(COLLEGIA)
            )

        identities = [task_key_tuple(task) for task in self.planned_tasks]
        if len(set(identities)) != len(identities):
            raise ValueError("planned_tasks must not contain duplicate task keys")

        task_aliases = {task.model_alias for task in self.planned_tasks}
        if task_aliases != set(model_aliases):
            raise ValueError("planned task models must exactly match metadata models")

        task_bout_ids = {task.bout_id for task in self.planned_tasks}
        if task_bout_ids != set(self.selected_bout_ids):
            raise ValueError(
                "planned task bout IDs must exactly match selected_bout_ids"
            )
        return self


class RunMetadata(RunSpec):
    """Immutable run manifest written before the first provider call."""

    run_id: NonEmptyString
    started_at: AwareDatetime

    @model_validator(mode="after")
    def check_run_id(self) -> Self:
        validate_run_id(self.run_id)
        return self

    def spec(self) -> RunSpec:
        fields = RunSpec.model_fields
        return RunSpec(**{name: getattr(self, name) for name in fields})


class BlockSummary(FrozenStrictModel):
    """Technical completeness for one bout/model/variant block."""

    bout_id: NonEmptyString
    model_alias: NonEmptyString
    variant: Variant
    planned: PositiveCount
    success: NonNegativeCount
    failed: NonNegativeCount
    pending: NonNegativeCount
    status: BlockStatus

    @model_validator(mode="after")
    def check_counts(self) -> Self:
        if self.success + self.failed + self.pending != self.planned:
            raise ValueError("block summary counts must add up to planned")
        return self


class RunSummary(FrozenStrictModel):
    """Technical-only run state; deliberately contains no winner statistics."""

    run_id: NonEmptyString
    planned: PositiveCount
    success: NonNegativeCount
    failed: NonNegativeCount
    pending: NonNegativeCount
    error_counts: dict[str, NonNegativeCount]
    blocks: tuple[BlockSummary, ...]
    updated_at: AwareDatetime

    @model_validator(mode="after")
    def check_counts(self) -> Self:
        if self.success + self.failed + self.pending != self.planned:
            raise ValueError("run summary counts must add up to planned")
        return self


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_run_id(result_roots: Sequence[Path | str]) -> str:
    """Return the next project-wide numeric run ID across all result roots."""

    if not result_roots:
        raise StorageError("at least one result root is required to allocate a run_id")

    highest = 0
    for configured_root in result_roots:
        root = Path(configured_root).resolve()
        if not root.exists():
            continue
        if not root.is_dir():
            raise StorageError(f"result root is not a directory: {root}")
        try:
            children = tuple(root.iterdir())
        except OSError as error:
            raise StorageError(
                f"cannot inspect result directory {root}: {error}"
            ) from error
        for child in children:
            if child.is_dir() and re.fullmatch(r"[0-9]+", child.name):
                highest = max(highest, int(child.name))

    return f"{highest + 1:03d}"


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise StorageError(
            "run_id must contain only letters, digits, dots, underscores, or hyphens"
        )
    if run_id in {".", ".."}:
        raise StorageError("run_id cannot be a path component")
    return run_id


def task_key_tuple(
    task: TaskKey | PlannedTaskIdentity,
) -> tuple[str, str, str, str, int]:
    return (
        task.model_alias,
        task.bout_id,
        task.variant,
        task.collegium,
        task.repeat_index,
    )


def build_run_spec(
    *,
    mode: ResultKind,
    dataset: DatasetName,
    tasks: Sequence[object],
    prompt_paths: Mapping[Collegium, str],
    model_versions: Mapping[str, str | None] | None = None,
) -> RunSpec:
    """Build secret-free resume metadata from a fully preflighted task plan."""

    if not tasks:
        raise StorageError("cannot create storage for an empty task plan")

    versions = {} if model_versions is None else dict(model_versions)
    identities: list[PlannedTaskIdentity] = []
    bout_ids: list[str] = []
    model_specs: list[RunModelSpec] = []
    seen_models: dict[str, RunModelSpec] = {}
    run_ids: set[str] = set()

    for item in tasks:
        try:
            key = item.key  # type: ignore[attr-defined]
            judge = item.judge  # type: ignore[attr-defined]
        except AttributeError as error:
            raise StorageError("tasks must be PlannedTask instances") from error
        if not isinstance(key, TaskKey):
            raise StorageError("planned task key must be a TaskKey")

        run_ids.add(key.run_id)
        identities.append(PlannedTaskIdentity.from_key(key))
        if key.bout_id not in bout_ids:
            bout_ids.append(key.bout_id)

        candidate = RunModelSpec(
            alias=judge.model_alias,
            provider=judge.provider,
            api_model_id=judge.api_model_id,
            model_version=versions.get(judge.model_alias),
            effective_parameters=judge.effective_parameters,
        )
        previous = seen_models.get(candidate.alias)
        if previous is not None and previous != candidate:
            raise StorageError(
                f"inconsistent effective configuration for model {candidate.alias!r}"
            )
        if previous is None:
            seen_models[candidate.alias] = candidate
            model_specs.append(candidate)

    if len(run_ids) != 1:
        raise StorageError("all planned tasks must use the same run_id")
    unknown_versions = sorted(set(versions) - set(seen_models))
    if unknown_versions:
        raise StorageError(
            "model_versions contains unplanned aliases: " + ", ".join(unknown_versions)
        )

    return RunSpec(
        mode=mode,
        dataset=dataset,
        selected_bout_ids=tuple(bout_ids),
        models=tuple(model_specs),
        prompt_paths=dict(prompt_paths),
        planned_tasks=tuple(identities),
    )


def _json_bytes(value: object, *, pretty: bool) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise StorageError(f"value is not safely JSON-serializable: {error}") from error
    return (text + "\n").encode("utf-8")


def _atomic_write_json(path: Path, value: object) -> None:
    temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("xb") as stream:
            stream.write(_json_bytes(value, pretty=True))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except OSError as error:
        raise StorageError(f"cannot atomically write {path}: {error}") from error
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _normalized_key(key: object) -> str:
    return "".join(character for character in str(key).lower() if character.isalnum())


def _sanitize_provider_value(value: JsonValue) -> JsonValue:
    """Remove commonly echoed credentials and request bodies from provider data."""

    if isinstance(value, dict):
        return {
            str(key): (
                _REDACTED
                if _normalized_key(key) in _SENSITIVE_PROVIDER_KEYS
                else _sanitize_provider_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_provider_value(item) for item in value]
    return value


def _record_payload(record: CallRecord) -> dict[str, JsonValue]:
    payload = record.model_dump(mode="json")
    response = payload.get("response")
    if isinstance(response, dict):
        response["raw_response"] = _sanitize_provider_value(response["raw_response"])
        usage = response.get("usage")
        if isinstance(usage, (dict, list)):
            response["usage"] = _sanitize_provider_value(usage)
    attempts = payload.get("attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            if isinstance(attempt, dict) and attempt.get("raw_response") is not None:
                attempt["raw_response"] = _sanitize_provider_value(
                    attempt["raw_response"]
                )
    return payload


def _read_model(path: Path, model_type: type[RunMetadata]) -> RunMetadata:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise StorageError(f"cannot read {path}: {error}") from error
    try:
        return model_type.model_validate_json(source)
    except ValidationError as error:
        raise StorageError(f"invalid {path.name}: {error}") from error


class RunStorage:
    """One run directory with an immutable manifest and append-only call journal."""

    def __init__(
        self,
        run_dir: Path,
        metadata: RunMetadata,
        records: tuple[CallRecord, ...],
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.run_dir = run_dir
        self.metadata = metadata
        self._clock = clock
        self._records = list(records)
        self._latest = self._validate_record_history(records)

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "metadata.json"

    @property
    def calls_path(self) -> Path:
        return self.run_dir / "calls.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def records(self) -> tuple[CallRecord, ...]:
        return tuple(self._records)

    @classmethod
    def create(
        cls,
        results_root: Path | str,
        *,
        run_id: str,
        spec: RunSpec,
        started_at: datetime | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> Self:
        """Atomically publish a new run directory after caller-side preflight."""

        validated_run_id = validate_run_id(run_id)
        start = started_at or clock()
        metadata = RunMetadata(
            **spec.model_dump(),
            run_id=validated_run_id,
            started_at=start,
        )
        root = Path(results_root).resolve()
        run_dir = root / validated_run_id
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise StorageError(f"cannot create results directory {root}: {error}") from error
        if run_dir.exists():
            raise StorageError(f"run directory already exists: {run_dir}")

        temp_dir = Path(
            tempfile.mkdtemp(prefix=f".{validated_run_id}.", suffix=".tmp", dir=root)
        )
        try:
            storage = cls(temp_dir, metadata, (), clock=clock)
            _atomic_write_json(
                storage.metadata_path, metadata.model_dump(mode="json")
            )
            with storage.calls_path.open("xb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            _atomic_write_json(
                storage.summary_path,
                storage.summary().model_dump(mode="json"),
            )
            temp_dir.rename(run_dir)
        except (OSError, StorageError):
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

        return cls(run_dir, metadata, (), clock=clock)

    @classmethod
    def resume(
        cls,
        results_root: Path | str,
        run_id: str,
        *,
        expected: RunSpec | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> Self:
        """Validate and reopen a run, rebuilding summary from journal truth."""

        validated_run_id = validate_run_id(run_id)
        run_dir = Path(results_root).resolve() / validated_run_id
        if not run_dir.is_dir():
            raise StorageError(f"run directory does not exist: {run_dir}")
        metadata = _read_model(run_dir / "metadata.json", RunMetadata)
        if metadata.run_id != validated_run_id:
            raise JournalCorruptionError(
                "metadata run_id does not match its run directory"
            )
        if expected is not None and metadata.spec() != expected:
            raise RunCompatibilityError(
                "current dataset, models, effective parameters, prompts, or task "
                "plan do not match the persisted run metadata"
            )

        records = cls._read_records(run_dir / "calls.jsonl")
        storage = cls(run_dir, metadata, records, clock=clock)
        _atomic_write_json(
            storage.summary_path,
            storage.summary().model_dump(mode="json"),
        )
        return storage

    @staticmethod
    def _read_records(path: Path) -> tuple[CallRecord, ...]:
        try:
            source = path.read_bytes()
        except OSError as error:
            raise StorageError(f"cannot read {path}: {error}") from error
        if not source:
            return ()

        records: list[CallRecord] = []
        for line_number, line in enumerate(source.splitlines(keepends=True), start=1):
            if not line.endswith((b"\n", b"\r\n")):
                raise JournalCorruptionError(
                    f"calls.jsonl line {line_number} is incomplete (missing newline)"
                )
            body = line.rstrip(b"\r\n")
            if not body.strip():
                raise JournalCorruptionError(
                    f"calls.jsonl line {line_number} is empty"
                )
            try:
                records.append(CallRecord.model_validate_json(body))
            except (ValidationError, UnicodeError) as error:
                raise JournalCorruptionError(
                    f"calls.jsonl line {line_number} is invalid: {error}"
                ) from error
        return tuple(records)

    def _validate_record_history(
        self,
        records: Sequence[CallRecord],
    ) -> dict[tuple[str, str, str, str, int], CallRecord]:
        planned = {
            task_key_tuple(task): task for task in self.metadata.planned_tasks
        }
        models = {model.alias: model for model in self.metadata.models}
        latest: dict[tuple[str, str, str, str, int], CallRecord] = {}

        for record in records:
            identity = task_key_tuple(record.task)
            if record.task.run_id != self.metadata.run_id:
                raise JournalCorruptionError("call record has a foreign run_id")
            if identity not in planned:
                raise JournalCorruptionError("call record is not in the persisted plan")

            model = models[record.task.model_alias]
            expected_prompt = self.metadata.prompt_paths[record.task.collegium]
            if (
                record.provider != model.provider
                or record.api_model_id != model.api_model_id
                or record.model_version != model.model_version
                or record.effective_parameters != model.effective_parameters
                or record.prompt_path != expected_prompt
            ):
                raise JournalCorruptionError(
                    "call record configuration differs from run metadata"
                )

            previous = latest.get(identity)
            if previous is None:
                if (
                    record.record_revision != 1
                    or record.previous_revision is not None
                    or record.manual_retry
                ):
                    raise JournalCorruptionError(
                        "first task record must be non-manual revision 1"
                    )
            else:
                if previous.status == "success":
                    raise JournalCorruptionError(
                        "a successful task has an impermissible later revision"
                    )
                if (
                    not record.manual_retry
                    or record.record_revision != previous.record_revision + 1
                    or record.previous_revision != previous.record_revision
                ):
                    raise JournalCorruptionError(
                        "manual retry revisions must form an unbroken chain"
                    )
            latest[identity] = record
        return latest

    def assert_compatible(self, expected: RunSpec) -> None:
        if self.metadata.spec() != expected:
            raise RunCompatibilityError(
                "current run specification differs from persisted metadata"
            )

    def tasks_to_run(self, *, retry_failed: bool = False) -> tuple[TaskKey, ...]:
        """Return pending tasks, plus failed tasks only when explicitly requested."""

        selected: list[TaskKey] = []
        for identity in self.metadata.planned_tasks:
            latest = self._latest.get(task_key_tuple(identity))
            if latest is None or (retry_failed and latest.status == "failed"):
                selected.append(identity.to_key(self.metadata.run_id))
        return tuple(selected)

    def revision_fields(
        self,
        task: TaskKey,
        *,
        retry_failed: bool = False,
    ) -> dict[str, int | bool | None]:
        """Return the only valid revision markers for a task's next record."""

        self._validate_task_key(task)
        latest = self._latest.get(task_key_tuple(task))
        if latest is None:
            return {
                "record_revision": 1,
                "previous_revision": None,
                "manual_retry": False,
            }
        if latest.status == "success":
            raise RecordRevisionError("a successful task cannot be retried")
        if not retry_failed:
            raise RecordRevisionError(
                "a failed task requires explicit retry_failed=True"
            )
        return {
            "record_revision": latest.record_revision + 1,
            "previous_revision": latest.record_revision,
            "manual_retry": True,
        }

    def append_record(self, record: CallRecord) -> CallRecord:
        """Durably append one completed task revision, then atomically refresh summary."""

        self._validate_record_history((*self._records, record))
        payload = _record_payload(record)
        line = _json_bytes(payload, pretty=False)
        persisted = CallRecord.model_validate_json(line)
        try:
            with self.calls_path.open("ab", buffering=0) as stream:
                stream.write(line)
                os.fsync(stream.fileno())
        except OSError as error:
            raise StorageError(f"cannot append {self.calls_path}: {error}") from error

        self._records.append(persisted)
        self._latest = self._validate_record_history(self._records)
        _atomic_write_json(
            self.summary_path,
            self.summary().model_dump(mode="json"),
        )
        return persisted

    def _validate_task_key(self, task: TaskKey) -> None:
        if task.run_id != self.metadata.run_id:
            raise RecordRevisionError("task run_id does not match storage run_id")
        planned = {task_key_tuple(item) for item in self.metadata.planned_tasks}
        if task_key_tuple(task) not in planned:
            raise RecordRevisionError("task is not part of the persisted plan")

    def summary(self) -> RunSummary:
        success = sum(record.status == "success" for record in self._latest.values())
        failed = sum(record.status == "failed" for record in self._latest.values())
        planned_count = len(self.metadata.planned_tasks)
        pending = planned_count - success - failed

        errors = Counter(
            record.error.error_type
            for record in self._latest.values()
            if record.status == "failed" and record.error is not None
        )

        grouped: dict[tuple[str, str, str], list[PlannedTaskIdentity]] = {}
        for task in self.metadata.planned_tasks:
            block_key = (task.bout_id, task.model_alias, task.variant)
            grouped.setdefault(block_key, []).append(task)

        blocks: list[BlockSummary] = []
        for (bout_id, model_alias, variant), tasks in grouped.items():
            current = [
                self._latest.get(task_key_tuple(task)) for task in tasks
            ]
            block_success = sum(
                record is not None and record.status == "success" for record in current
            )
            block_failed = sum(
                record is not None and record.status == "failed" for record in current
            )
            block_pending = len(tasks) - block_success - block_failed
            if block_success == len(tasks):
                status: BlockStatus = "completed"
            elif block_success > 0:
                status = "partial"
            elif block_failed > 0:
                status = "failed"
            else:
                status = "pending"
            blocks.append(
                BlockSummary(
                    bout_id=bout_id,
                    model_alias=model_alias,
                    variant=variant,
                    planned=len(tasks),
                    success=block_success,
                    failed=block_failed,
                    pending=block_pending,
                    status=status,
                )
            )

        return RunSummary(
            run_id=self.metadata.run_id,
            planned=planned_count,
            success=success,
            failed=failed,
            pending=pending,
            error_counts=dict(sorted(errors.items())),
            blocks=tuple(blocks),
            updated_at=self._clock(),
        )


__all__ = [
    "BlockSummary",
    "JournalCorruptionError",
    "PlannedTaskIdentity",
    "RecordRevisionError",
    "RunCompatibilityError",
    "RunMetadata",
    "RunModelSpec",
    "RunSpec",
    "RunStorage",
    "RunSummary",
    "StorageError",
    "build_run_spec",
    "generate_run_id",
]
