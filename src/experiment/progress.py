"""Typed operational progress events and human-readable run logging."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Literal, Protocol, TextIO


LogLevel = Literal["INFO", "WARNING", "ERROR"]
EventName = Literal[
    "session.start",
    "block.start",
    "attempt.failed",
    "attempt.retry",
    "task.failed",
    "task.interrupted",
    "block.finish",
    "session.finish",
    "session.interrupted",
    "session.aborted",
]
LogValue = str | int | float | bool | None

_UNQUOTED_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/\\-]*$")


class OperationalLogError(RuntimeError):
    """The run-local operational journal cannot be opened or updated."""


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """One structured, secret-free event emitted by the execution pipeline."""

    name: EventName
    level: LogLevel
    fields: tuple[tuple[str, LogValue], ...] = ()


class ExecutionObserver(Protocol):
    """Consumer for operational execution events."""

    def emit(self, event: ExecutionEvent) -> None: ...


def event(
    name: EventName,
    level: LogLevel = "INFO",
    **fields: LogValue,
) -> ExecutionEvent:
    """Build an event while preserving the caller-defined field order."""

    return ExecutionEvent(name=name, level=level, fields=tuple(fields.items()))


def format_duration(seconds: float | None) -> str:
    """Return a compact duration suitable for an operational log field."""

    if seconds is None:
        return "unknown"
    value = max(0.0, seconds)
    if value < 60:
        return f"{value:.1f}s"
    rounded = int(round(value))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


class RunLogger:
    """Write events to the console and append them to one run-local log file."""

    def __init__(
        self,
        run_dir: Path,
        *,
        output: TextIO,
        error_output: TextIO,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = run_dir / "run.log"
        self._output = output
        self._error_output = error_output
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        try:
            self._file = self.path.open("a", encoding="utf-8", newline="\n")
        except OSError as error:
            raise OperationalLogError(
                f"cannot open operational log {self.path}"
            ) from error

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._file.close()
        except OSError as error:
            raise OperationalLogError(
                f"cannot close operational log {self.path}"
            ) from error

    def emit(self, emitted: ExecutionEvent) -> None:
        line = self._format(emitted)
        console = self._output if emitted.level == "INFO" else self._error_output
        print(line, file=console, flush=True)
        try:
            print(line, file=self._file, flush=True)
        except OSError as error:
            raise OperationalLogError(
                f"cannot write operational log {self.path}"
            ) from error

    def _format(self, emitted: ExecutionEvent) -> str:
        timestamp = self._clock().astimezone(timezone.utc).isoformat(timespec="seconds")
        timestamp = timestamp.replace("+00:00", "Z")
        suffix = " ".join(
            f"{key}={self._format_value(value)}" for key, value in emitted.fields
        )
        prefix = f"{timestamp} {emitted.level} {emitted.name}"
        return prefix if not suffix else f"{prefix} {suffix}"

    @staticmethod
    def _format_value(value: LogValue) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if _UNQUOTED_VALUE.fullmatch(value):
            return value
        return json.dumps(value, ensure_ascii=False)


__all__ = [
    "EventName",
    "ExecutionEvent",
    "ExecutionObserver",
    "LogLevel",
    "LogValue",
    "OperationalLogError",
    "RunLogger",
    "event",
    "format_duration",
]
