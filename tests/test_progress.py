from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO

from src.experiment.progress import RunLogger, event, format_duration


NOW = datetime(2026, 10, 17, 12, 34, 56, tzinfo=timezone.utc)


def test_run_logger_routes_levels_and_appends_utf8_file(tmp_path):
    stdout = StringIO()
    stderr = StringIO()

    with RunLogger(
        tmp_path,
        output=stdout,
        error_output=stderr,
        clock=lambda: NOW,
    ) as logger:
        logger.emit(event("session.start", run_id="001", action="new"))
        logger.emit(
            event(
                "attempt.failed",
                "WARNING",
                error_type="TransientJudgeError",
                message="temporary provider failure",
            )
        )

    assert stdout.getvalue() == (
        "2026-10-17T12:34:56Z INFO session.start run_id=001 action=new\n"
    )
    assert stderr.getvalue() == (
        "2026-10-17T12:34:56Z WARNING attempt.failed "
        'error_type=TransientJudgeError message="temporary provider failure"\n'
    )
    assert (tmp_path / "run.log").read_text(encoding="utf-8") == (
        stdout.getvalue() + stderr.getvalue()
    )

    with RunLogger(
        tmp_path,
        output=StringIO(),
        error_output=StringIO(),
        clock=lambda: NOW,
    ) as logger:
        logger.emit(event("session.finish", run_id="001", status="success"))

    assert (tmp_path / "run.log").read_text(encoding="utf-8").endswith(
        "2026-10-17T12:34:56Z INFO session.finish "
        "run_id=001 status=success\n"
    )


def test_duration_format_is_compact_and_human_readable():
    assert format_duration(None) == "unknown"
    assert format_duration(0.04) == "0.0s"
    assert format_duration(12.34) == "12.3s"
    assert format_duration(120) == "2m00s"
    assert format_duration(3723) == "1h02m03s"
