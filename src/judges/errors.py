"""Typed failures shared by all judge adapters."""

from __future__ import annotations


class JudgeError(Exception):
    """Base adapter failure with runner-facing retry and abort semantics."""

    retryable = False
    abort_run = False


class JudgeConfigurationError(JudgeError):
    """Missing credentials, unsupported model, or another run-wide setup error."""

    abort_run = True


class TransientJudgeError(JudgeError):
    """Temporary timeout, rate limit, or provider failure that may be retried."""

    retryable = True


class PermanentJudgeTransportError(JudgeError):
    """Non-transient request or transport failure local to a planned vote."""


class JudgeResponseFormatError(JudgeError):
    """Malformed or schema-invalid provider response that may be requested again."""

    retryable = True
