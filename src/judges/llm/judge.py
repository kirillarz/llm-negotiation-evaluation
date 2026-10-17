"""GigaChat adapter implementing exactly one normalized judge vote."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
from typing import Literal, Protocol

import httpx
from gigachat import GigaChat
from gigachat.exceptions import (
    AuthenticationError,
    BadRequestError,
    ForbiddenError,
    GigaChatException,
    ModelNotSpecifiedError,
    NotFoundError,
    RateLimitError,
    RequestEntityTooLargeError,
    ServerError,
    UnprocessableEntityError,
)
from gigachat.models import Chat
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from src.experiment.config import LLMConfig
from src.experiment.prompts import PromptSet
from src.experiment.schemas import Collegium, JudgeResponse, PreparedBout
from src.judges.base import BaseJudge
from src.judges.errors import (
    JudgeConfigurationError,
    JudgeResponseFormatError,
    PermanentJudgeTransportError,
    TransientJudgeError,
)


class _WinnerPayload(BaseModel):
    """The only substantive field GigaChat is allowed to return."""

    model_config = ConfigDict(extra="forbid", strict=True)

    winner: Literal["player_1", "player_2"]


class _GigaChatClient(Protocol):
    def __enter__(self) -> _GigaChatClient: ...

    def __exit__(self, *args: object) -> None: ...

    def chat(self, payload: Mapping[str, object]) -> object: ...


ClientFactory = Callable[..., _GigaChatClient]

_RESERVED_PARAMETERS = frozenset({"model", "messages", "response_format"})
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "credentials",
        "accesstoken",
        "apikey",
        "cookie",
        "setcookie",
    }
)


class LLMJudge(BaseJudge):
    """Stateless-per-call adapter for the pilot ``GigaChat-2-Pro`` model."""

    SUPPORTED_MODEL_IDS = frozenset({"GigaChat-2-Pro"})
    REQUIRED_PARAMETERS = frozenset({"temperature"})

    def __init__(
        self,
        config: LLMConfig,
        prompts: PromptSet,
        *,
        environ: Mapping[str, str] | None = None,
        client_factory: ClientFactory = GigaChat,
    ) -> None:
        super().__init__(config, prompts, environ=environ)
        self._client_factory = client_factory

    def _validate_provider_configuration(self) -> None:
        conflicts = sorted(_RESERVED_PARAMETERS.intersection(self._config.parameters))
        if conflicts:
            raise JudgeConfigurationError(
                "GigaChat parameters are managed by the adapter and cannot be "
                "configured: " + ", ".join(conflicts)
            )
        if self._config.parameters.get("n", 1) != 1:
            raise JudgeConfigurationError("GigaChat parameter n must be exactly 1")
        if self._config.parameters.get("stream", False) is not False:
            raise JudgeConfigurationError("GigaChat streaming is not supported")
        allowed = set(Chat.model_fields) - _RESERVED_PARAMETERS
        unknown = sorted(set(self._config.parameters) - allowed)
        if unknown:
            raise JudgeConfigurationError(
                "Unsupported GigaChat parameters: " + ", ".join(unknown)
            )
        for name, value in self._config.parameters.items():
            annotation = Chat.model_fields[name].annotation
            try:
                TypeAdapter(annotation).validate_python(value, strict=True)
            except ValidationError as error:
                raise JudgeConfigurationError(
                    f"Invalid GigaChat parameter {name!r}: "
                    f"{error.errors(include_url=False)}"
                ) from error

    def _evaluate_once(
        self,
        bout: PreparedBout,
        collegium: Collegium,
    ) -> JudgeResponse:
        payload = self._request_payload(bout, collegium)
        try:
            # A fresh client and a two-message request make every planned vote
            # independent. SDK retries are disabled because only the runner owns
            # the two-attempt policy and records each technical attempt.
            with self._client_factory(
                credentials=self._secret(),
                timeout=self._config.timeout_seconds,
                verify_ssl_certs=self._config.verify_ssl_certs,
                max_retries=0,
            ) as client:
                response = client.chat(payload)
        except Exception as error:
            self._raise_normalized_transport_error(error)

        raw_response = self._json_value(response, field="GigaChat response")
        try:
            choices = response.choices
            if len(choices) != 1:
                raise ValueError(f"expected one choice, got {len(choices)}")
            content = choices[0].message.content
            if not isinstance(content, str):
                raise TypeError("choice content is not a string")
            parsed = _WinnerPayload.model_validate_json(self._strip_json_fence(content))
            usage_source = getattr(response, "usage", None)
            usage_value = (
                None
                if usage_source is None
                else self._json_value(usage_source, field="GigaChat usage")
            )
            if usage_value is not None and not isinstance(usage_value, dict):
                raise TypeError("usage is not an object")
        except (AttributeError, IndexError, TypeError, ValueError, ValidationError) as error:
            raise JudgeResponseFormatError(
                f"Invalid structured GigaChat response: {error}"
            ) from error

        return JudgeResponse(
            winner=parsed.winner,
            probabilities=None,
            raw_response=raw_response,
            usage=usage_value,
        )

    @property
    def effective_parameters(self) -> dict[str, JsonValue]:
        """Include the explicit TLS policy in reproducibility metadata."""

        parameters = super().effective_parameters
        parameters["verify_ssl_certs"] = self._config.verify_ssl_certs
        return parameters

    def _request_payload(
        self, bout: PreparedBout, collegium: Collegium
    ) -> dict[str, object]:
        prompt = self.prompt_for(collegium)
        bout_json = json.dumps(
            bout.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload: dict[str, object] = deepcopy(self._config.parameters)
        payload.update(
            {
                "model": self.api_model_id,
                "messages": [
                    {"role": "system", "content": prompt.text},
                    {
                        "role": "user",
                        "content": "Полные данные управленческого поединка (JSON):\n" + bout_json,
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "schema": _WinnerPayload.model_json_schema(),
                    "strict": True,
                },
            }
        )
        return payload

    @staticmethod
    def _strip_json_fence(content: str) -> str:
        value = content.strip()
        if value.startswith("```json"):
            value = value[7:]
        elif value.startswith("```"):
            value = value[3:]
        if value.endswith("```"):
            value = value[:-3]
        return value.strip()

    @staticmethod
    def _json_value(value: object, *, field: str) -> JsonValue:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", by_alias=True)
        try:
            # Round-tripping rejects custom objects and returns a detached,
            # storage-safe JSON value without retaining SDK response state.
            serialized = json.loads(json.dumps(value, ensure_ascii=False))
        except (TypeError, ValueError) as error:
            raise JudgeResponseFormatError(f"{field} is not JSON-serializable") from error
        return LLMJudge._redact_sensitive_fields(serialized)

    @staticmethod
    def _redact_sensitive_fields(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            sanitized: dict[str, JsonValue] = {}
            for key, nested in value.items():
                normalized = key.lower().replace("-", "").replace("_", "")
                sanitized[key] = (
                    "[REDACTED]"
                    if normalized in _SENSITIVE_KEYS
                    else LLMJudge._redact_sensitive_fields(nested)
                )
            return sanitized
        if isinstance(value, list):
            return [LLMJudge._redact_sensitive_fields(item) for item in value]
        return value

    @staticmethod
    def _raise_normalized_transport_error(error: Exception) -> None:
        if isinstance(error, (AuthenticationError, ForbiddenError, NotFoundError, ModelNotSpecifiedError)):
            raise JudgeConfigurationError(
                f"GigaChat rejected credentials or model configuration ({type(error).__name__})"
            ) from error
        if isinstance(error, (RateLimitError, ServerError, httpx.TimeoutException, httpx.TransportError)):
            raise TransientJudgeError(
                f"Temporary GigaChat failure ({type(error).__name__})"
            ) from error
        if isinstance(
            error,
            (BadRequestError, RequestEntityTooLargeError, UnprocessableEntityError),
        ):
            raise PermanentJudgeTransportError(
                f"GigaChat rejected the request ({type(error).__name__})"
            ) from error
        if isinstance(error, GigaChatException):
            raise PermanentJudgeTransportError(
                f"GigaChat SDK failure ({type(error).__name__})"
            ) from error
        raise error


__all__ = ["LLMJudge"]
