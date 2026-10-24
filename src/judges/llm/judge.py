"""AITunnel Qwen adapter for one normalized judge vote."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
from math import isfinite
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError
import requests

from src.experiment.config import LLMConfig
from src.experiment.data_loader import swap_players
from src.experiment.prompts import PromptSet
from src.experiment.schemas import Collegium, JudgeResponse, Player, PreparedBout
from src.judges.base import BaseJudge
from src.judges.errors import (
    JudgeConfigurationError,
    JudgeResponseFormatError,
    PermanentJudgeTransportError,
    TransientJudgeError,
)


class _ChoicePayload(BaseModel):
    """The only content Qwen may return."""

    model_config = ConfigDict(extra="forbid", strict=True)
    choice: Literal["player_1", "player_2"]


class _HTTPResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> object: ...


PostCallable = Callable[..., _HTTPResponse]
_RESERVED_PARAMETERS = frozenset(
    {"model", "messages", "response_format", "stream", "n"}
)
_SENSITIVE_KEYS = frozenset(
    {"authorization", "credentials", "accesstoken", "apikey", "cookie", "setcookie"}
)
_NON_NEGATIVE_USAGE_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cost_rub",
    }
)


class LLMJudge(BaseJudge):
    """Stateless Qwen adapter through AITunnel chat completions."""

    SUPPORTED_MODEL_IDS = frozenset({"qwen3.7-flash"})
    REQUIRED_PARAMETERS = frozenset({"max_tokens"})

    def __init__(
        self,
        config: LLMConfig,
        prompts: PromptSet,
        *,
        environ: Mapping[str, str] | None = None,
        post: PostCallable = requests.post,
    ) -> None:
        super().__init__(config, prompts, environ=environ)
        self._post = post

    def _validate_provider_configuration(self) -> None:
        conflicts = sorted(_RESERVED_PARAMETERS.intersection(self._config.parameters))
        if conflicts:
            raise JudgeConfigurationError(
                "Qwen request fields are managed by the adapter and cannot "
                "be configured: " + ", ".join(conflicts)
            )
        max_tokens = self._config.parameters.get("max_tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 16:
            raise JudgeConfigurationError("max_tokens must be an integer >= 16")
        reasoning = self._config.parameters.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, dict):
                raise JudgeConfigurationError("reasoning must be an object")
            unknown = sorted(set(reasoning) - {"effort", "exclude", "context", "mode"})
            if unknown:
                raise JudgeConfigurationError(
                    "Unsupported reasoning fields: " + ", ".join(unknown)
                )
            effort = reasoning.get("effort")
            if effort is not None and effort not in {
                "none", "minimal", "low", "medium", "high", "xhigh", "max"
            }:
                raise JudgeConfigurationError("reasoning.effort is unsupported")
            exclude = reasoning.get("exclude")
            if exclude is not None and not isinstance(exclude, bool):
                raise JudgeConfigurationError("reasoning.exclude must be boolean")

    def _evaluate_once(
        self, bout: PreparedBout, collegium: Collegium
    ) -> JudgeResponse:
        secret = self._secret()
        canonical_bout, canonical_labels_swapped = self._canonicalize_bout(bout)
        try:
            response = self._post(
                str(self._config.api_url),
                headers={
                    "Authorization": f"Bearer {secret}",
                    "Content-Type": "application/json",
                },
                json=self._request_payload(canonical_bout, collegium),
                timeout=self._config.timeout_seconds,
                allow_redirects=False,
            )
        except requests.Timeout as error:
            raise TransientJudgeError("AITunnel Qwen request timed out") from error
        except requests.ConnectionError as error:
            raise TransientJudgeError(
                "AITunnel Qwen connection was interrupted"
            ) from error
        except requests.RequestException as error:
            raise PermanentJudgeTransportError(
                f"AITunnel Qwen transport failure ({type(error).__name__})"
            ) from error

        self._check_http_status(response.status_code)
        try:
            decoded = response.json()
        except (TypeError, ValueError) as error:
            raise JudgeResponseFormatError("AITunnel Qwen returned invalid JSON") from error
        raw_response = self._json_value(decoded, secret=secret)
        if not isinstance(raw_response, dict):
            raise JudgeResponseFormatError("AITunnel response must be a JSON object")

        try:
            choices = raw_response["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("expected exactly one choice")
            choice_record = choices[0]
            if not isinstance(choice_record, dict):
                raise TypeError("choices[0] must be an object")
            finish_reason = choice_record.get("finish_reason")
            if finish_reason != "stop":
                raise ValueError(f"finish_reason must be stop, got {finish_reason!r}")
            message = choice_record["message"]
            if not isinstance(message, dict):
                raise TypeError("choices[0].message must be an object")
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError("message.content must be a string")
            parsed = _ChoicePayload.model_validate_json(content)
            usage = self._usage(raw_response.get("usage"))
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise JudgeResponseFormatError(
                f"Invalid structured AITunnel Qwen response: {error}"
            ) from error

        raw_response["_adapter"] = {
            "canonical_labels_swapped": canonical_labels_swapped,
        }
        return JudgeResponse(
            winner=(
                self._swap_player(parsed.choice)
                if canonical_labels_swapped
                else parsed.choice
            ),
            probabilities=None,
            raw_response=raw_response,
            usage=usage,
        )

    @staticmethod
    def _player_profile(bout: PreparedBout, player: Player) -> tuple[object, ...]:
        """Build the same label-independent ordering key used by Clef."""

        return tuple(
            (
                getattr(round_.roles, player),
                tuple(
                    (turn.turn_id, turn.text)
                    for turn in round_.turns
                    if turn.speaker == player
                ),
            )
            for round_ in bout.rounds
        )

    @classmethod
    def _canonicalize_bout(cls, bout: PreparedBout) -> tuple[PreparedBout, bool]:
        player_1_profile = cls._player_profile(bout, "player_1")
        player_2_profile = cls._player_profile(bout, "player_2")
        if player_1_profile <= player_2_profile:
            return bout, False
        return swap_players(bout), True

    @staticmethod
    def _swap_player(player: Player) -> Player:
        return "player_2" if player == "player_1" else "player_1"

    def _request_payload(
        self, bout: PreparedBout, collegium: Collegium
    ) -> dict[str, object]:
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
                    {"role": "system", "content": self.prompt_for(collegium).text},
                    {
                        "role": "user",
                        "content": (
                            "Полные данные управленческого поединка (JSON):\n"
                            + bout_json
                            + "\nВыбери ровно одного игрока. Верни только JSON "
                            + '{"choice":"player_1"} или {"choice":"player_2"}.'
                        ),
                    },
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "judge_vote",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "choice": {
                                    "type": "string",
                                    "enum": ["player_1", "player_2"],
                                }
                            },
                            "required": ["choice"],
                            "additionalProperties": False,
                        },
                    },
                },
            }
        )
        return payload

    @staticmethod
    def _usage(value: JsonValue | None) -> dict[str, JsonValue] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("usage must be an object")
        for field in _NON_NEGATIVE_USAGE_FIELDS.intersection(value):
            measurement = value[field]
            if (
                not isinstance(measurement, (int, float))
                or isinstance(measurement, bool)
                or not isfinite(measurement)
                or measurement < 0
            ):
                raise ValueError(f"usage.{field} must be a finite non-negative number")
        return value

    @staticmethod
    def _check_http_status(status_code: int) -> None:
        if 200 <= status_code < 300:
            return
        if status_code == 429 or 500 <= status_code < 600:
            raise TransientJudgeError(f"Temporary AITunnel Qwen HTTP {status_code}")
        if status_code in (401, 402, 403, 404):
            raise JudgeConfigurationError(
                "AITunnel rejected credentials, balance, endpoint, or model "
                f"(HTTP {status_code})"
            )
        raise PermanentJudgeTransportError(
            f"AITunnel rejected the Qwen request (HTTP {status_code})"
        )

    @staticmethod
    def _json_value(value: object, *, secret: str) -> JsonValue:
        try:
            detached = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise JudgeResponseFormatError(
                "AITunnel response is not safely JSON-serializable"
            ) from error
        return LLMJudge._redact_sensitive_fields(detached, secret=secret)

    @staticmethod
    def _redact_sensitive_fields(value: JsonValue, *, secret: str) -> JsonValue:
        if isinstance(value, dict):
            sanitized: dict[str, JsonValue] = {}
            for key, nested in value.items():
                normalized = key.lower().replace("-", "").replace("_", "")
                sanitized[key] = (
                    "[REDACTED]"
                    if normalized in _SENSITIVE_KEYS
                    else LLMJudge._redact_sensitive_fields(nested, secret=secret)
                )
            return sanitized
        if isinstance(value, list):
            return [
                LLMJudge._redact_sensitive_fields(item, secret=secret)
                for item in value
            ]
        if isinstance(value, str) and secret:
            return value.replace(secret, "[REDACTED]")
        return value


__all__ = ["LLMJudge"]
