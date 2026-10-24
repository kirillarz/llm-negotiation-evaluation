"""AITunnel Clef adapter implementing exactly one normalized judge vote."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
from math import isfinite
from typing import Protocol

from pydantic import JsonValue, ValidationError
import requests

from src.experiment.config import DecisionConfig
from src.experiment.prompts import PromptSet
from src.experiment.schemas import (
    Collegium,
    JudgeResponse,
    PlayerProbabilities,
    PreparedBout,
)
from src.judges.base import BaseJudge
from src.judges.errors import (
    JudgeConfigurationError,
    JudgeResponseFormatError,
    PermanentJudgeTransportError,
    TransientJudgeError,
)


class _HTTPResponse(Protocol):
    status_code: int

    def json(self) -> object: ...


PostCallable = Callable[..., _HTTPResponse]

_PLAYERS = ("player_1", "player_2")
_COLLEGIUM_TARGETS = {
    "hiring": "руководителя, к которому вы лично предпочтёте наняться",
    "negotiation": "представителя, которого вы отправите на сложные переговоры",
    "ownership": "управляющего, которому вы доверите свою собственность или ресурс",
}
_RESERVED_PARAMETERS = frozenset({"model", "state", "questions"})
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
_NON_NEGATIVE_USAGE_FIELDS = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "cost_rub"}
)


class DecisionJudge(BaseJudge):
    """Stateless-per-call adapter for the pilot Clef model through AITunnel."""

    SUPPORTED_MODEL_IDS = frozenset({"clef"})

    def __init__(
        self,
        config: DecisionConfig,
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
                "Clef parameters are managed by the adapter and cannot be "
                "configured: " + ", ".join(conflicts)
            )

    def _evaluate_once(
        self,
        bout: PreparedBout,
        collegium: Collegium,
    ) -> JudgeResponse:
        secret = self._secret()
        try:
            response = self._post(
                str(self._config.api_url),
                headers={
                    "Authorization": f"Bearer {secret}",
                    "Content-Type": "application/json",
                },
                json=self._request_payload(bout, collegium),
                timeout=self._config.timeout_seconds,
                allow_redirects=False,
            )
        except requests.Timeout as error:
            raise TransientJudgeError("AITunnel request timed out") from error
        except requests.ConnectionError as error:
            raise TransientJudgeError("AITunnel connection was interrupted") from error
        except requests.RequestException as error:
            raise PermanentJudgeTransportError(
                f"AITunnel transport failure ({type(error).__name__})"
            ) from error

        self._check_http_status(response.status_code)
        try:
            decoded = response.json()
        except (TypeError, ValueError) as error:
            raise JudgeResponseFormatError(
                "AITunnel returned invalid JSON"
            ) from error

        raw_response = self._json_value(decoded, secret=secret)
        if not isinstance(raw_response, dict):
            raise JudgeResponseFormatError("AITunnel response must be a JSON object")

        try:
            answers = raw_response["answers"]
            if not isinstance(answers, dict):
                raise TypeError("answers must be an object")
            vote = answers["vote"]
            if not isinstance(vote, dict):
                raise TypeError("answers.vote must be an object")

            provider_choice = vote["choice"]
            if provider_choice not in _PLAYERS:
                raise ValueError("answers.vote.choice must be player_1 or player_2")
            if "type" in vote and vote["type"] != "choice":
                raise ValueError("answers.vote.type must be choice")

            probabilities = self._probabilities(vote.get("probabilities"))
            usage = self._usage(raw_response.get("usage"))
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise JudgeResponseFormatError(
                f"Invalid structured AITunnel response: {error}"
            ) from error

        return JudgeResponse(
            winner=provider_choice,
            probabilities=probabilities,
            raw_response=raw_response,
            usage=usage,
        )

    def _request_payload(
        self, bout: PreparedBout, collegium: Collegium
    ) -> dict[str, object]:
        payload: dict[str, object] = deepcopy(self._config.parameters)
        payload.update(
            {
                "model": self.api_model_id,
                "state": bout.model_dump(mode="json"),
                "questions": {
                    "vote": {
                        "type": "choice",
                        "instructions": self.prompt_for(collegium).text,
                        "criteria": {
                            player: (
                                f"Выбрать {player}, только если именно действия этого "
                                f"человека делают его предпочтительнее как "
                                f"{_COLLEGIUM_TARGETS[collegium]}; номер метки, порядок "
                                "предъявления и общая победа в поединке не являются "
                                "критериями"
                            )
                            for player in _PLAYERS
                        },
                    }
                },
            }
        )
        return payload

    @staticmethod
    def _probabilities(value: JsonValue | None) -> PlayerProbabilities | None:
        if value is None:
            return None
        if not isinstance(value, dict) or set(value) != set(_PLAYERS):
            raise ValueError(
                "answers.vote.probabilities must contain exactly player_1 and player_2"
            )
        normalized: dict[str, float] = {}
        for player in _PLAYERS:
            probability = value[player]
            if (
                not isinstance(probability, (int, float))
                or isinstance(probability, bool)
                or not isfinite(probability)
            ):
                raise TypeError(f"probability for {player} must be a finite number")
            normalized[player] = float(probability)
        return PlayerProbabilities.model_validate(normalized)

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
            raise TransientJudgeError(f"Temporary AITunnel HTTP {status_code}")
        if status_code in (401, 403, 404):
            raise JudgeConfigurationError(
                f"AITunnel rejected credentials, endpoint, or model (HTTP {status_code})"
            )
        raise PermanentJudgeTransportError(
            f"AITunnel rejected the request (HTTP {status_code})"
        )

    @staticmethod
    def _json_value(value: object, *, secret: str) -> JsonValue:
        try:
            detached = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError) as error:
            raise JudgeResponseFormatError(
                "AITunnel response is not safely JSON-serializable"
            ) from error
        return DecisionJudge._redact_sensitive_fields(detached, secret=secret)

    @staticmethod
    def _redact_sensitive_fields(value: JsonValue, *, secret: str) -> JsonValue:
        if isinstance(value, dict):
            sanitized: dict[str, JsonValue] = {}
            for key, nested in value.items():
                normalized = key.lower().replace("-", "").replace("_", "")
                sanitized[key] = (
                    "[REDACTED]"
                    if normalized in _SENSITIVE_KEYS
                    else DecisionJudge._redact_sensitive_fields(nested, secret=secret)
                )
            return sanitized
        if isinstance(value, list):
            return [
                DecisionJudge._redact_sensitive_fields(item, secret=secret)
                for item in value
            ]
        if isinstance(value, str) and secret:
            return value.replace(secret, "[REDACTED]")
        return value


__all__ = ["DecisionJudge"]
