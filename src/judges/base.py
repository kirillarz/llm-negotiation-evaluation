"""Provider-independent contract for one AI-judge vote."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
import os
from typing import ClassVar, final

from pydantic import JsonValue

from src.experiment.config import DecisionConfig, LLMConfig
from src.experiment.prompts import Prompt, PromptSet
from src.experiment.schemas import Collegium, JudgeResponse, PreparedBout

from .errors import JudgeConfigurationError, JudgeResponseFormatError


JudgeConfig = LLMConfig | DecisionConfig


class BaseJudge(ABC):
    """Common, stateless-per-call adapter boundary.

    Subclasses implement ``_evaluate_once`` with exactly one provider request. The
    public ``evaluate`` method deliberately contains no retry, batching, variant,
    collegium-loop, persistence, or conversation-history logic. Those concerns
    belong to the runner.
    """

    SUPPORTED_MODEL_IDS: ClassVar[frozenset[str]] = frozenset()
    REQUIRED_PARAMETERS: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        config: JudgeConfig,
        prompts: PromptSet,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        # A deep snapshot prevents later mutation of the source configuration from
        # making nominally identical repeats use different effective parameters.
        self._config = config.model_copy(deep=True)
        self._prompts = prompts
        self._environ = os.environ if environ is None else environ

    @property
    def model_alias(self) -> str:
        return self._config.alias

    @property
    def provider(self) -> str:
        return self._config.provider

    @property
    def api_model_id(self) -> str:
        return self._config.api_model_id

    @property
    def effective_parameters(self) -> dict[str, JsonValue]:
        """Return a defensive copy of the adapter's fixed request parameters."""

        return deepcopy(self._config.parameters)

    def prompt_for(self, collegium: Collegium) -> Prompt:
        """Return the one shared substantive prompt selected for this vote."""

        return self._prompts.for_collegium(collegium)

    def _secret(self) -> str:
        """Read the configured credential without exposing it through public state."""

        value = self._environ.get(self._config.secret_env)
        if value is None or not value.strip():
            raise JudgeConfigurationError(
                f"Environment variable {self._config.secret_env} is missing or empty "
                f"for adapter {self.model_alias!r}"
            )
        return value

    @final
    def preflight(self) -> None:
        """Validate all locally checkable adapter settings without network access."""

        if not self.SUPPORTED_MODEL_IDS:
            raise JudgeConfigurationError(
                f"Adapter {type(self).__name__} declares no supported model IDs"
            )
        if self.api_model_id not in self.SUPPORTED_MODEL_IDS:
            supported = ", ".join(sorted(self.SUPPORTED_MODEL_IDS))
            raise JudgeConfigurationError(
                f"Unsupported model ID {self.api_model_id!r} for adapter "
                f"{self.model_alias!r}; supported IDs: {supported}"
            )

        missing_parameters = sorted(
            name
            for name in self.REQUIRED_PARAMETERS
            if name not in self._config.parameters
            or self._config.parameters[name] is None
        )
        if missing_parameters:
            raise JudgeConfigurationError(
                f"Missing required parameters for adapter {self.model_alias!r}: "
                + ", ".join(missing_parameters)
            )

        self._secret()
        self._validate_provider_configuration()

    def _validate_provider_configuration(self) -> None:
        """Hook for provider-specific local validation beyond the shared schema."""

    @final
    def evaluate(
        self,
        bout: PreparedBout,
        collegium: Collegium,
    ) -> JudgeResponse:
        """Perform one substantive vote with no internal retry or saved history."""

        response = self._evaluate_once(bout=bout, collegium=collegium)
        if not isinstance(response, JudgeResponse):
            raise JudgeResponseFormatError(
                f"Adapter {self.model_alias!r} returned "
                f"{type(response).__name__}, expected JudgeResponse"
            )
        return response

    @abstractmethod
    def _evaluate_once(
        self,
        bout: PreparedBout,
        collegium: Collegium,
    ) -> JudgeResponse:
        """Make exactly one provider request and normalize its single vote."""


__all__ = ["BaseJudge", "JudgeConfig"]
