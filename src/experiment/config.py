"""Strict loading and validation for the experiment YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    HttpUrl,
    JsonValue,
    StringConstraints,
    ValidationError,
    model_validator,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[a-z][a-z0-9_-]*$",
    ),
]
EnvironmentVariable = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Z][A-Z0-9_]*$",
    ),
]
PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]


def _parse_path(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value.strip())
    raise ValueError("path must be a non-empty string")


ConfiguredPath = Annotated[Path, BeforeValidator(_parse_path)]


class ConfigError(ValueError):
    """Configuration cannot be read or does not satisfy its contract."""


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DatasetPaths(StrictConfigModel):
    pilot: ConfiguredPath
    main: ConfiguredPath


class ResultPaths(StrictConfigModel):
    smoke: ConfiguredPath
    pilot: ConfiguredPath
    main: ConfiguredPath


class PromptPaths(StrictConfigModel):
    hiring: ConfiguredPath
    negotiation: ConfiguredPath
    ownership: ConfiguredPath


class LLMConfig(StrictConfigModel):
    alias: Identifier
    provider: Identifier
    api_model_id: NonEmptyString
    secret_env: EnvironmentVariable
    timeout_seconds: PositiveStrictInt
    verify_ssl_certs: bool
    parameters: dict[str, JsonValue]


class DecisionConfig(StrictConfigModel):
    alias: Identifier
    provider: Identifier
    api_model_id: NonEmptyString
    api_url: HttpUrl
    secret_env: EnvironmentVariable
    timeout_seconds: PositiveStrictInt
    parameters: dict[str, JsonValue]


class ModelConfigs(StrictConfigModel):
    llm: LLMConfig
    decision: DecisionConfig

    @model_validator(mode="after")
    def check_unique_aliases(self) -> Self:
        if self.llm.alias == self.decision.alias:
            raise ValueError("models.llm.alias and models.decision.alias must differ")
        return self


class ExecutionConfig(StrictConfigModel):
    max_attempts: Literal[2]


class ExperimentConfig(StrictConfigModel):
    datasets: DatasetPaths
    results: ResultPaths
    prompts: PromptPaths
    models: ModelConfigs
    execution: ExecutionConfig


def _resolve_path(path: Path, config_dir: Path) -> Path:
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _with_resolved_paths(config: ExperimentConfig, config_dir: Path) -> ExperimentConfig:
    datasets = config.datasets.model_copy(
        update={
            name: _resolve_path(getattr(config.datasets, name), config_dir)
            for name in ("pilot", "main")
        }
    )
    results = config.results.model_copy(
        update={
            name: _resolve_path(getattr(config.results, name), config_dir)
            for name in ("smoke", "pilot", "main")
        }
    )
    prompts = config.prompts.model_copy(
        update={
            name: _resolve_path(getattr(config.prompts, name), config_dir)
            for name in ("hiring", "negotiation", "ownership")
        }
    )
    return config.model_copy(
        update={"datasets": datasets, "results": results, "prompts": prompts}
    )


def _format_validation_error(error: ValidationError) -> str:
    messages = []
    for item in error.errors(include_url=False, include_context=False):
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        messages.append(f"{location}: {item['msg']}")
    return "; ".join(messages)


def _validate_input_paths(config: ExperimentConfig) -> None:
    problems: list[str] = []
    for name in ("pilot", "main"):
        path = getattr(config.datasets, name)
        if not path.is_dir():
            problems.append(f"datasets.{name}: directory does not exist: {path}")
    for name in ("hiring", "negotiation", "ownership"):
        path = getattr(config.prompts, name)
        if not path.is_file():
            problems.append(f"prompts.{name}: file does not exist: {path}")
    if problems:
        raise ConfigError("Invalid experiment configuration: " + "; ".join(problems))


def load_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
    *,
    validate_input_paths: bool = True,
) -> ExperimentConfig:
    """Load and structurally validate YAML, optionally checking all runner inputs.

    The default remains deliberately strict for experiment execution.  Offline
    consumers may disable the broad path check and validate only the paths they
    actually use.
    """

    config_path = Path(path).expanduser().resolve()
    try:
        source = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"Cannot read configuration file {config_path}: {error}") from error

    try:
        raw = yaml.safe_load(source)
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in configuration file {config_path}: {error}") from error

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Invalid experiment configuration {config_path}: root must be a mapping"
        )

    try:
        config = ExperimentConfig.model_validate(raw)
    except ValidationError as error:
        details = _format_validation_error(error)
        raise ConfigError(
            f"Invalid experiment configuration {config_path}: {details}"
        ) from error

    config = _with_resolved_paths(config, config_path.parent)
    if validate_input_paths:
        _validate_input_paths(config)
    return config
