"""Command-line entry point for offline calculation of scientific metrics."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import sys
from typing import Annotated, Literal, TextIO

from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

from src.experiment.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    ExperimentConfig,
    load_config,
)


EXIT_SUCCESS = 0
EXIT_USAGE_ERROR = 2
SMOKE_UNSUPPORTED_MESSAGE = "Smoke-запуски не поддерживают расчёт научных метрик"

Dataset = Literal["smoke", "pilot", "main"]
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ConfigLoader = Callable[[Path | str], ExperimentConfig]


class AnalysisRequest(BaseModel):
    """Validated selection passed from the CLI to the offline pipeline."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    dataset: Dataset
    run_id: NonEmptyString
    config_path: Path


Analyzer = Callable[[AnalysisRequest, ExperimentConfig], None]


class CLIError(ValueError):
    """Arguments or local inputs prevent metrics calculation."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="python -m src.calculate_metrics")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("smoke", "pilot", "main"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def run(
    argv: Sequence[str] | None = None,
    *,
    analyzer: Analyzer | None = None,
    config_loader: ConfigLoader = load_config,
    output: TextIO = sys.stdout,
    error_output: TextIO = sys.stderr,
) -> int:
    """Validate a calculation request and return a stable process exit code."""

    try:
        args = build_parser().parse_args(argv)
        if args.dataset == "smoke":
            raise CLIError(SMOKE_UNSUPPORTED_MESSAGE)

        config = config_loader(args.config)
        request = AnalysisRequest(
            dataset=args.dataset,
            run_id=args.run_id,
            config_path=args.config.resolve(),
        )
        if analyzer is not None:
            analyzer(request, config)
        else:
            print(
                "Запрос принят; чтение запуска и расчёт метрик будут добавлены "
                "на следующих этапах.",
                file=output,
            )
        return EXIT_SUCCESS
    except (CLIError, ConfigError, ValidationError) as error:
        print(f"Ошибка расчёта метрик: {error}", file=error_output)
        return EXIT_USAGE_ERROR


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
