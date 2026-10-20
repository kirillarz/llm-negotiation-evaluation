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
from src.metrics.loading import AnalysisInputError, load_analysis_inputs
from src.metrics.aggregation import aggregate_analysis_data
from src.metrics.bootstrap import calculate_bootstrap
from src.metrics.calculations import calculate_metrics
from src.metrics.inference import calculate_inference
from src.metrics.preparation import prepare_analysis_data
from src.metrics.reporting import ReportingError, console_summary, write_reports


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
    config_loader: ConfigLoader | None = None,
    output: TextIO = sys.stdout,
    error_output: TextIO = sys.stderr,
) -> int:
    """Validate a calculation request and return a stable process exit code."""

    try:
        args = build_parser().parse_args(argv)
        if args.dataset == "smoke":
            raise CLIError(SMOKE_UNSUPPORTED_MESSAGE)

        config = (
            load_config(args.config, validate_input_paths=False)
            if config_loader is None
            else config_loader(args.config)
        )
        request = AnalysisRequest(
            dataset=args.dataset,
            run_id=args.run_id,
            config_path=args.config.resolve(),
        )
        if analyzer is not None:
            analyzer(request, config)
        else:
            inputs = load_analysis_inputs(config, request.dataset, request.run_id)
            prepared = prepare_analysis_data(inputs)
            aggregated = aggregate_analysis_data(prepared)
            calculated = calculate_metrics(prepared, aggregated)
            bootstrap = calculate_bootstrap(prepared, aggregated, config)
            inference = calculate_inference(prepared, aggregated, config)
            output_dir = write_reports(
                inputs, prepared, aggregated, calculated, bootstrap, inference
            )
            print(
                console_summary(inputs, prepared, calculated, output_dir),
                file=output,
            )
        return EXIT_SUCCESS
    except (
        CLIError,
        ConfigError,
        AnalysisInputError,
        ReportingError,
        ValidationError,
    ) as error:
        print(f"Ошибка расчёта метрик: {error}", file=error_output)
        return EXIT_USAGE_ERROR


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
