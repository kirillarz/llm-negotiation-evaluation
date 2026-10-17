"""Single command-line entry point for collecting AI-judge votes."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
import sys
from typing import TextIO, cast

from dotenv import load_dotenv
from pydantic import ValidationError

from src.experiment.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    ExperimentConfig,
    load_config,
)
from src.experiment.data_loader import DatasetLoadError, PreparedBoutVariants, load_dataset
from src.experiment.prompts import PromptLoadError, PromptSet, load_prompts
from src.experiment.runner import (
    PlannedTask,
    TaskPlanError,
    build_task_plan,
    execute_task_plan,
)
from src.experiment.schemas import Collegium
from src.experiment.storage import (
    RunMetadata,
    RunStorage,
    StorageError,
    build_run_spec,
    generate_run_id,
)
from src.judges.base import BaseJudge, JudgeConfig
from src.judges.errors import JudgeConfigurationError


EXIT_SUCCESS = 0
EXIT_PARTIAL = 1
EXIT_PREFLIGHT_ERROR = 2
EXIT_INTERRUPTED = 130

ModelSelection = str
JudgeFactory = Callable[[JudgeConfig, PromptSet, Mapping[str, str]], BaseJudge]


class CLIError(ValueError):
    """Arguments or local orchestration state make a run unsafe."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(prog="python -m src.run_experiment")
    parser.add_argument("--model", choices=("llm", "decision", "all"))
    parser.add_argument("--dataset", choices=("pilot", "main"))
    parser.add_argument("--bout-id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--collegium", choices=("hiring", "negotiation", "ownership"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-main", action="store_true")
    parser.add_argument("--resume", metavar="RUN_ID")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.resume:
        forbidden = [
            name
            for name in ("model", "dataset", "bout_id")
            if getattr(args, name) is not None
        ]
        forbidden.extend(
            name
            for name in ("smoke", "collegium", "dry_run", "confirm_main")
            if getattr(args, name)
        )
        if forbidden:
            flags = ", ".join("--" + name.replace("_", "-") for name in forbidden)
            raise CLIError(f"--resume cannot be combined with {flags}")
        return

    if args.model is None or args.dataset is None:
        raise CLIError("new runs require both --model and --dataset")
    if args.retry_failed:
        raise CLIError("--retry-failed requires --resume")
    if args.smoke and (args.bout_id is None or args.collegium is None):
        raise CLIError("--smoke requires --bout-id and --collegium")
    if not args.smoke and args.collegium is not None:
        raise CLIError("--collegium can only be used with --smoke")
    if args.confirm_main and args.dataset != "main":
        raise CLIError("--confirm-main can only be used with --dataset main")


def _default_factories() -> dict[str, JudgeFactory]:
    """Load provider adapters lazily; stages 7 and 8 supply these modules."""

    try:
        from src.judges.decision.judge import DecisionJudge
        from src.judges.llm.judge import LLMJudge
    except ImportError as error:
        raise CLIError(
            "provider adapters are not implemented yet; inject test adapters for "
            "stage 6 or complete stages 7 and 8"
        ) from error
    return {
        "llm": lambda config, prompts, environ: LLMJudge(
            config, prompts, environ=environ
        ),
        "decision": lambda config, prompts, environ: DecisionJudge(
            config, prompts, environ=environ
        ),
    }


def _model_keys(selection: ModelSelection) -> tuple[str, ...]:
    if selection == "all":
        return ("llm", "decision")
    if selection in ("llm", "decision"):
        return (selection,)
    raise CLIError(f"unsupported model selection: {selection!r}")


def _make_judges(
    config: ExperimentConfig,
    prompts: PromptSet,
    model_keys: Sequence[str],
    factories: Mapping[str, JudgeFactory] | None,
    environ: Mapping[str, str],
) -> tuple[BaseJudge, ...]:
    selected_factories = _default_factories() if factories is None else factories
    judges: list[BaseJudge] = []
    for key in model_keys:
        factory = selected_factories.get(key)
        if factory is None:
            raise CLIError(f"no adapter factory is registered for model {key!r}")
        model_config = getattr(config.models, key)
        judges.append(factory(model_config, prompts, environ))

    # Atomic preflight: construct every selected adapter before checking any, then
    # check all of them before task execution or storage creation.
    preflight_errors: list[str] = []
    for judge in judges:
        try:
            judge.preflight()
        except JudgeConfigurationError as error:
            preflight_errors.append(f"{judge.model_alias}: {error}")
    if preflight_errors:
        raise JudgeConfigurationError(
            "Selected adapter preflight failed: " + "; ".join(preflight_errors)
        )
    return tuple(judges)


def _select_bouts(
    bouts: tuple[PreparedBoutVariants, ...], selected_ids: Sequence[str]
) -> tuple[PreparedBoutVariants, ...]:
    by_id = {bout.bout_id: bout for bout in bouts}
    missing = [bout_id for bout_id in selected_ids if bout_id not in by_id]
    if missing:
        raise DatasetLoadError(
            "Persisted bout IDs are absent from the current dataset: "
            + ", ".join(missing)
        )
    return tuple(by_id[bout_id] for bout_id in selected_ids)


def _bind_persisted_plan(
    metadata: RunMetadata,
    bouts: tuple[PreparedBoutVariants, ...],
    judges: tuple[BaseJudge, ...],
) -> tuple[PlannedTask, ...]:
    bouts_by_id = {bout.bout_id: bout for bout in bouts}
    judges_by_alias = {judge.model_alias: judge for judge in judges}
    tasks = []
    for identity in metadata.planned_tasks:
        try:
            bout = bouts_by_id[identity.bout_id].get(identity.variant)
            judge = judges_by_alias[identity.model_alias]
        except KeyError as error:
            raise TaskPlanError(
                "persisted task plan cannot be bound to current inputs"
            ) from error
        tasks.append(
            PlannedTask(
                key=identity.to_key(metadata.run_id),
                bout=bout,
                judge=judge,
            )
        )
    return tuple(tasks)


def _find_resume_root(config: ExperimentConfig, run_id: str) -> Path:
    roots = (config.results.smoke, config.results.pilot, config.results.main)
    matches = [root for root in roots if (root / run_id).is_dir()]
    if not matches:
        raise StorageError(f"run directory does not exist for run_id {run_id!r}")
    if len(matches) > 1:
        raise StorageError(f"run_id {run_id!r} exists in multiple result roots")
    return matches[0]


def _read_metadata(root: Path, run_id: str) -> RunMetadata:
    path = root / run_id / "metadata.json"
    try:
        return RunMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValidationError) as error:
        raise StorageError(f"cannot read valid run metadata from {path}: {error}") from error


def _keys_for_metadata(
    config: ExperimentConfig, metadata: RunMetadata
) -> tuple[str, ...]:
    aliases = {model.alias for model in metadata.models}
    configured = {
        "llm": config.models.llm.alias,
        "decision": config.models.decision.alias,
    }
    keys = tuple(key for key in ("llm", "decision") if configured[key] in aliases)
    if {configured[key] for key in keys} != aliases:
        raise CLIError("persisted model aliases do not match the current configuration")
    return keys


def _dry_run_report(
    *,
    dataset: str,
    bouts: tuple[PreparedBoutVariants, ...],
    tasks: tuple[PlannedTask, ...],
    results_root: Path,
) -> str:
    counts = Counter(
        (task.key.model_alias, task.key.variant, task.key.collegium) for task in tasks
    )
    lines = [
        "Dry run passed; no run_id was created and no result files were written.",
        f"Dataset: {dataset}",
        "Selected bout IDs: " + ", ".join(bout.bout_id for bout in bouts),
        f"Total planned calls: {len(tasks)}",
        "Planned calls by model / variant / collegium:",
    ]
    lines.extend(
        f"  {model} / {variant} / {collegium}: {count}"
        for (model, variant, collegium), count in sorted(counts.items())
    )
    expected = results_root / "<run_id>"
    lines.extend(
        [
            "Expected result paths:",
            f"  {expected / 'metadata.json'}",
            f"  {expected / 'calls.jsonl'}",
            f"  {expected / 'summary.json'}",
        ]
    )
    return "\n".join(lines)


def _run_new(
    args: argparse.Namespace,
    config: ExperimentConfig,
    factories: Mapping[str, JudgeFactory] | None,
    environ: Mapping[str, str],
    output: TextIO,
) -> int:
    # Keep this order in sync with the technical regulations: validate every
    # selected JSON before loading prompts or checking adapters.
    bouts = load_dataset(config, args.dataset, bout_id=args.bout_id)
    prompts = load_prompts(config.prompts)
    judges = _make_judges(
        config, prompts, _model_keys(args.model), factories, environ
    )
    preflight_plan = build_task_plan(
        run_id="dry-run-placeholder",
        bouts=bouts,
        judges=judges,
        smoke=args.smoke,
        bout_id=args.bout_id,
        collegium=cast(Collegium | None, args.collegium),
    )
    mode = "smoke" if args.smoke else args.dataset
    results_root = getattr(config.results, mode)

    if args.dry_run:
        print(
            _dry_run_report(
                dataset=args.dataset,
                bouts=bouts,
                tasks=preflight_plan,
                results_root=results_root,
            ),
            file=output,
        )
        return EXIT_SUCCESS
    if args.dataset == "main" and not args.confirm_main:
        raise CLIError(
            "real runs on the main dataset require --confirm-main; this technical "
            "flag does not replace organizational approval"
        )

    run_id = generate_run_id()
    tasks = build_task_plan(
        run_id=run_id,
        bouts=bouts,
        judges=judges,
        smoke=args.smoke,
        bout_id=args.bout_id,
        collegium=cast(Collegium | None, args.collegium),
    )
    spec = build_run_spec(
        mode=mode,
        dataset=args.dataset,
        tasks=tasks,
        prompt_paths=prompts.metadata_paths(),
    )
    storage = RunStorage.create(results_root, run_id=run_id, spec=spec)
    print(f"Run ID: {run_id}\nResults: {storage.run_dir}", file=output)
    try:
        result = execute_task_plan(
            tasks,
            storage,
            max_attempts=config.execution.max_attempts,
        )
    except KeyboardInterrupt:
        # Every completed task is already durable; the interrupted task remains pending.
        summary = storage.summary()
        print(
            f"Interrupted: success={summary.success}, failed={summary.failed}, "
            f"pending={summary.pending}",
            file=output,
        )
        return EXIT_INTERRUPTED
    return _report_execution(result, output)


def _run_resume(
    args: argparse.Namespace,
    config: ExperimentConfig,
    factories: Mapping[str, JudgeFactory] | None,
    environ: Mapping[str, str],
    output: TextIO,
) -> int:
    root = _find_resume_root(config, args.resume)
    metadata = _read_metadata(root, args.resume)
    expected_root = getattr(config.results, metadata.mode)
    if root.resolve() != expected_root.resolve():
        raise StorageError(
            f"run metadata mode {metadata.mode!r} does not match result root {root}"
        )
    # Dataset, models, selection and mode come exclusively from immutable metadata.
    all_bouts = load_dataset(config, metadata.dataset)
    bouts = _select_bouts(all_bouts, metadata.selected_bout_ids)
    prompts = load_prompts(config.prompts)
    judges = _make_judges(
        config,
        prompts,
        _keys_for_metadata(config, metadata),
        factories,
        environ,
    )
    tasks = _bind_persisted_plan(metadata, bouts, judges)
    versions = {model.alias: model.model_version for model in metadata.models}
    expected = build_run_spec(
        mode=metadata.mode,
        dataset=metadata.dataset,
        tasks=tasks,
        prompt_paths=prompts.metadata_paths(),
        model_versions=versions,
    )
    storage = RunStorage.resume(root, args.resume, expected=expected)
    print(f"Resuming run: {args.resume}\nResults: {storage.run_dir}", file=output)
    try:
        result = execute_task_plan(
            tasks,
            storage,
            max_attempts=config.execution.max_attempts,
            retry_failed=args.retry_failed,
            model_versions=versions,
        )
    except KeyboardInterrupt:
        summary = storage.summary()
        print(
            f"Interrupted: success={summary.success}, failed={summary.failed}, "
            f"pending={summary.pending}",
            file=output,
        )
        return EXIT_INTERRUPTED
    return _report_execution(result, output)


def _report_execution(result, output: TextIO) -> int:
    summary = result.summary
    print(
        f"Finished: success={summary.success}, failed={summary.failed}, "
        f"pending={summary.pending}",
        file=output,
    )
    if result.aborted:
        print(
            f"Run stopped by configuration error: {result.abort_error.message}",
            file=output,
        )
        return EXIT_PREFLIGHT_ERROR
    if summary.failed or summary.pending:
        return EXIT_PARTIAL
    return EXIT_SUCCESS


def run(
    argv: Sequence[str] | None = None,
    *,
    judge_factories: Mapping[str, JudgeFactory] | None = None,
    environ: Mapping[str, str] | None = None,
    output: TextIO = sys.stdout,
    error_output: TextIO = sys.stderr,
) -> int:
    """Run the CLI and return a stable process exit code."""

    try:
        args = build_parser().parse_args(argv)
        _validate_args(args)
        config = load_config(args.config)
        if environ is None:
            load_dotenv()
            selected_environ: Mapping[str, str] = os.environ
        else:
            selected_environ = environ
        if args.resume:
            return _run_resume(
                args, config, judge_factories, selected_environ, output
            )
        return _run_new(args, config, judge_factories, selected_environ, output)
    except (
        CLIError,
        ConfigError,
        DatasetLoadError,
        PromptLoadError,
        JudgeConfigurationError,
        TaskPlanError,
        StorageError,
    ) as exc:
        print(f"Preflight error: {exc}", file=error_output)
        return EXIT_PREFLIGHT_ERROR


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
