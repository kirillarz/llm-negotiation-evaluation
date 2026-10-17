"""Deterministic planning primitives for the experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import JsonValue

from .data_loader import PreparedBoutVariants
from .prompts import COLLEGIA, Prompt
from .schemas import Collegium, PreparedBout, TaskKey, Variant

if TYPE_CHECKING:
    from src.judges.base import BaseJudge


VARIANTS: tuple[Variant, ...] = ("original", "swapped")
REPEAT_INDICES: tuple[int, ...] = (1, 2, 3)


class TaskPlanError(ValueError):
    """A requested experiment plan is empty, ambiguous, or inconsistent."""


@dataclass(frozen=True, slots=True)
class PlannedTask:
    """One planned vote bound to immutable input and one configured adapter."""

    key: TaskKey
    bout: PreparedBout
    judge: BaseJudge

    @property
    def prompt(self) -> Prompt:
        """Resolve the shared prompt solely from the stable collegium key."""

        return self.judge.prompt_for(self.key.collegium)

    @property
    def effective_parameters(self) -> dict[str, JsonValue]:
        """Return the adapter's fixed, secret-free parameter snapshot."""

        return self.judge.effective_parameters


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not run_id.strip():
        raise TaskPlanError("run_id must be a non-empty string")
    return run_id.strip()


def _validate_bouts(
    bouts: tuple[PreparedBoutVariants, ...],
) -> tuple[PreparedBoutVariants, ...]:
    if not bouts:
        raise TaskPlanError("Cannot build a task plan without prepared bouts")
    ids = [bout.bout_id for bout in bouts]
    duplicates = sorted({bout_id for bout_id in ids if ids.count(bout_id) > 1})
    if duplicates:
        raise TaskPlanError(
            "Prepared bout IDs must be unique; duplicates: " + ", ".join(duplicates)
        )
    return bouts


def _validate_judges(judges: tuple[BaseJudge, ...]) -> tuple[BaseJudge, ...]:
    if not judges:
        raise TaskPlanError("Cannot build a task plan without judge adapters")
    aliases = [judge.model_alias for judge in judges]
    duplicates = sorted({alias for alias in aliases if aliases.count(alias) > 1})
    if duplicates:
        raise TaskPlanError(
            "Judge model aliases must be unique; duplicates: " + ", ".join(duplicates)
        )
    return judges


def _select_bouts(
    bouts: tuple[PreparedBoutVariants, ...],
    bout_id: str | None,
) -> tuple[PreparedBoutVariants, ...]:
    if bout_id is None:
        return bouts
    if not isinstance(bout_id, str) or not bout_id.strip():
        raise TaskPlanError("bout_id must be a non-empty string when provided")
    requested = bout_id.strip()
    matches = tuple(bout for bout in bouts if bout.bout_id == requested)
    if not matches:
        available = ", ".join(bout.bout_id for bout in bouts)
        raise TaskPlanError(
            f"Bout ID {requested!r} is not available for planning; "
            f"available IDs: {available}"
        )
    return matches


def _smoke_dimensions(
    *,
    smoke: bool,
    bout_id: str | None,
    collegium: Collegium | None,
) -> tuple[tuple[Variant, ...], tuple[Collegium, ...], tuple[int, ...]]:
    if smoke:
        if bout_id is None or not isinstance(bout_id, str) or not bout_id.strip():
            raise TaskPlanError("Smoke mode requires an explicit bout_id")
        if collegium is None:
            raise TaskPlanError("Smoke mode requires an explicit collegium")
        if collegium not in COLLEGIA:
            raise TaskPlanError(
                f"Unknown collegium {collegium!r}; expected one of: "
                + ", ".join(COLLEGIA)
            )
        return (("original",), (collegium,), (1,))

    if collegium is not None:
        raise TaskPlanError("collegium can only restrict a smoke plan")
    return (VARIANTS, COLLEGIA, REPEAT_INDICES)


def build_task_plan(
    *,
    run_id: str,
    bouts: tuple[PreparedBoutVariants, ...],
    judges: tuple[BaseJudge, ...],
    smoke: bool = False,
    bout_id: str | None = None,
    collegium: Collegium | None = None,
) -> tuple[PlannedTask, ...]:
    """Build tasks in stable bout/model/variant/collegium/repeat order.

    The caller supplies ordered tuples produced by dataset and model selection.
    Repeats bind to the same frozen ``PreparedBout`` and adapter instance, so their
    prompt and effective parameters are identical and no response/history is part
    of a planned task.
    """

    validated_run_id = _validate_run_id(run_id)
    validated_bouts = _validate_bouts(bouts)
    validated_judges = _validate_judges(judges)
    variants, collegia, repeat_indices = _smoke_dimensions(
        smoke=smoke,
        bout_id=bout_id,
        collegium=collegium,
    )
    selected_bouts = _select_bouts(validated_bouts, bout_id)

    tasks: list[PlannedTask] = []
    for bout_variants in selected_bouts:
        for judge in validated_judges:
            for variant in variants:
                prepared_bout = bout_variants.get(variant)
                for selected_collegium in collegia:
                    for repeat_index in repeat_indices:
                        key = TaskKey(
                            run_id=validated_run_id,
                            model_alias=judge.model_alias,
                            bout_id=bout_variants.bout_id,
                            variant=variant,
                            collegium=selected_collegium,
                            repeat_index=repeat_index,
                        )
                        tasks.append(
                            PlannedTask(key=key, bout=prepared_bout, judge=judge)
                        )
    return tuple(tasks)


__all__ = [
    "PlannedTask",
    "REPEAT_INDICES",
    "TaskPlanError",
    "VARIANTS",
    "build_task_plan",
]
