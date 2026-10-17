"""Shared experiment infrastructure."""

from .data_loader import (
    DatasetLoadError,
    PreparedBoutVariants,
    load_dataset,
    prepare_bout,
    prepared_bout_payload,
    serialize_prepared_bout,
    swap_players,
)
from .prompts import COLLEGIA, Prompt, PromptLoadError, PromptSet, load_prompts
from .runner import PlannedTask, TaskPlanError, build_task_plan
from .schemas import (
    CallRecord,
    Collegium,
    JudgeResponse,
    Player,
    PreparedBout,
    TaskKey,
    TechnicalAttempt,
    TranscriptSchema,
    Variant,
    load_transcript,
)

__all__ = [
    "CallRecord",
    "COLLEGIA",
    "Collegium",
    "DatasetLoadError",
    "JudgeResponse",
    "Player",
    "PlannedTask",
    "Prompt",
    "PromptLoadError",
    "PromptSet",
    "PreparedBout",
    "PreparedBoutVariants",
    "TaskKey",
    "TaskPlanError",
    "TechnicalAttempt",
    "TranscriptSchema",
    "Variant",
    "build_task_plan",
    "load_dataset",
    "load_prompts",
    "load_transcript",
    "prepare_bout",
    "prepared_bout_payload",
    "serialize_prepared_bout",
    "swap_players",
]
