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
    "Prompt",
    "PromptLoadError",
    "PromptSet",
    "PreparedBout",
    "PreparedBoutVariants",
    "TaskKey",
    "TechnicalAttempt",
    "TranscriptSchema",
    "Variant",
    "load_dataset",
    "load_prompts",
    "load_transcript",
    "prepare_bout",
    "prepared_bout_payload",
    "serialize_prepared_bout",
    "swap_players",
]
