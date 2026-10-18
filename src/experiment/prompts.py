"""Strict loading of the shared substantive collegium prompts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Iterator, Mapping

from .config import PromptPaths
from .schemas import Collegium


COLLEGIA: tuple[Collegium, ...] = ("hiring", "negotiation", "ownership")
PROMPT_FILENAMES: Mapping[Collegium, str] = MappingProxyType(
    {
        "hiring": "hiring.txt",
        "negotiation": "negotiation.txt",
        "ownership": "ownership.txt",
    }
)


class PromptLoadError(ValueError):
    """The common prompt set is incomplete, ambiguous, or unreadable."""


@dataclass(frozen=True, slots=True)
class Prompt:
    """One substantive prompt and its source path."""

    collegium: Collegium
    text: str
    path: Path


@dataclass(frozen=True, slots=True)
class PromptSet(Mapping[Collegium, Prompt]):
    """The complete immutable set of shared substantive prompts."""

    hiring: Prompt
    negotiation: Prompt
    ownership: Prompt

    def __getitem__(self, collegium: Collegium) -> Prompt:
        return self.for_collegium(collegium)

    def __iter__(self) -> Iterator[Collegium]:
        return iter(COLLEGIA)

    def __len__(self) -> int:
        return len(COLLEGIA)

    def for_collegium(self, collegium: Collegium) -> Prompt:
        return getattr(self, collegium)

    def metadata_paths(self) -> dict[Collegium, str]:
        """Return JSON-ready source paths for future run metadata."""

        return {
            collegium: str(self.for_collegium(collegium).path)
            for collegium in COLLEGIA
        }


def _configured_paths(paths: PromptPaths) -> dict[Collegium, Path]:
    return {collegium: getattr(paths, collegium).resolve() for collegium in COLLEGIA}


def load_prompts(paths: PromptPaths) -> PromptSet:
    """Load exactly the three canonical prompts from one common directory."""

    configured = _configured_paths(paths)
    directories = {path.parent for path in configured.values()}
    if len(directories) != 1:
        raise PromptLoadError(
            "Common prompts must be stored together in one directory; got: "
            + ", ".join(sorted(str(path) for path in directories))
        )

    directory = next(iter(directories))
    expected_paths = {
        collegium: directory / PROMPT_FILENAMES[collegium] for collegium in COLLEGIA
    }
    misconfigured = [
        collegium
        for collegium in COLLEGIA
        if configured[collegium] != expected_paths[collegium]
    ]
    if misconfigured:
        details = ", ".join(
            f"{name}={configured[name].name!r} (expected {PROMPT_FILENAMES[name]!r})"
            for name in misconfigured
        )
        raise PromptLoadError(f"Non-canonical common prompt paths: {details}")

    if not directory.is_dir():
        raise PromptLoadError(f"Common prompt directory does not exist: {directory}")

    actual_files = {entry.name for entry in directory.iterdir() if entry.is_file()}
    expected_files = set(PROMPT_FILENAMES.values())
    missing = sorted(expected_files - actual_files)
    extra = sorted(actual_files - expected_files)
    if missing or extra:
        problems = []
        if missing:
            problems.append("missing files: " + ", ".join(missing))
        if extra:
            problems.append("unexpected files: " + ", ".join(extra))
        raise PromptLoadError(
            f"Invalid common prompt set in {directory}: " + "; ".join(problems)
        )

    loaded: dict[Collegium, Prompt] = {}
    for collegium in COLLEGIA:
        path = expected_paths[collegium]
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as error:
            raise PromptLoadError(f"Cannot read common prompt {path}: {error}") from error
        if not text:
            raise PromptLoadError(f"Common prompt is empty: {path}")
        loaded[collegium] = Prompt(collegium=collegium, text=text, path=path)

    return PromptSet(**loaded)
