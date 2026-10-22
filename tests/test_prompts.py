from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment.config import PromptPaths, load_config
from src.experiment.prompts import COLLEGIA, PromptLoadError, load_prompts


PROMPT_NAMES = {
    "hiring": "hiring.txt",
    "negotiation": "negotiation.txt",
    "ownership": "ownership.txt",
}


def _write_prompt_set(directory: Path) -> PromptPaths:
    directory.mkdir(parents=True)
    for collegium, filename in PROMPT_NAMES.items():
        (directory / filename).write_text(
            f"Shared criteria for {collegium}", encoding="utf-8"
        )
    return PromptPaths(
        **{
            collegium: directory / filename
            for collegium, filename in PROMPT_NAMES.items()
        }
    )


def test_default_prompt_set_is_complete_and_exposes_metadata_paths():
    prompts = load_prompts(load_config().prompts)

    assert tuple(prompt.collegium for prompt in (
        prompts.hiring,
        prompts.negotiation,
        prompts.ownership,
    )) == COLLEGIA
    assert "к кому из них" in prompts.hiring.text
    assert "кого из них" in prompts.negotiation.text
    assert "кому из них" in prompts.ownership.text
    assert all(
        "ФИНАЛЬНАЯ ПРОВЕРКА ГРАНИЦ КРИТЕРИЯ" in prompts[collegium].text
        for collegium in COLLEGIA
    )
    assert tuple(prompts) == COLLEGIA
    assert prompts["hiring"] is prompts.hiring
    assert prompts.metadata_paths() == {
        collegium: str(prompts.for_collegium(collegium).path)
        for collegium in COLLEGIA
    }


def test_prompt_loader_rejects_missing_file(tmp_path: Path):
    paths = _write_prompt_set(tmp_path / "common")
    paths.ownership.unlink()

    with pytest.raises(PromptLoadError, match=r"missing files: ownership\.txt"):
        load_prompts(paths)


def test_prompt_loader_rejects_extra_file(tmp_path: Path):
    paths = _write_prompt_set(tmp_path / "common")
    (paths.hiring.parent / "legacy.txt").write_text("legacy", encoding="utf-8")

    with pytest.raises(PromptLoadError, match=r"unexpected files: legacy\.txt"):
        load_prompts(paths)


def test_prompt_loader_rejects_empty_file(tmp_path: Path):
    paths = _write_prompt_set(tmp_path / "common")
    paths.negotiation.write_text(" \n\t", encoding="utf-8")

    with pytest.raises(PromptLoadError, match=r"Common prompt is empty: .*negotiation\.txt"):
        load_prompts(paths)


def test_prompt_loader_requires_one_directory(tmp_path: Path):
    paths = _write_prompt_set(tmp_path / "common")
    other = tmp_path / "other"
    other.mkdir()
    other_ownership = other / "ownership.txt"
    other_ownership.write_text("criteria", encoding="utf-8")
    split_paths = paths.model_copy(update={"ownership": other_ownership})

    with pytest.raises(PromptLoadError, match="one directory"):
        load_prompts(split_paths)


def test_prompt_loader_rejects_noncanonical_filename(tmp_path: Path):
    paths = _write_prompt_set(tmp_path / "common")
    old_name = paths.negotiation.parent / "sending-to-negotiations.txt"
    paths.negotiation.rename(old_name)
    legacy_paths = paths.model_copy(update={"negotiation": old_name})

    with pytest.raises(PromptLoadError, match="Non-canonical common prompt paths"):
        load_prompts(legacy_paths)
