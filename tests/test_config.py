from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.experiment.config import DEFAULT_CONFIG_PATH, ConfigError, load_config


def _write_config(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _default_data() -> dict:
    loaded = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_default_config_is_valid_and_paths_are_absolute():
    config = load_config()

    assert config.models.llm.alias == "gpt6-luna-pro"
    assert str(config.models.llm.api_url) == "https://api.aitunnel.ru/v1/chat/completions"
    assert config.models.decision.alias == "clef"
    assert config.execution.max_attempts == 2
    assert config.datasets.pilot.is_absolute()
    assert config.datasets.pilot.is_dir()
    assert config.prompts.hiring.is_file()


def test_unknown_config_field_has_a_readable_location(tmp_path: Path):
    data = _default_data()
    data["models"]["llm"]["credentials"] = "must-not-be-configured-here"

    with pytest.raises(ConfigError, match=r"models\.llm\.credentials"):
        load_config(_write_config(tmp_path, data))


def test_missing_config_field_has_a_readable_location(tmp_path: Path):
    data = _default_data()
    del data["models"]["decision"]["secret_env"]

    with pytest.raises(ConfigError, match=r"models\.decision\.secret_env"):
        load_config(_write_config(tmp_path, data))


def test_llm_api_url_is_required(tmp_path: Path):
    data = _default_data()
    del data["models"]["llm"]["api_url"]

    with pytest.raises(ConfigError, match=r"models\.llm\.api_url"):
        load_config(_write_config(tmp_path, data))


def test_max_attempts_is_fixed_at_two(tmp_path: Path):
    data = _default_data()
    data["execution"]["max_attempts"] = 3

    with pytest.raises(ConfigError, match=r"execution\.max_attempts"):
        load_config(_write_config(tmp_path, data))


def test_config_rejects_missing_input_path(tmp_path: Path):
    data = _default_data()
    data["datasets"]["pilot"] = "missing-pilot"

    with pytest.raises(ConfigError, match=r"datasets\.pilot: directory does not exist"):
        load_config(_write_config(tmp_path, data))


def test_config_can_skip_broad_input_path_checks_for_offline_consumers(tmp_path: Path):
    data = _default_data()
    data["datasets"]["pilot"] = "missing-pilot"
    data["datasets"]["main"] = "missing-main"
    data["prompts"]["hiring"] = "missing-prompt.txt"

    config = load_config(
        _write_config(tmp_path, data),
        validate_input_paths=False,
    )

    assert config.datasets.pilot == (tmp_path / "missing-pilot").resolve()
    assert config.prompts.hiring == (tmp_path / "missing-prompt.txt").resolve()
