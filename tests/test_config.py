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

    assert config.models.llm.alias == "gigachat"
    assert config.models.llm.verify_ssl_certs is False
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


def test_llm_tls_verification_setting_is_required(tmp_path: Path):
    data = _default_data()
    del data["models"]["llm"]["verify_ssl_certs"]

    with pytest.raises(ConfigError, match=r"models\.llm\.verify_ssl_certs"):
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
