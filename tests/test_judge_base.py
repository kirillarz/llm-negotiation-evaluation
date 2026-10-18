from __future__ import annotations

from typing import cast

import pytest

from src.experiment.config import LLMConfig, load_config
from src.experiment.data_loader import load_dataset
from src.experiment.prompts import load_prompts
from src.experiment.schemas import Collegium, JudgeResponse, PreparedBout
from src.judges.base import BaseJudge
from src.judges.errors import JudgeConfigurationError, JudgeResponseFormatError


class FakeJudge(BaseJudge):
    SUPPORTED_MODEL_IDS = frozenset({"GigaChat-2-Pro"})
    REQUIRED_PARAMETERS = frozenset({"temperature"})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests: list[tuple[PreparedBout, Collegium]] = []

    def _evaluate_once(
        self,
        bout: PreparedBout,
        collegium: Collegium,
    ) -> JudgeResponse:
        self.requests.append((bout, collegium))
        return JudgeResponse(
            winner="player_1",
            probabilities=None,
            raw_response={"winner": "player_1"},
        )


def _judge(
    *,
    environ: dict[str, str] | None = None,
    config: LLMConfig | None = None,
) -> FakeJudge:
    experiment_config = load_config()
    return FakeJudge(
        config or experiment_config.models.llm,
        load_prompts(experiment_config.prompts),
        environ=environ if environ is not None else {"GIGACHAT_CREDENTIALS": "secret"},
    )


def test_preflight_checks_secret_model_and_required_parameters_locally():
    judge = _judge()

    judge.preflight()

    assert judge.model_alias == "gigachat"
    assert judge.api_model_id == "GigaChat-2-Pro"
    assert judge.requests == []


@pytest.mark.parametrize("value", [None, "", "   "])
def test_preflight_rejects_missing_or_empty_secret(value: str | None):
    environ = {} if value is None else {"GIGACHAT_CREDENTIALS": value}

    with pytest.raises(
        JudgeConfigurationError,
        match="GIGACHAT_CREDENTIALS is missing or empty",
    ):
        _judge(environ=environ).preflight()


def test_preflight_rejects_unsupported_model_without_exposing_secret():
    config = load_config().models.llm.model_copy(
        update={"api_model_id": "unsupported-model"}
    )
    secret = "must-never-appear-in-errors"

    with pytest.raises(JudgeConfigurationError) as error:
        _judge(
            config=config,
            environ={"GIGACHAT_CREDENTIALS": secret},
        ).preflight()

    assert "unsupported-model" in str(error.value)
    assert secret not in str(error.value)


def test_preflight_rejects_missing_required_parameter():
    config = load_config().models.llm.model_copy(update={"parameters": {}})

    with pytest.raises(JudgeConfigurationError, match="temperature"):
        _judge(config=config).preflight()


def test_evaluate_delegates_to_exactly_one_adapter_request():
    judge = _judge()
    bout = load_dataset(load_config(), "pilot", bout_id="1")[0].original

    response = judge.evaluate(bout=bout, collegium="hiring")

    assert response.winner == "player_1"
    assert judge.requests == [(bout, "hiring")]


def test_evaluate_rejects_a_non_contract_result():
    class InvalidJudge(FakeJudge):
        def _evaluate_once(self, bout, collegium):
            return cast(JudgeResponse, {"winner": "player_1"})

    config = load_config()
    judge = InvalidJudge(
        config.models.llm,
        load_prompts(config.prompts),
        environ={"GIGACHAT_CREDENTIALS": "secret"},
    )
    bout = load_dataset(config, "pilot", bout_id="1")[0].original

    with pytest.raises(JudgeResponseFormatError, match="expected JudgeResponse"):
        judge.evaluate(bout=bout, collegium="hiring")


def test_effective_parameters_are_an_immutable_snapshot():
    config = load_config().models.llm
    judge = _judge(config=config)

    returned = judge.effective_parameters
    returned["temperature"] = 0.9
    config.parameters["temperature"] = 0.8

    assert judge.effective_parameters == {"temperature": 0.1}
