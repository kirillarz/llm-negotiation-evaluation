from __future__ import annotations

import json

import pytest
import requests

from src.experiment.config import load_config
from src.experiment.data_loader import load_dataset
from src.experiment.prompts import load_prompts
from src.judges.errors import (
    JudgeConfigurationError,
    JudgeResponseFormatError,
    PermanentJudgeTransportError,
    TransientJudgeError,
)
from src.judges.llm.judge import LLMJudge


class FakeResponse:
    def __init__(
        self,
        payload: object | None = None,
        *,
        status_code: int = 200,
        json_error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = "provider detail that must not enter adapter errors"
        self.json_error = json_error

    def json(self) -> object:
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class FakePost:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _response(content: str = '{"choice":"player_2"}') -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "gpt-6-luna-pro",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": 101,
            "completion_tokens": 8,
            "total_tokens": 109,
            "cost_rub": 0.01,
        },
        "authorization": "must-not-be-saved",
    }


def _judge(post: FakePost, *, config=None, environ=None) -> LLMJudge:
    experiment_config = load_config()
    return LLMJudge(
        config or experiment_config.models.llm,
        load_prompts(experiment_config.prompts),
        environ={"AITUNNEL_API_KEY": "test-secret"} if environ is None else environ,
        post=post,
    )


def _bout():
    config = load_config()
    return load_dataset(config, "pilot", bout_id="1")[0].original


def test_one_vote_posts_full_bout_with_minimal_strict_choice_schema():
    post = FakePost(FakeResponse(_response()))
    judge = _judge(post)
    bout = _bout()

    first = judge.evaluate(bout=bout, collegium="hiring")
    second = judge.evaluate(bout=bout, collegium="hiring")

    assert len(post.calls) == 2
    _, request = post.calls[0]
    assert request["headers"]["Authorization"] == "Bearer test-secret"
    assert request["timeout"] == 120
    assert request["allow_redirects"] is False
    payload = request["json"]
    assert payload["model"] == "gpt-6-luna-pro"
    assert payload["max_tokens"] == 64
    assert payload["verbosity"] == "low"
    assert payload["reasoning"] == {"effort": "none", "exclude": True}
    messages = payload["messages"]
    assert len(messages) == 2
    assert messages[0]["content"] == judge.prompt_for("hiring").text
    user_content = messages[1]["content"]
    sent_bout = json.loads(user_content.split("\n", 1)[1].split("\nВыбери", 1)[0])
    assert sent_bout == bout.model_dump(mode="json")
    assert "human_votes" not in user_content
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert set(schema["properties"]) == {"choice"}
    assert schema["additionalProperties"] is False

    assert first.winner == second.winner == "player_2"
    assert first.probabilities is None
    assert first.usage == {
        "prompt_tokens": 101,
        "completion_tokens": 8,
        "total_tokens": 109,
        "cost_rub": 0.01,
    }
    assert first.raw_response["authorization"] == "[REDACTED]"


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"choice":"draw"}',
        '{"choice":"player_1","reason":"extra"}',
        '{"winner":"player_1"}',
    ],
)
def test_invalid_structured_response_is_format_error(content: str):
    judge = _judge(FakePost(FakeResponse(_response(content))))
    with pytest.raises(JudgeResponseFormatError):
        judge.evaluate(bout=_bout(), collegium="ownership")


def test_length_finish_reason_is_rejected_instead_of_parsing_partial_output():
    payload = _response()
    payload["choices"][0]["finish_reason"] = "length"
    judge = _judge(FakePost(FakeResponse(payload)))
    with pytest.raises(JudgeResponseFormatError, match="finish_reason"):
        judge.evaluate(bout=_bout(), collegium="hiring")


@pytest.mark.parametrize("error", [requests.Timeout(), requests.ConnectionError()])
def test_transport_errors_are_exposed_without_internal_retry(error):
    post = FakePost(error)
    judge = _judge(post)
    with pytest.raises(TransientJudgeError):
        judge.evaluate(bout=_bout(), collegium="negotiation")
    assert len(post.calls) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_temporary_http_failures_are_retryable(status: int):
    judge = _judge(FakePost(FakeResponse({}, status_code=status)))
    with pytest.raises(TransientJudgeError):
        judge.evaluate(bout=_bout(), collegium="hiring")


@pytest.mark.parametrize("status", [401, 402, 403, 404])
def test_configuration_http_failures_abort_run_without_secret(status: int):
    judge = _judge(FakePost(FakeResponse({}, status_code=status)))
    with pytest.raises(JudgeConfigurationError) as raised:
        judge.evaluate(bout=_bout(), collegium="hiring")
    assert raised.value.abort_run
    assert "test-secret" not in str(raised.value)


def test_bad_request_is_permanent():
    judge = _judge(FakePost(FakeResponse({}, status_code=400)))
    with pytest.raises(PermanentJudgeTransportError) as raised:
        judge.evaluate(bout=_bout(), collegium="hiring")
    assert not raised.value.retryable


def test_invalid_provider_json_is_format_error():
    response = FakeResponse(json_error=ValueError("bad json"))
    with pytest.raises(JudgeResponseFormatError):
        _judge(FakePost(response)).evaluate(bout=_bout(), collegium="hiring")


def test_preflight_rejects_missing_shared_aitunnel_secret_without_calling_api():
    post = FakePost(FakeResponse(_response()))
    judge = _judge(post, environ={"UNRELATED": "value"})
    with pytest.raises(JudgeConfigurationError, match="AITUNNEL_API_KEY"):
        judge.preflight()
    assert post.calls == []


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"max_tokens": 8}, "max_tokens"),
        ({"max_tokens": "64"}, "max_tokens"),
        ({"max_tokens": 64, "messages": []}, "cannot be configured"),
        ({"max_tokens": 64, "reasoning": "minimal"}, "reasoning must be"),
        (
            {"max_tokens": 64, "reasoning": {"effort": "extreme"}},
            "reasoning.effort",
        ),
        ({"max_tokens": 64, "verbosity": "tiny"}, "verbosity"),
    ],
)
def test_preflight_rejects_invalid_minimal_output_parameters(parameters, message):
    config = load_config().models.llm.model_copy(update={"parameters": parameters})
    judge = _judge(FakePost(FakeResponse(_response())), config=config)
    with pytest.raises(JudgeConfigurationError, match=message):
        judge.preflight()
