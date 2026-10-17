from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from gigachat.exceptions import (
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServerError,
)
from gigachat.models import ChatCompletion

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


def _response(content: str = '{"winner":"player_2"}') -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "created": 1_700_000_000,
            "model": "GigaChat-2-Pro",
            "usage": {
                "prompt_tokens": 101,
                "completion_tokens": 5,
                "total_tokens": 106,
            },
            "object": "chat.completion",
            "x_headers": {
                "x-request-id": "request-123",
                "authorization": "must-not-be-saved",
            },
        }
    )


class FakeClient:
    def __init__(self, factory: "FakeClientFactory") -> None:
        self.factory = factory
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def chat(self, payload: dict[str, object]) -> object:
        self.calls.append(payload)
        if isinstance(self.factory.outcome, Exception):
            raise self.factory.outcome
        return self.factory.outcome


class FakeClientFactory:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.kwargs: list[dict[str, Any]] = []
        self.clients: list[FakeClient] = []

    def __call__(self, **kwargs: Any) -> FakeClient:
        self.kwargs.append(kwargs)
        client = FakeClient(self)
        self.clients.append(client)
        return client


def _judge(factory: FakeClientFactory, *, config=None) -> LLMJudge:
    experiment_config = load_config()
    return LLMJudge(
        config or experiment_config.models.llm,
        load_prompts(experiment_config.prompts),
        environ={"GIGACHAT_CREDENTIALS": "test-secret"},
        client_factory=factory,
    )


def _bout():
    config = load_config()
    return load_dataset(config, "pilot", bout_id="1")[0].original


def test_one_vote_uses_fresh_client_full_bout_and_strict_winner_schema():
    factory = FakeClientFactory(_response())
    judge = _judge(factory)
    bout = _bout()

    first = judge.evaluate(bout=bout, collegium="hiring")
    second = judge.evaluate(bout=bout, collegium="hiring")

    assert len(factory.clients) == 2
    assert all(len(client.calls) == 1 and client.closed for client in factory.clients)
    assert factory.kwargs == [
        {"credentials": "test-secret", "timeout": 120, "max_retries": 0},
        {"credentials": "test-secret", "timeout": 120, "max_retries": 0},
    ]

    payload = factory.clients[0].calls[0]
    assert payload["model"] == "GigaChat-2-Pro"
    assert payload["temperature"] == 0.1
    messages = payload["messages"]
    assert isinstance(messages, list) and len(messages) == 2
    assert messages[0]["content"] == judge.prompt_for("hiring").text
    sent_bout = json.loads(messages[1]["content"].split("\n", 1)[1])
    assert sent_bout == bout.model_dump(mode="json")
    assert "human_votes" not in messages[1]["content"]
    assert payload["response_format"]["strict"] is True
    assert set(payload["response_format"]["schema"]["properties"]) == {"winner"}

    assert first.winner == second.winner == "player_2"
    assert first.probabilities is None
    assert first.usage == {
        "prompt_tokens": 101,
        "completion_tokens": 5,
        "total_tokens": 106,
        "precached_prompt_tokens": None,
    }
    assert first.raw_response["choices"][0]["message"]["content"] == (
        '{"winner":"player_2"}'
    )
    assert first.raw_response["x_headers"] == {
        "x-request-id": "request-123",
        "authorization": "[REDACTED]",
    }


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"winner":"draw"}',
        '{"winner":"player_1","reason":"extra"}',
    ],
)
def test_invalid_structured_response_is_retryable_format_error(content: str):
    judge = _judge(FakeClientFactory(_response(content)))

    with pytest.raises(JudgeResponseFormatError):
        judge.evaluate(bout=_bout(), collegium="ownership")


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("timed out"),
        RateLimitError("https://example.invalid", 429, b"limited", None),
    ],
)
def test_temporary_sdk_failures_are_exposed_to_runner_without_internal_retry(error):
    factory = FakeClientFactory(error)
    judge = _judge(factory)

    with pytest.raises(TransientJudgeError):
        judge.evaluate(bout=_bout(), collegium="negotiation")

    assert len(factory.clients) == 1
    assert len(factory.clients[0].calls) == 1


def test_authentication_failure_is_run_wide_and_does_not_expose_secret():
    error = AuthenticationError("https://example.invalid", 401, b"denied", None)
    judge = _judge(FakeClientFactory(error))

    with pytest.raises(JudgeConfigurationError) as raised:
        judge.evaluate(bout=_bout(), collegium="hiring")

    assert raised.value.abort_run
    assert "test-secret" not in str(raised.value)
    assert b"denied".decode() not in str(raised.value)


def test_server_error_is_retryable_and_bad_request_is_permanent():
    server_error = ServerError("https://example.invalid", 500, b"failed", None)
    with pytest.raises(TransientJudgeError):
        _judge(FakeClientFactory(server_error)).evaluate(
            bout=_bout(), collegium="hiring"
        )

    bad_request = BadRequestError("https://example.invalid", 400, b"bad", None)
    with pytest.raises(PermanentJudgeTransportError) as raised:
        _judge(FakeClientFactory(bad_request)).evaluate(
            bout=_bout(), collegium="hiring"
        )
    assert not raised.value.retryable


def test_preflight_rejects_missing_llm_secret_without_calling_sdk():
    factory = FakeClientFactory(_response())
    config = load_config()
    judge = LLMJudge(
        config.models.llm,
        load_prompts(config.prompts),
        environ={},
        client_factory=factory,
    )

    with pytest.raises(JudgeConfigurationError, match="GIGACHAT_CREDENTIALS"):
        judge.preflight()

    assert factory.clients == []


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"temperature": 0.1, "n": 2}, "n must be exactly 1"),
        ({"temperature": 0.1, "stream": True}, "streaming is not supported"),
        ({"temperature": 0.1, "messages": []}, "cannot be configured"),
        ({"temperature": 0.1, "unknown": 1}, "Unsupported GigaChat parameters"),
        ({"temperature": "hot"}, "Invalid GigaChat parameter"),
    ],
)
def test_preflight_rejects_parameters_that_break_single_independent_vote(
    parameters, message
):
    config = load_config().models.llm.model_copy(update={"parameters": parameters})
    judge = _judge(FakeClientFactory(_response()), config=config)

    with pytest.raises(JudgeConfigurationError, match=message):
        judge.preflight()
