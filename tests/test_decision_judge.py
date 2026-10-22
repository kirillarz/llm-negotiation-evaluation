from __future__ import annotations

from typing import Any

import pytest
import requests

from src.experiment.config import load_config
from src.experiment.data_loader import load_dataset
from src.experiment.prompts import load_prompts
from src.judges.decision.judge import DecisionJudge
from src.judges.errors import (
    JudgeConfigurationError,
    JudgeResponseFormatError,
    PermanentJudgeTransportError,
    TransientJudgeError,
)


class FakeResponse:
    def __init__(self, status_code: int = 200, body: object = None) -> None:
        self.status_code = status_code
        self.body = body

    def json(self) -> object:
        if isinstance(self.body, BaseException):
            raise self.body
        return self.body


class FakePost:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[tuple[tuple[object, ...], dict[str, Any]]] = []

    def __call__(self, *args: object, **kwargs: Any) -> FakeResponse:
        self.calls.append((args, kwargs))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        assert isinstance(self.outcome, FakeResponse)
        return self.outcome


def _response(**vote_fields: object) -> FakeResponse:
    vote = {
        "type": "choice",
        "choice": "player_2",
        "probabilities": {"player_1": 0.25, "player_2": 0.75},
        "confidence": 0.9,
        **vote_fields,
    }
    return FakeResponse(
        body={
            "id": "decision-123",
            "model": "clef",
            "answers": {"vote": vote},
            "usage": {"input_tokens": 101, "cost_rub": 0.12},
            "authorization": "must-not-be-saved",
        }
    )


def _judge(post: FakePost, *, config=None, environ=None) -> DecisionJudge:
    experiment_config = load_config()
    return DecisionJudge(
        config or experiment_config.models.decision,
        load_prompts(experiment_config.prompts),
        environ=environ or {"AITUNNEL_API_KEY": "test-secret"},
        post=post,
    )


def _bout():
    config = load_config()
    return load_dataset(config, "pilot", bout_id="1")[0].original


def test_one_vote_sends_full_state_and_normalizes_real_provider_fields():
    post = FakePost(_response())
    judge = _judge(post)
    bout = _bout()

    result = judge.evaluate(bout=bout, collegium="ownership")

    assert len(post.calls) == 1
    args, kwargs = post.calls[0]
    assert args == ("https://api.aitunnel.ru/v1/decisions",)
    assert kwargs["headers"] == {
        "Authorization": "Bearer test-secret",
        "Content-Type": "application/json",
    }
    assert kwargs["timeout"] == 120
    assert kwargs["allow_redirects"] is False
    payload = kwargs["json"]
    assert payload["model"] == "clef"
    assert payload["state"] == bout.model_dump(mode="json")
    assert "human_votes" not in str(payload["state"])
    assert payload["questions"] == {
        "vote": {
            "type": "choice",
            "instructions": judge.prompt_for("ownership").text,
            "criteria": {
                "player_1": (
                    "Выбрать player_1 только если именно он предпочтительнее как "
                    "управляющего, которому вы доверите свою собственность или ресурс; "
                    "общая победа в поединке не является критерием"
                ),
                "player_2": (
                    "Выбрать player_2 только если именно он предпочтительнее как "
                    "управляющего, которому вы доверите свою собственность или ресурс; "
                    "общая победа в поединке не является критерием"
                ),
            },
        }
    }

    assert result.winner == "player_2"
    assert result.probabilities is not None
    assert result.probabilities.model_dump() == {"player_1": 0.25, "player_2": 0.75}
    assert result.usage == {"input_tokens": 101, "cost_rub": 0.12}
    assert result.raw_response["answers"]["vote"]["confidence"] == 0.9
    assert result.raw_response["authorization"] == "[REDACTED]"
    assert result.raw_response["_adapter"] == {
        "canonical_labels_swapped": False
    }


def test_label_swapped_bouts_use_identical_canonical_state_and_map_results_back():
    config = load_config()
    variants = load_dataset(config, "pilot", bout_id="1")[0]
    post = FakePost(_response())
    judge = _judge(post)

    original = judge.evaluate(bout=variants.original, collegium="hiring")
    swapped = judge.evaluate(bout=variants.swapped, collegium="hiring")

    original_payload = post.calls[0][1]["json"]
    swapped_payload = post.calls[1][1]["json"]
    assert original_payload["state"] == swapped_payload["state"]
    assert original.winner == "player_2"
    assert swapped.winner == "player_1"
    assert original.probabilities is not None
    assert swapped.probabilities is not None
    assert original.probabilities.model_dump() == {
        "player_1": 0.25,
        "player_2": 0.75,
    }
    assert swapped.probabilities.model_dump() == {
        "player_1": 0.75,
        "player_2": 0.25,
    }
    assert swapped.raw_response["_adapter"] == {
        "canonical_labels_swapped": True
    }


def test_each_vote_is_exactly_one_request_without_internal_retry():
    post = FakePost(FakeResponse(status_code=429))
    judge = _judge(post)

    with pytest.raises(TransientJudgeError):
        judge.evaluate(bout=_bout(), collegium="hiring")

    assert len(post.calls) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_http_statuses_are_transient(status: int):
    with pytest.raises(TransientJudgeError):
        _judge(FakePost(FakeResponse(status_code=status))).evaluate(
            bout=_bout(), collegium="negotiation"
        )


def test_timeout_is_transient():
    post = FakePost(requests.Timeout("late"))
    with pytest.raises(TransientJudgeError):
        _judge(post).evaluate(bout=_bout(), collegium="hiring")
    assert len(post.calls) == 1


def test_connection_error_is_transient():
    post = FakePost(requests.ConnectionError("reset"))
    with pytest.raises(TransientJudgeError):
        _judge(post).evaluate(bout=_bout(), collegium="hiring")
    assert len(post.calls) == 1


@pytest.mark.parametrize("status", [401, 403, 404])
def test_auth_endpoint_and_model_statuses_abort_run(status: int):
    with pytest.raises(JudgeConfigurationError) as raised:
        _judge(FakePost(FakeResponse(status_code=status))).evaluate(
            bout=_bout(), collegium="hiring"
        )
    assert raised.value.abort_run


@pytest.mark.parametrize("status", [400, 402, 422])
def test_request_errors_are_non_retryable(status: int):
    with pytest.raises(PermanentJudgeTransportError) as raised:
        _judge(FakePost(FakeResponse(status_code=status))).evaluate(
            bout=_bout(), collegium="hiring"
        )
    assert not raised.value.retryable


@pytest.mark.parametrize(
    "body",
    [
        ValueError("broken JSON"),
        [],
        {},
        {"answers": {"vote": {"type": "choice", "choice": "draw"}}},
        {"answers": {"vote": {"type": "text", "choice": "player_1"}}},
        {
            "answers": {
                "vote": {
                    "type": "choice",
                    "choice": "player_1",
                    "probabilities": {"player_1": 0.8, "player_2": 0.3},
                }
            }
        },
        {
            "answers": {"vote": {"type": "choice", "choice": "player_1"}},
            "usage": {"input_tokens": -1},
        },
    ],
)
def test_invalid_provider_responses_are_retryable_format_errors(body: object):
    post = FakePost(FakeResponse(body=body))
    with pytest.raises(JudgeResponseFormatError) as raised:
        _judge(post).evaluate(bout=_bout(), collegium="ownership")
    assert raised.value.retryable


def test_optional_probabilities_and_usage_can_be_absent():
    response = FakeResponse(
        body={"answers": {"vote": {"type": "choice", "choice": "player_1"}}}
    )
    result = _judge(FakePost(response)).evaluate(
        bout=_bout(), collegium="negotiation"
    )
    assert result.winner == "player_1"
    assert result.probabilities is None
    assert result.usage is None


def test_preflight_rejects_missing_secret_and_reserved_parameters():
    missing = _judge(FakePost(_response()), environ={"AITUNNEL_API_KEY": " "})
    with pytest.raises(JudgeConfigurationError, match="AITUNNEL_API_KEY"):
        missing.preflight()

    config = load_config().models.decision.model_copy(
        update={"parameters": {"state": {}}}
    )
    with pytest.raises(JudgeConfigurationError, match="cannot be configured: state"):
        _judge(FakePost(_response()), config=config).preflight()
