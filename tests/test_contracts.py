from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from src.experiment.schemas import (
    AttemptError,
    CallRecord,
    JudgeResponse,
    PreparedBout,
    PreparedRoles,
    PreparedRound,
    PreparedTurn,
    TaskKey,
    TechnicalAttempt,
)
from src.judges.errors import (
    JudgeConfigurationError,
    JudgeResponseFormatError,
    PermanentJudgeTransportError,
    TransientJudgeError,
)


NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def _prepared_bout() -> PreparedBout:
    return PreparedBout(
        situation="Ситуация",
        rounds=(
            PreparedRound(
                round_id=1,
                roles=PreparedRoles(player_1="Роль 1", player_2="Роль 2"),
                turns=(
                    PreparedTurn(
                        turn_id=1,
                        speaker="player_1",
                        text="Реплика",
                    ),
                ),
            ),
        ),
    )


def _task() -> TaskKey:
    return TaskKey(
        run_id="run-1",
        model_alias="clef",
        bout_id="1",
        variant="original",
        collegium="hiring",
        repeat_index=1,
    )


def test_prepared_bout_is_whitelist_only_and_immutable():
    bout = _prepared_bout()

    assert set(bout.model_dump()) == {"situation", "rounds"}
    with pytest.raises(ValidationError):
        PreparedBout.model_validate(
            {
                "situation": "Ситуация",
                "rounds": bout.rounds,
                "human_votes": [],
            }
        )
    with pytest.raises(ValidationError):
        bout.situation = "Изменённая ситуация"


def test_judge_response_accepts_provider_probabilities_for_both_players():
    response = JudgeResponse(
        winner="player_1",
        probabilities={"player_1": 0.7, "player_2": 0.3},
        raw_response={"choice": "player_1"},
        usage={"input_tokens": 100},
    )

    assert response.probabilities is not None
    assert response.probabilities.player_1 == 0.7


@pytest.mark.parametrize(
    "response",
    [
        {
            "winner": "draw",
            "probabilities": None,
            "raw_response": {},
        },
        {
            "winner": "player_1",
            "probabilities": {"player_1": 1.0},
            "raw_response": {},
        },
        {
            "winner": "player_1",
            "probabilities": {"player_1": 0.8, "player_2": 0.3},
            "raw_response": {},
        },
        {
            "winner": "player_1",
            "probabilities": {
                "player_1": 0.8,
                "player_2": 0.2,
                "estimated": 0.5,
            },
            "raw_response": {},
        },
    ],
)
def test_judge_response_rejects_invalid_vote_or_probabilities(response):
    with pytest.raises(ValidationError):
        JudgeResponse.model_validate(response)


def test_llm_response_can_omit_probabilities():
    response = JudgeResponse(
        winner="player_2",
        probabilities=None,
        raw_response='{"winner":"player_2"}',
        usage=None,
    )

    assert response.probabilities is None


def test_successful_call_record_is_internally_consistent():
    attempt = TechnicalAttempt(
        attempt_index=1,
        status="success",
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
        raw_response={"choice": "player_1"},
    )
    response = JudgeResponse(
        winner="player_1",
        probabilities={"player_1": 0.6, "player_2": 0.4},
        raw_response={"choice": "player_1"},
    )

    record = CallRecord(
        task=_task(),
        status="success",
        provider="aitunnel",
        api_model_id="clef",
        prompt_path="prompts/common/hiring.txt",
        effective_parameters={},
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
        attempts=(attempt,),
        response=response,
    )

    assert record.response == response
    assert record.error is None


def test_failed_attempt_requires_error_details():
    with pytest.raises(ValidationError, match="failed attempt must contain an error"):
        TechnicalAttempt(
            attempt_index=1,
            status="failed",
            started_at=NOW,
            finished_at=NOW,
        )


def test_failed_call_rejects_successful_attempt():
    attempt = TechnicalAttempt(
        attempt_index=1,
        status="success",
        started_at=NOW,
        finished_at=NOW,
    )
    error = AttemptError(
        error_type="PermanentJudgeTransportError",
        message="HTTP 400",
        retryable=False,
    )

    with pytest.raises(ValidationError, match="cannot contain a successful attempt"):
        CallRecord(
            task=_task(),
            status="failed",
            provider="aitunnel",
            api_model_id="clef",
            prompt_path="prompts/common/hiring.txt",
            effective_parameters={},
            started_at=NOW,
            finished_at=NOW,
            attempts=(attempt,),
            error=error,
        )


def test_adapter_errors_expose_runner_retry_policy():
    assert JudgeConfigurationError.retryable is False
    assert JudgeConfigurationError.abort_run is True
    assert TransientJudgeError.retryable is True
    assert TransientJudgeError.abort_run is False
    assert PermanentJudgeTransportError.retryable is False
    assert PermanentJudgeTransportError.abort_run is False
    assert JudgeResponseFormatError.retryable is True
    assert JudgeResponseFormatError.abort_run is False
