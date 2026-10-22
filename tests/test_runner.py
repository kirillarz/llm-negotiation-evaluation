from __future__ import annotations

import pytest

from src.experiment.config import load_config
from src.experiment.data_loader import load_dataset
from src.experiment.prompts import COLLEGIA, load_prompts
from src.experiment.runner import (
    REPEAT_INDICES,
    VARIANTS,
    TaskPlanError,
    build_task_plan,
)
from src.experiment.schemas import Collegium, JudgeResponse, PreparedBout
from src.judges.base import BaseJudge


class PlanningJudge(BaseJudge):
    SUPPORTED_MODEL_IDS = frozenset({"gpt-6-luna-pro", "clef"})

    def _evaluate_once(
        self,
        bout: PreparedBout,
        collegium: Collegium,
    ) -> JudgeResponse:
        return JudgeResponse(
            winner="player_1",
            probabilities=None,
            raw_response={"winner": "player_1"},
        )


def _inputs():
    config = load_config()
    prompts = load_prompts(config.prompts)
    bouts = load_dataset(config, "pilot")
    judges = (
        PlanningJudge(
            config.models.llm,
            prompts,
            environ={"AITUNNEL_API_KEY": "secret"},
        ),
        PlanningJudge(
            config.models.decision,
            prompts,
            environ={"AITUNNEL_API_KEY": "secret"},
        ),
    )
    return bouts, judges


def test_full_plan_has_6_tasks_per_model_and_bout_in_stable_order():
    bouts, judges = _inputs()
    selected_bouts = (bouts[0],)

    plan = build_task_plan(
        run_id="run-1",
        bouts=selected_bouts,
        judges=judges,
    )

    assert len(plan) == 12
    expected_dimensions = [
        (model_alias, variant, collegium, repeat_index)
        for model_alias in ("gpt6-luna-pro", "clef")
        for variant in VARIANTS
        for collegium in COLLEGIA
        for repeat_index in REPEAT_INDICES
    ]
    assert [
        (
            task.key.model_alias,
            task.key.variant,
            task.key.collegium,
            task.key.repeat_index,
        )
        for task in plan
    ] == expected_dimensions
    assert all(task.key.run_id == "run-1" for task in plan)
    assert all(task.key.bout_id == "1" for task in plan)


def test_full_plan_uses_bout_then_model_order():
    bouts, judges = _inputs()

    plan = build_task_plan(
        run_id="run-1",
        bouts=bouts[:2],
        judges=judges,
    )

    assert len(plan) == 24
    blocks = [
        (plan[index].key.bout_id, plan[index].key.model_alias)
        for index in range(0, len(plan), 6)
    ]
    assert blocks == [
        ("1", "gpt6-luna-pro"),
        ("1", "clef"),
        ("2", "gpt6-luna-pro"),
        ("2", "clef"),
    ]


def test_smoke_plan_has_one_original_task_per_selected_model():
    bouts, judges = _inputs()

    plan = build_task_plan(
        run_id="smoke-1",
        bouts=bouts,
        judges=judges,
        smoke=True,
        bout_id="2",
        collegium="ownership",
    )

    assert len(plan) == 2
    assert [task.key.model_alias for task in plan] == ["gpt6-luna-pro", "clef"]
    assert all(task.key.bout_id == "2" for task in plan)
    assert all(task.key.variant == "original" for task in plan)
    assert all(task.key.collegium == "ownership" for task in plan)
    assert all(task.key.repeat_index == 1 for task in plan)


@pytest.mark.parametrize(
    ("bout_id", "collegium", "message"),
    [
        (None, "hiring", "explicit bout_id"),
        ("1", None, "explicit collegium"),
    ],
)
def test_smoke_requires_explicit_bout_and_collegium(
    bout_id,
    collegium,
    message,
):
    bouts, judges = _inputs()

    with pytest.raises(TaskPlanError, match=message):
        build_task_plan(
            run_id="smoke-1",
            bouts=bouts,
            judges=judges,
            smoke=True,
            bout_id=bout_id,
            collegium=collegium,
        )


def test_full_plan_does_not_accept_a_collegium_restriction():
    bouts, judges = _inputs()

    with pytest.raises(TaskPlanError, match="only restrict a smoke plan"):
        build_task_plan(
            run_id="run-1",
            bouts=(bouts[0],),
            judges=judges,
            collegium="hiring",
        )


def test_each_collegium_has_one_vote_per_variant():
    bouts, judges = _inputs()
    plan = build_task_plan(
        run_id="run-1",
        bouts=(bouts[0],),
        judges=(judges[0],),
    )
    original = plan[:3]

    assert [task.key.repeat_index for task in original] == [1, 1, 1]
    assert [task.key.collegium for task in original] == list(COLLEGIA)
    assert all(task.bout is original[0].bout for task in original)
    assert all(task.judge is original[0].judge for task in original)
    assert all(not hasattr(task, "response") for task in original)
    assert all(not hasattr(task, "history") for task in original)


def test_plan_rejects_empty_inputs_and_duplicate_model_aliases():
    bouts, judges = _inputs()

    with pytest.raises(TaskPlanError, match="without prepared bouts"):
        build_task_plan(run_id="run-1", bouts=(), judges=judges)
    with pytest.raises(TaskPlanError, match="without judge adapters"):
        build_task_plan(run_id="run-1", bouts=(bouts[0],), judges=())
    with pytest.raises(TaskPlanError, match="aliases must be unique"):
        build_task_plan(
            run_id="run-1",
            bouts=(bouts[0],),
            judges=(judges[0], judges[0]),
        )
