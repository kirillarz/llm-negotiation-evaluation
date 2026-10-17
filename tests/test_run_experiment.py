from __future__ import annotations

from collections import deque
from io import StringIO
from pathlib import Path

import yaml

from src.experiment.config import DEFAULT_CONFIG_PATH
from src.experiment.schemas import Collegium, JudgeResponse, PreparedBout
from src.judges.base import BaseJudge
from src.judges.errors import PermanentJudgeTransportError
from src.run_experiment import (
    EXIT_INTERRUPTED,
    EXIT_PARTIAL,
    EXIT_PREFLIGHT_ERROR,
    EXIT_SUCCESS,
    run,
)


class CLIFakeJudge(BaseJudge):
    SUPPORTED_MODEL_IDS = frozenset({"GigaChat-2-Pro", "clef"})

    def __init__(self, *args, controller, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller = controller

    def _evaluate_once(
        self, bout: PreparedBout, collegium: Collegium
    ) -> JudgeResponse:
        self.controller["calls"].append((self.model_alias, collegium, bout))
        outcomes = self.controller.get("outcomes")
        outcome = outcomes.popleft() if outcomes else "player_1"
        if isinstance(outcome, BaseException):
            raise outcome
        return JudgeResponse(
            winner=outcome,
            probabilities=None,
            raw_response={"winner": outcome},
        )


def _config(tmp_path: Path) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    source_root = DEFAULT_CONFIG_PATH.parent.parent
    raw["datasets"] = {
        "pilot": str(source_root / "data" / "pilot_dataset"),
        "main": str(source_root / "data" / "dataset"),
    }
    raw["prompts"] = {
        name: str(source_root / "prompts" / "common" / f"{name}.txt")
        for name in ("hiring", "negotiation", "ownership")
    }
    raw["results"] = {
        name: str(tmp_path / "results" / name)
        for name in ("smoke", "pilot", "main")
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def _factories(controller):
    def factory(config, prompts, environ):
        return CLIFakeJudge(
            config,
            prompts,
            environ=environ,
            controller=controller,
        )

    return {"llm": factory, "decision": factory}


def _invoke(tmp_path: Path, argv, controller=None, environ=None):
    controller = controller or {"calls": []}
    stdout = StringIO()
    stderr = StringIO()
    code = run(
        [*argv, "--config", str(_config(tmp_path))],
        judge_factories=_factories(controller),
        environ=environ
        or {
            "GIGACHAT_CREDENTIALS": "secret",
            "AITUNNEL_API_KEY": "secret",
        },
        output=stdout,
        error_output=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue(), controller


def test_dry_run_preflights_everything_without_creating_results(tmp_path: Path):
    code, stdout, stderr, controller = _invoke(
        tmp_path,
        [
            "--model",
            "all",
            "--dataset",
            "pilot",
            "--bout-id",
            "1",
            "--dry-run",
        ],
    )

    assert code == EXIT_SUCCESS
    assert stderr == ""
    assert controller["calls"] == []
    assert "Total planned calls: 36" in stdout
    assert "Selected bout IDs: 1" in stdout
    assert "no run_id was created" in stdout
    assert not (tmp_path / "results").exists()


def test_missing_one_adapter_secret_causes_zero_calls_and_no_storage(tmp_path: Path):
    code, _, stderr, controller = _invoke(
        tmp_path,
        ["--model", "all", "--dataset", "pilot", "--bout-id", "1", "--dry-run"],
        environ={"GIGACHAT_CREDENTIALS": "secret"},
    )

    assert code == EXIT_PREFLIGHT_ERROR
    assert "AITUNNEL_API_KEY" in stderr
    assert controller["calls"] == []
    assert not (tmp_path / "results").exists()


def test_main_real_run_requires_confirmation_but_dry_run_does_not(tmp_path: Path):
    dry_code, _, _, _ = _invoke(
        tmp_path,
        ["--model", "llm", "--dataset", "main", "--bout-id", "6", "--dry-run"],
    )
    real_code, _, stderr, controller = _invoke(
        tmp_path,
        ["--model", "llm", "--dataset", "main", "--bout-id", "6"],
    )

    assert dry_code == EXIT_SUCCESS
    assert real_code == EXIT_PREFLIGHT_ERROR
    assert "--confirm-main" in stderr
    assert controller["calls"] == []
    assert not (tmp_path / "results" / "main").exists()


def test_smoke_run_is_saved_and_resume_does_not_repeat_success(tmp_path: Path):
    code, stdout, _, controller = _invoke(
        tmp_path,
        [
            "--model",
            "all",
            "--dataset",
            "pilot",
            "--bout-id",
            "1",
            "--smoke",
            "--collegium",
            "hiring",
        ],
    )
    run_id = stdout.split("Run ID: ", 1)[1].splitlines()[0]

    assert code == EXIT_SUCCESS
    assert len(controller["calls"]) == 2
    run_dir = tmp_path / "results" / "smoke" / run_id
    assert len((run_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()) == 2

    resumed_code, _, _, resumed_controller = _invoke(
        tmp_path,
        ["--resume", run_id],
    )
    assert resumed_code == EXIT_SUCCESS
    assert resumed_controller["calls"] == []


def test_retry_failed_is_an_explicit_linked_revision(tmp_path: Path):
    controller = {
        "calls": [],
        "outcomes": deque([PermanentJudgeTransportError("bad request")]),
    }
    code, stdout, _, _ = _invoke(
        tmp_path,
        [
            "--model",
            "llm",
            "--dataset",
            "pilot",
            "--bout-id",
            "1",
            "--smoke",
            "--collegium",
            "hiring",
        ],
        controller=controller,
    )
    run_id = stdout.split("Run ID: ", 1)[1].splitlines()[0]
    assert code == EXIT_PARTIAL

    retry_controller = {"calls": [], "outcomes": deque(["player_2"])}
    retry_code, _, _, _ = _invoke(
        tmp_path,
        ["--resume", run_id, "--retry-failed"],
        controller=retry_controller,
    )

    assert retry_code == EXIT_SUCCESS
    assert len(retry_controller["calls"]) == 1
    lines = (
        tmp_path / "results" / "smoke" / run_id / "calls.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert '"record_revision":2' in lines[1]
    assert '"previous_revision":1' in lines[1]
    assert '"manual_retry":true' in lines[1]


def test_keyboard_interrupt_leaves_task_pending_for_resume(tmp_path: Path):
    controller = {"calls": [], "outcomes": deque([KeyboardInterrupt()])}
    code, stdout, _, _ = _invoke(
        tmp_path,
        [
            "--model",
            "llm",
            "--dataset",
            "pilot",
            "--bout-id",
            "1",
            "--smoke",
            "--collegium",
            "hiring",
        ],
        controller=controller,
    )
    run_id = stdout.split("Run ID: ", 1)[1].splitlines()[0]

    assert code == EXIT_INTERRUPTED
    calls = tmp_path / "results" / "smoke" / run_id / "calls.jsonl"
    assert calls.read_text(encoding="utf-8") == ""

    resumed_code, _, _, resumed = _invoke(tmp_path, ["--resume", run_id])
    assert resumed_code == EXIT_SUCCESS
    assert len(resumed["calls"]) == 1


def test_resume_rejects_attempts_to_override_persisted_parameters(tmp_path: Path):
    code, _, stderr, _ = _invoke(
        tmp_path,
        ["--resume", "some-run", "--dataset", "main"],
    )

    assert code == EXIT_PREFLIGHT_ERROR
    assert "--resume cannot be combined" in stderr
