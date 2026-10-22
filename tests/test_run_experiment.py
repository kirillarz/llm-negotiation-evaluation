from __future__ import annotations

from collections import deque
from io import StringIO
from pathlib import Path

import yaml

from src.experiment.config import DEFAULT_CONFIG_PATH
from src.experiment.schemas import Collegium, JudgeResponse, PreparedBout
from src.judges.base import BaseJudge
from src.judges.errors import (
    JudgeConfigurationError,
    PermanentJudgeTransportError,
    TransientJudgeError,
)
from src.run_experiment import (
    EXIT_INTERRUPTED,
    EXIT_PARTIAL,
    EXIT_PREFLIGHT_ERROR,
    EXIT_SUCCESS,
    run,
)


class CLIFakeJudge(BaseJudge):
    SUPPORTED_MODEL_IDS = frozenset({"gpt-6-luna-pro", "clef"})

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


def _config(tmp_path: Path, *, pilot_dataset: Path | None = None) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    source_root = DEFAULT_CONFIG_PATH.parent.parent
    raw["datasets"] = {
        "pilot": str(pilot_dataset or source_root / "data" / "pilot_dataset"),
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


def _invoke(
    tmp_path: Path,
    argv,
    controller=None,
    environ=None,
    *,
    pilot_dataset: Path | None = None,
):
    controller = controller or {"calls": []}
    stdout = StringIO()
    stderr = StringIO()
    code = run(
        [
            *argv,
            "--config",
            str(_config(tmp_path, pilot_dataset=pilot_dataset)),
        ],
        judge_factories=_factories(controller),
        environ={"AITUNNEL_API_KEY": "secret"} if environ is None else environ,
        output=stdout,
        error_output=stderr,
    )
    return code, stdout.getvalue(), stderr.getvalue(), controller


def test_invalid_selected_file_causes_zero_calls_for_all_adapters(tmp_path: Path):
    dataset = tmp_path / "selected-dataset"
    dataset.mkdir()
    (dataset / "broken.json").write_text("{not valid json", encoding="utf-8")

    code, _, stderr, controller = _invoke(
        tmp_path,
        ["--model", "all", "--dataset", "pilot"],
        pilot_dataset=dataset,
    )

    assert code == EXIT_PREFLIGHT_ERROR
    assert "broken.json" in stderr
    assert controller["calls"] == []
    assert not (tmp_path / "results").exists()


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


def test_missing_shared_aitunnel_secret_causes_zero_calls_and_no_storage(tmp_path: Path):
    code, _, stderr, controller = _invoke(
        tmp_path,
        ["--model", "all", "--dataset", "pilot", "--bout-id", "1", "--dry-run"],
        environ={},
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
    run_log = (
        tmp_path / "results" / "smoke" / run_id / "run.log"
    ).read_text(encoding="utf-8")
    assert "action=resume" in run_log
    assert "retry_failed=true" in run_log
    assert "manual_retries=1" in run_log


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
    run_log = (calls.parent / "run.log").read_text(encoding="utf-8")
    assert " WARNING task.interrupted " in run_log
    assert " WARNING session.interrupted " in run_log
    assert "pending=1" in run_log

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


def test_successful_run_logs_blocks_to_console_and_run_file(tmp_path: Path):
    code, stdout, stderr, _ = _invoke(
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
    )
    run_id = stdout.split("Run ID: ", 1)[1].splitlines()[0]
    run_log = (
        tmp_path / "results" / "smoke" / run_id / "run.log"
    ).read_text(encoding="utf-8")

    assert code == EXIT_SUCCESS
    assert stderr == ""
    assert " INFO session.start " in stdout
    assert "action=new" in stdout
    assert " INFO block.start " in stdout
    assert "tasks=1 manual_retries=0 request_timeout=2m00s max_attempts=2" in stdout
    assert " INFO block.finish " in stdout
    assert "progress=1/1 percent=100.0" in stdout
    assert "duration=" in stdout
    assert "elapsed=" in stdout
    assert "eta=0.0s" in stdout
    assert " INFO session.finish " in stdout
    assert run_log.count("session.start") == 1
    assert run_log.count("block.start") == 1
    assert run_log.count("block.finish") == 1
    assert "player_1" not in run_log
    assert "player_2" not in run_log


def test_retry_is_logged_immediately_without_exception_text(tmp_path: Path):
    sensitive_text = "secret and transcript fragment"
    controller = {
        "calls": [],
        "outcomes": deque([TransientJudgeError(sensitive_text), "player_1"]),
    }

    code, stdout, stderr, _ = _invoke(
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
    run_log = (
        tmp_path / "results" / "smoke" / run_id / "run.log"
    ).read_text(encoding="utf-8")

    assert code == EXIT_SUCCESS
    assert " WARNING attempt.failed " in stderr
    assert "error_type=TransientJudgeError" in stderr
    assert 'message="temporary provider failure"' in stderr
    assert " WARNING attempt.retry " in stderr
    assert sensitive_text not in stderr
    assert sensitive_text not in run_log
    assert "task.failed" not in run_log


def test_resume_appends_a_distinct_session_to_run_log(tmp_path: Path):
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
    )
    run_id = stdout.split("Run ID: ", 1)[1].splitlines()[0]
    assert code == EXIT_SUCCESS

    resumed_code, _, _, resumed = _invoke(tmp_path, ["--resume", run_id])
    run_log = (
        tmp_path / "results" / "smoke" / run_id / "run.log"
    ).read_text(encoding="utf-8")

    assert resumed_code == EXIT_SUCCESS
    assert resumed["calls"] == []
    assert run_log.count("session.start") == 2
    assert run_log.count("session.finish") == 2
    assert "action=resume" in run_log
    assert "selected=0" in run_log


def test_final_task_failure_is_logged_and_session_is_partial(tmp_path: Path):
    sensitive_text = "must not reach operational log"
    controller = {
        "calls": [],
        "outcomes": deque([PermanentJudgeTransportError(sensitive_text)]),
    }

    code, stdout, stderr, _ = _invoke(
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
    run_log = (
        tmp_path / "results" / "smoke" / run_id / "run.log"
    ).read_text(encoding="utf-8")

    assert code == EXIT_PARTIAL
    assert " ERROR task.failed " in stderr
    assert "abort_run=false" in stderr
    assert " WARNING block.finish " in stderr
    assert " WARNING session.finish " in stderr
    assert "status=partial" in run_log
    assert sensitive_text not in run_log


def test_runtime_configuration_failure_logs_abort_and_stops_run(tmp_path: Path):
    controller = {
        "calls": [],
        "outcomes": deque([JudgeConfigurationError("unsafe details")]),
    }

    code, stdout, stderr, _ = _invoke(
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
    run_log = (
        tmp_path / "results" / "smoke" / run_id / "run.log"
    ).read_text(encoding="utf-8")

    assert code == EXIT_PREFLIGHT_ERROR
    assert " ERROR task.failed " in stderr
    assert "abort_run=true" in stderr
    assert " ERROR session.aborted " in stderr
    assert "error_type=JudgeConfigurationError" in run_log
    assert "unsafe details" not in run_log
