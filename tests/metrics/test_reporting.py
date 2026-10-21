from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.experiment.storage import RunMetadata, RunModelSpec
from src.metrics.aggregation import aggregate_analysis_data
from src.metrics.bootstrap import calculate_bootstrap
from src.metrics.calculations import calculate_metrics
from src.metrics.inference import calculate_inference
from src.metrics.loading import AnalysisInputs, InputFile
from src.metrics.preparation import prepare_analysis_data
from src.metrics.reporting import MANAGED_REPORT_FILES, ReportingError, write_reports
from tests.metrics.test_preparation import _inputs


def _pipeline(tmp_path: Path):
    source = _inputs()
    models = tuple(
        RunModelSpec(
            alias=alias,
            provider="test",
            api_model_id=f"test-{alias}",
            effective_parameters={},
        )
        for alias in ("model-a", "model-b")
    )
    metadata = RunMetadata(
        run_id="run-1",
        mode="pilot",
        dataset="pilot",
        selected_bout_ids=source.metadata.selected_bout_ids,
        models=models,
        prompt_paths={
            "hiring": "hiring.txt",
            "negotiation": "negotiation.txt",
            "ownership": "ownership.txt",
        },
        planned_tasks=source.metadata.planned_tasks,
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    run_dir = tmp_path / "results" / "pilot" / "run-1"
    run_dir.mkdir(parents=True)
    inputs = AnalysisInputs(
        dataset="pilot",
        run=SimpleNamespace(metadata=metadata, run_dir=run_dir),
        latest_records=source.latest_records,
        transcripts=source.transcripts,
        input_files=(InputFile(path=run_dir / "metadata.json", sha256="a" * 64),),
    )
    prepared = prepare_analysis_data(inputs)
    aggregated = aggregate_analysis_data(prepared)
    calculated = calculate_metrics(prepared, aggregated)
    config = SimpleNamespace(
        llm=SimpleNamespace(alias="model-a"),
        decision=SimpleNamespace(alias="model-b"),
    )
    bootstrap = calculate_bootstrap(
        prepared, aggregated, config, replications=20, seed=42
    )
    inference = calculate_inference(
        prepared,
        aggregated,
        config,
        bootstrap_replications=20,
        permutations=20,
        seed=42,
    )
    return inputs, prepared, aggregated, calculated, bootstrap, inference


def _publish(tmp_path: Path):
    args = _pipeline(tmp_path)
    output = write_reports(
        *args,
        clock=lambda: datetime(2026, 10, 20, 12, 30, tzinfo=timezone.utc),
    )
    return output, args


def test_writes_complete_valid_report_set_with_stable_headers(tmp_path: Path):
    output, _ = _publish(tmp_path)

    assert {path.name for path in output.iterdir()} == set(MANAGED_REPORT_FILES)
    payload = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-1"
    assert payload["dataset"] == "pilot"
    assert payload["bootstrap"]["replications"] == 20
    assert payload["inference"]["family_size"] == 7
    assert len(payload["inference"]["tests"]) == 7
    assert payload["input_files"][0]["sha256"] == "a" * 64
    assert payload["exclusions"]
    assert "NaN" not in (output / "metrics.json").read_text(encoding="utf-8")

    with (output / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert any(row["metric"] == "agreement_rate" for row in rows)
    assert any(row["hypothesis_id"] == "H2" for row in rows)
    assert any(row["status"] == "not_applicable" and row["value"] == "" for row in rows)
    assert "N_votes" not in rows[0]
    assert {
        "N_model_votes", "N_human_votes", "N_comparisons"
    }.issubset(rows[0])
    agreement = next(row for row in rows if row["metric"] == "agreement_rate")
    assert agreement["N_model_votes"] == agreement["N_human_votes"]
    assert int(agreement["N_comparisons"]) == int(agreement["N_cases"])
    inference_row = next(row for row in rows if row["hypothesis_id"] == "H2")
    assert all(
        inference_row[field] == ""
        for field in ("N_model_votes", "N_human_votes", "N_comparisons")
    )
    serialized = json.dumps(payload)
    assert '"n_votes"' not in serialized
    assert '"n_model_votes"' in serialized
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Пилотная выборка" in report
    assert "Проверка гипотез H1–H3" in report


def test_failure_during_staging_preserves_previous_reports(tmp_path: Path, monkeypatch):
    output, args = _publish(tmp_path)
    previous = {name: (output / name).read_bytes() for name in MANAGED_REPORT_FILES}

    import src.metrics.reporting as reporting

    original = reporting._write_csv
    calls = 0

    def fail_on_second_csv(*write_args, **write_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic staging failure")
        return original(*write_args, **write_kwargs)

    monkeypatch.setattr(reporting, "_write_csv", fail_on_second_csv)
    with pytest.raises(ReportingError, match="synthetic staging failure"):
        write_reports(*args)

    assert {name: (output / name).read_bytes() for name in MANAGED_REPORT_FILES} == previous
    assert not list(output.parent.glob(".metrics-staging-*"))


def test_republication_replaces_only_managed_files(tmp_path: Path):
    output, args = _publish(tmp_path)
    extra = output / "team-notes.txt"
    extra.write_text("keep me", encoding="utf-8")
    old_metrics = (output / "metrics.json").read_text(encoding="utf-8")

    write_reports(
        *args,
        clock=lambda: datetime(2026, 10, 20, 13, 0, tzinfo=timezone.utc),
    )

    assert extra.read_text(encoding="utf-8") == "keep me"
    assert (output / "metrics.json").read_text(encoding="utf-8") != old_metrics
    assert not list(output.parent.glob(".metrics-staging-*"))
