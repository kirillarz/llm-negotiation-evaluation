from __future__ import annotations

from io import StringIO
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.calculate_metrics import (
    EXIT_SUCCESS,
    EXIT_USAGE_ERROR,
    SMOKE_UNSUPPORTED_MESSAGE,
    run,
)
from src.metrics import (
    BOOTSTRAP_REPLICATIONS,
    BOOTSTRAP_SEED,
    CALCULATION_RULES_VERSION,
    COLLEGIA,
    CONFIDENCE_LEVEL,
    EXACT_PERMUTATION_MAX_BOUTS,
    INFERENCE_ALPHA,
    MONTE_CARLO_PERMUTATIONS,
    PERCENTILE_BOUNDS,
    PERCENTILE_METHOD,
)
from src.metrics.contracts import (
    BoutCollegiumCase,
    Exclusion,
    MetricResult,
    MetricScope,
    NormalizedVote,
)


def test_algorithm_constants_are_fixed():
    assert CALCULATION_RULES_VERSION == "1.2.0"
    assert COLLEGIA == ("hiring", "negotiation", "ownership")
    assert BOOTSTRAP_REPLICATIONS == 10_000
    assert BOOTSTRAP_SEED == 42
    assert CONFIDENCE_LEVEL == 0.95
    assert PERCENTILE_BOUNDS == (2.5, 97.5)
    assert PERCENTILE_METHOD == "linear_type_7"
    assert INFERENCE_ALPHA == 0.05
    assert EXACT_PERMUTATION_MAX_BOUTS == 20
    assert MONTE_CARLO_PERMUTATIONS == 100_000


def test_normalized_vote_enforces_source_specific_fields():
    vote = NormalizedVote(
        source="model",
        bout_id="bout-1",
        collegium="hiring",
        ordinal=1,
        winner="player_1",
        model_alias="clef",
        variant="original",
        probability_player_1=0.7,
    )

    assert vote.winner == "player_1"
    with pytest.raises(ValidationError, match="model votes require"):
        NormalizedVote(
            source="model",
            bout_id="bout-1",
            collegium="hiring",
            ordinal=1,
            winner="player_1",
        )


def test_bout_collegium_case_requires_matching_complete_vote_triples():
    human_votes = tuple(
        NormalizedVote(
            source="human",
            bout_id="bout-1",
            collegium="hiring",
            ordinal=ordinal,
            winner="player_1",
        )
        for ordinal in (1, 2, 3)
    )
    model_votes = tuple(
        NormalizedVote(
            source="model",
            bout_id="bout-1",
            collegium="hiring",
            ordinal=ordinal,
            winner="player_2",
            model_alias="clef",
            variant="original",
        )
        for ordinal in (1, 2, 3)
    )

    case = BoutCollegiumCase(
        bout_id="bout-1",
        collegium="hiring",
        model_alias="clef",
        variant="original",
        model_votes=model_votes,
        human_votes=human_votes,
    )

    assert len(case.model_votes) == len(case.human_votes) == 3
    with pytest.raises(ValidationError, match="do not match the case identity"):
        BoutCollegiumCase(
            bout_id="another-bout",
            collegium="hiring",
            model_alias="clef",
            variant="original",
            model_votes=model_votes,
            human_votes=human_votes,
        )


def test_scope_has_unique_bouts_and_an_explicit_denominator():
    scope = MetricScope(
        scope_id="individual-clef",
        kind="model",
        model_alias="clef",
        bout_ids=("1", "2"),
    )

    assert scope.n_bouts == 2
    with pytest.raises(ValidationError, match="must be unique"):
        MetricScope(
            scope_id="bad",
            kind="common",
            bout_ids=("1", "1"),
        )


def test_exclusion_preserves_machine_and_human_readable_reasons():
    exclusion = Exclusion(
        scope_id="individual-clef",
        bout_id="bout-1",
        model_alias="clef",
        reason_code="incomplete_original",
        reason="Only 8 of 9 original votes succeeded",
    )

    assert exclusion.reason_code == "incomplete_original"
    assert exclusion.reason.startswith("Only 8")


def test_metric_result_serializes_na_as_null_with_reason():
    result = MetricResult.not_applicable(
        "cohens_kappa",
        "only one class is present",
        n_bouts=2,
        n_cases=6,
    )

    payload = json.loads(result.model_dump_json())
    assert payload == {
        "metric": "cohens_kappa",
        "status": "not_applicable",
        "value": None,
        "n_bouts": 2,
        "n_cases": 6,
        "n_model_votes": None,
        "n_human_votes": None,
        "n_comparisons": None,
        "reason": "only one class is present",
    }


@pytest.mark.parametrize(
    "field",
    ("n_model_votes", "n_human_votes", "n_comparisons"),
)
def test_metric_result_rejects_negative_specialized_counts(field):
    with pytest.raises(ValidationError):
        MetricResult.ok(
            "agreement_rate",
            1.0,
            n_bouts=1,
            n_cases=3,
            **{field: -1},
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["--dataset", "pilot"],
        ["--run-id", "005"],
        ["--dataset", "pilot", "--run-id", ""],
        ["--dataset", "unknown", "--run-id", "005"],
    ],
)
def test_cli_requires_dataset_and_run_id(argv):
    stderr = StringIO()

    code = run(argv, error_output=stderr)

    assert code == EXIT_USAGE_ERROR
    assert "Ошибка расчёта метрик" in stderr.getvalue()


def test_smoke_fails_before_config_or_analyzer_and_creates_no_metrics(tmp_path: Path):
    called = []
    run_dir = tmp_path / "results" / "smoke" / "001"
    run_dir.mkdir(parents=True)

    code = run(
        [
            "--dataset",
            "smoke",
            "--run-id",
            "001",
            "--config",
            str(tmp_path / "missing.yaml"),
        ],
        config_loader=lambda path: called.append(("config", path)),
        analyzer=lambda request, config: called.append(("analyzer", request)),
        error_output=(stderr := StringIO()),
    )

    assert code == EXIT_USAGE_ERROR
    assert SMOKE_UNSUPPORTED_MESSAGE in stderr.getvalue()
    assert called == []
    assert not (run_dir / "metrics").exists()


@pytest.mark.parametrize("dataset", ["pilot", "main"])
def test_cli_passes_explicit_request_and_config_to_pipeline(tmp_path: Path, dataset):
    sentinel_config = object()
    seen = {}
    config_path = tmp_path / "custom.yaml"

    def config_loader(path):
        seen["loaded_path"] = path
        return sentinel_config

    def analyzer(request, config):
        seen["request"] = request
        seen["config"] = config

    code = run(
        [
            "--dataset",
            dataset,
            "--run-id",
            "005",
            "--config",
            str(config_path),
        ],
        config_loader=config_loader,
        analyzer=analyzer,
    )

    assert code == EXIT_SUCCESS
    assert seen["loaded_path"] == config_path
    assert seen["request"].dataset == dataset
    assert seen["request"].run_id == "005"
    assert seen["request"].config_path == config_path.resolve()
    assert seen["config"] is sentinel_config


def test_default_cli_reports_a_missing_explicit_run_as_technical_error(tmp_path: Path):
    config_source = Path("src/config.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_source, encoding="utf-8")
    stderr = StringIO()

    code = run(
        [
            "--dataset",
            "pilot",
            "--run-id",
            "definitely-missing",
            "--config",
            str(config_path),
        ],
        error_output=stderr,
    )

    assert code == EXIT_USAGE_ERROR
    assert "Selected dataset directory does not exist" in stderr.getvalue()
