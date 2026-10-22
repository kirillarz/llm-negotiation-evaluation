"""Pure metric calculations over stage-four case aggregates.

Eligibility is decided by :mod:`src.metrics.preparation`.  This module only
filters already eligible aggregates to an explicitly supplied bout scope and
never treats model repeats or human judges as independent cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import fmean
from typing import Iterable, Sequence

from src.experiment.schemas import Collegium, Player
from src.metrics import COLLEGIA
from src.metrics.aggregation import AggregatedAnalysis, AggregatedCase
from src.metrics.contracts import MetricResult
from src.metrics.preparation import PreparedAnalysis


EMPTY_SCOPE_REASON = "No eligible bout-collegium cases are available"
UNDEFINED_COHEN_REASON = "Cohen's kappa is undefined because expected agreement is 1"
UNDEFINED_FLEISS_REASON = "Fleiss' kappa is undefined because expected agreement is 1"
MISSING_HUMAN_CLASS_REASON = (
    "Macro-F1 is undefined because at least one class is absent from human "
    "majority labels"
)
NO_PROBABILITIES_REASON = (
    "No eligible bout-collegium cases contain every planned provider probability"
)
NO_REPEATS_REASON = "Repeatability requires at least two planned model votes per case"


@dataclass(frozen=True, slots=True)
class AgreementMetrics:
    accuracy: MetricResult
    by_collegium: dict[Collegium, MetricResult]
    cohens_kappa: MetricResult
    macro_f1: MetricResult


@dataclass(frozen=True, slots=True)
class IndividualJudgeMetrics:
    majority_to_individual: MetricResult
    all_pairs: MetricResult


@dataclass(frozen=True, slots=True)
class HumanAgreementMetrics:
    pairwise_agreement: MetricResult
    fleiss_kappa: MetricResult


@dataclass(frozen=True, slots=True)
class RepeatabilityMetrics:
    unanimity_rate: MetricResult
    pairwise_repeatability: MetricResult


@dataclass(frozen=True, slots=True)
class PositionBiasMetrics:
    label_flip_rate: MetricResult
    vote_share_change: MetricResult
    by_collegium: dict[Collegium, tuple[MetricResult, MetricResult]]


@dataclass(frozen=True, slots=True)
class ProbabilityMetrics:
    brier_score: MetricResult
    brier_score_individual: MetricResult
    vote_share_mae: MetricResult
    vote_share_mae_individual: MetricResult
    coverage: MetricResult


@dataclass(frozen=True, slots=True)
class ModelScopeMetrics:
    model_alias: str
    scope_id: str
    bout_ids: tuple[str, ...]
    agreement: AgreementMetrics
    individual_judge: IndividualJudgeMetrics
    repeatability: RepeatabilityMetrics
    probability: ProbabilityMetrics


@dataclass(frozen=True, slots=True)
class PositionScopeMetrics:
    model_alias: str
    scope_id: str
    bout_ids: tuple[str, ...]
    metrics: PositionBiasMetrics


@dataclass(frozen=True, slots=True)
class HumanScopeMetrics:
    scope_id: str
    bout_ids: tuple[str, ...]
    metrics: HumanAgreementMetrics


@dataclass(frozen=True, slots=True)
class CalculatedMetrics:
    model_scopes: tuple[ModelScopeMetrics, ...]
    position_scopes: tuple[PositionScopeMetrics, ...]
    human_scopes: tuple[HumanScopeMetrics, ...]


def _counts(cases: Sequence[AggregatedCase]) -> tuple[int, int]:
    return len({case.bout_id for case in cases}), len(cases)


def _model_vote_count(
    cases: Sequence[AggregatedCase], *, variant: str = "original"
) -> int:
    return sum(
        len(case.original_votes if variant == "original" else case.swapped_votes)
        for case in cases
    )


def _na(
    metric: str,
    cases: Sequence[AggregatedCase],
    reason: str,
    *,
    n_model_votes: int | None = None,
    n_human_votes: int | None = None,
    n_comparisons: int | None = None,
) -> MetricResult:
    n_bouts, n_cases = _counts(cases)
    return MetricResult.not_applicable(
        metric,
        reason,
        n_bouts=n_bouts,
        n_cases=n_cases,
        n_model_votes=n_model_votes,
        n_human_votes=n_human_votes,
        n_comparisons=n_comparisons,
    )


def _mean_result(
    metric: str,
    cases: Sequence[AggregatedCase],
    values: Iterable[float],
    *,
    n_model_votes: int | None = None,
    n_human_votes: int | None = None,
    n_comparisons: int | None = None,
) -> MetricResult:
    n_bouts, n_cases = _counts(cases)
    if not cases:
        return MetricResult.not_applicable(
            metric,
            EMPTY_SCOPE_REASON,
            n_bouts=n_bouts,
            n_cases=n_cases,
            n_model_votes=n_model_votes,
            n_human_votes=n_human_votes,
            n_comparisons=n_comparisons,
        )
    return MetricResult.ok(
        metric,
        fmean(values),
        n_bouts=n_bouts,
        n_cases=n_cases,
        n_model_votes=n_model_votes,
        n_human_votes=n_human_votes,
        n_comparisons=n_comparisons,
    )


def _majorities(case: AggregatedCase) -> tuple[Player, Player]:
    model = case.original_tally.majority
    human = case.human_tally.majority
    if model is None or human is None or not case.original_complete:
        raise ValueError("metrics require complete eligible original cases")
    return model, human


def agreement_rate(cases: Sequence[AggregatedCase]) -> MetricResult:
    """Agreement of model and human majorities at case level."""

    return _mean_result(
        "agreement_rate",
        cases,
        (_majorities(case)[0] == _majorities(case)[1] for case in cases),
        n_model_votes=_model_vote_count(cases),
        n_human_votes=3 * len(cases),
        n_comparisons=len(cases),
    )


def agreement_by_collegium(
    cases: Sequence[AggregatedCase],
) -> dict[Collegium, MetricResult]:
    return {
        collegium: agreement_rate(
            tuple(case for case in cases if case.collegium == collegium)
        ).model_copy(update={"metric": f"agreement_rate_{collegium}"})
        for collegium in COLLEGIA
    }


def cohens_kappa(cases: Sequence[AggregatedCase]) -> MetricResult:
    if not cases:
        return _na(
            "cohens_kappa",
            cases,
            EMPTY_SCOPE_REASON,
            n_model_votes=0,
            n_human_votes=0,
            n_comparisons=0,
        )
    labels = [_majorities(case) for case in cases]
    observed = fmean(model == human for model, human in labels)
    model_p1 = fmean(model == "player_1" for model, _ in labels)
    human_p1 = fmean(human == "player_1" for _, human in labels)
    expected = model_p1 * human_p1 + (1 - model_p1) * (1 - human_p1)
    denominator = 1 - expected
    if denominator == 0:
        return _na(
            "cohens_kappa",
            cases,
            UNDEFINED_COHEN_REASON,
            n_model_votes=_model_vote_count(cases),
            n_human_votes=3 * len(cases),
            n_comparisons=len(cases),
        )
    n_bouts, n_cases = _counts(cases)
    return MetricResult.ok(
        "cohens_kappa",
        (observed - expected) / denominator,
        n_bouts=n_bouts,
        n_cases=n_cases,
        n_model_votes=_model_vote_count(cases),
        n_human_votes=3 * n_cases,
        n_comparisons=n_cases,
    )


def macro_f1(cases: Sequence[AggregatedCase]) -> MetricResult:
    if not cases:
        return _na(
            "macro_f1",
            cases,
            EMPTY_SCOPE_REASON,
            n_model_votes=0,
            n_human_votes=0,
            n_comparisons=0,
        )
    labels = [_majorities(case) for case in cases]
    human_labels = {human for _, human in labels}
    if human_labels != {"player_1", "player_2"}:
        return _na(
            "macro_f1",
            cases,
            MISSING_HUMAN_CLASS_REASON,
            n_model_votes=_model_vote_count(cases),
            n_human_votes=3 * len(cases),
            n_comparisons=len(cases),
        )

    scores = []
    for label in ("player_1", "player_2"):
        true_positive = sum(
            model == label and human == label for model, human in labels
        )
        false_positive = sum(
            model == label and human != label for model, human in labels
        )
        false_negative = sum(
            model != label and human == label for model, human in labels
        )
        denominator = 2 * true_positive + false_positive + false_negative
        if denominator == 0:
            return _na(
                "macro_f1",
                cases,
                f"F1 is undefined for class {label}",
                n_model_votes=_model_vote_count(cases),
                n_human_votes=3 * len(cases),
                n_comparisons=len(cases),
            )
        scores.append(2 * true_positive / denominator)
    n_bouts, n_cases = _counts(cases)
    return MetricResult.ok(
        "macro_f1",
        fmean(scores),
        n_bouts=n_bouts,
        n_cases=n_cases,
        n_model_votes=_model_vote_count(cases),
        n_human_votes=3 * n_cases,
        n_comparisons=n_cases,
    )


def agreement_metrics(cases: Sequence[AggregatedCase]) -> AgreementMetrics:
    return AgreementMetrics(
        accuracy=agreement_rate(cases),
        by_collegium=agreement_by_collegium(cases),
        cohens_kappa=cohens_kappa(cases),
        macro_f1=macro_f1(cases),
    )


def individual_judge_agreement(
    cases: Sequence[AggregatedCase],
) -> IndividualJudgeMetrics:
    for case in cases:
        _majorities(case)
        if len(case.human_votes) != 3 or not case.original_votes:
            raise ValueError("individual agreement requires complete model and human votes")
    model_vote_count = _model_vote_count(cases)
    human_vote_count = 3 * len(cases)
    majority = _mean_result(
        "individual_judge_majority_to_individual",
        cases,
        (
            sum(
                vote.winner == case.original_tally.majority
                for vote in case.human_votes
            )
            / 3
            for case in cases
        ),
        n_model_votes=model_vote_count,
        n_human_votes=human_vote_count,
        n_comparisons=3 * len(cases),
    )
    all_pairs = _mean_result(
        "individual_judge_all_pairs",
        cases,
        (
            sum(
                model.winner == human.winner
                for model in case.original_votes
                for human in case.human_votes
            )
            / (len(case.original_votes) * len(case.human_votes))
            for case in cases
        ),
        n_model_votes=model_vote_count,
        n_human_votes=human_vote_count,
        n_comparisons=sum(
            len(case.original_votes) * len(case.human_votes) for case in cases
        ),
    )
    return IndividualJudgeMetrics(majority, all_pairs)


def _unique_human_cases(
    cases: Sequence[AggregatedCase],
) -> tuple[AggregatedCase, ...]:
    unique: dict[tuple[str, Collegium], AggregatedCase] = {}
    for case in cases:
        key = (case.bout_id, case.collegium)
        previous = unique.setdefault(key, case)
        if tuple(v.winner for v in previous.human_votes) != tuple(
            v.winner for v in case.human_votes
        ):
            raise ValueError("human votes disagree between model aggregates")
    return tuple(unique.values())


def human_human_agreement(
    cases: Sequence[AggregatedCase],
) -> HumanAgreementMetrics:
    cases = _unique_human_cases(cases)
    for case in cases:
        if len(case.human_votes) != 3:
            raise ValueError("human agreement requires complete vote triples")
    pairwise = _mean_result(
        "human_pairwise_agreement",
        cases,
        (
            sum(
                left.winner == right.winner
                for left, right in combinations(case.human_votes, 2)
            )
            / 3
            for case in cases
        ),
        n_human_votes=3 * len(cases),
        n_comparisons=3 * len(cases),
    )
    if not cases:
        fleiss = _na(
            "fleiss_kappa",
            cases,
            EMPTY_SCOPE_REASON,
            n_human_votes=0,
            n_comparisons=0,
        )
    else:
        n_raters = 3
        observed = fmean(
            (
                case.human_tally.player_1**2
                + case.human_tally.player_2**2
                - n_raters
            )
            / (n_raters * (n_raters - 1))
            for case in cases
        )
        total_votes = n_raters * len(cases)
        p1 = sum(case.human_tally.player_1 for case in cases) / total_votes
        expected = p1**2 + (1 - p1) ** 2
        denominator = 1 - expected
        if denominator == 0:
            fleiss = _na(
                "fleiss_kappa",
                cases,
                UNDEFINED_FLEISS_REASON,
                n_human_votes=total_votes,
                n_comparisons=3 * len(cases),
            )
        else:
            n_bouts, n_cases = _counts(cases)
            fleiss = MetricResult.ok(
                "fleiss_kappa",
                (observed - expected) / denominator,
                n_bouts=n_bouts,
                n_cases=n_cases,
                n_human_votes=total_votes,
                n_comparisons=3 * n_cases,
            )
    return HumanAgreementMetrics(pairwise, fleiss)


def repeatability(
    cases: Sequence[AggregatedCase], *, variant: str = "original"
) -> RepeatabilityMetrics:
    if variant not in {"original", "swapped"}:
        raise ValueError("variant must be 'original' or 'swapped'")

    def votes_for(case: AggregatedCase):
        votes = case.original_votes if variant == "original" else case.swapped_votes
        complete = (
            case.original_complete
            if variant == "original"
            else case.swapped_complete
        )
        if not complete or not votes:
            raise ValueError(f"repeatability requires complete {variant} votes")
        return votes

    vote_groups = [votes_for(case) for case in cases]
    suffix = "" if variant == "original" else "_swapped"
    n_model_votes = sum(len(votes) for votes in vote_groups)
    if vote_groups and any(len(votes) < 2 for votes in vote_groups):
        return RepeatabilityMetrics(
            _na(
                f"unanimity_rate{suffix}", cases, NO_REPEATS_REASON,
                n_model_votes=n_model_votes, n_comparisons=0,
            ),
            _na(
                f"pairwise_repeatability{suffix}", cases, NO_REPEATS_REASON,
                n_model_votes=n_model_votes, n_comparisons=0,
            ),
        )
    comparison_count = sum(
        len(votes) * (len(votes) - 1) // 2 for votes in vote_groups
    )
    unanimity = _mean_result(
        f"unanimity_rate{suffix}",
        cases,
        (len({vote.winner for vote in votes}) == 1 for votes in vote_groups),
        n_model_votes=n_model_votes,
    )
    pairwise = _mean_result(
        f"pairwise_repeatability{suffix}",
        cases,
        (
            sum(left.winner == right.winner for left, right in combinations(votes, 2))
            / (len(votes) * (len(votes) - 1) / 2)
            for votes in vote_groups
        ),
        n_model_votes=n_model_votes,
        n_comparisons=comparison_count,
    )
    return RepeatabilityMetrics(unanimity, pairwise)


def _position_pair(
    cases: Sequence[AggregatedCase], metric_suffix: str = ""
) -> tuple[MetricResult, MetricResult]:
    for case in cases:
        if (
            not case.position_eligible
            or case.position_winner_changed is None
            or case.position_vote_share_change is None
        ):
            raise ValueError("position metrics require position-eligible cases")
    flips = _mean_result(
        f"label_flip_rate{metric_suffix}",
        cases,
        (case.position_winner_changed for case in cases),
        n_model_votes=_model_vote_count(cases) + _model_vote_count(cases, variant="swapped"),
        n_comparisons=len(cases),
    )
    shares = _mean_result(
        f"vote_share_change{metric_suffix}",
        cases,
        (case.position_vote_share_change for case in cases),
        n_model_votes=_model_vote_count(cases) + _model_vote_count(cases, variant="swapped"),
        n_comparisons=len(cases),
    )
    return flips, shares


def position_bias(cases: Sequence[AggregatedCase]) -> PositionBiasMetrics:
    flips, shares = _position_pair(cases)
    by_collegium = {
        collegium: _position_pair(
            tuple(case for case in cases if case.collegium == collegium),
            f"_{collegium}",
        )
        for collegium in COLLEGIA
    }
    return PositionBiasMetrics(flips, shares, by_collegium)


def probability_metrics(cases: Sequence[AggregatedCase]) -> ProbabilityMetrics:
    for case in cases:
        _majorities(case)
    eligible = tuple(
        case
        for case in cases
        if case.original_probabilities
        and all(value is not None for value in case.original_probabilities)
    )
    n_bouts, n_cases = _counts(cases)
    if cases:
        coverage = MetricResult.ok(
            "probability_coverage",
            len(eligible) / len(cases),
            n_bouts=n_bouts,
            n_cases=n_cases,
            n_model_votes=_model_vote_count(eligible),
        )
    else:
        coverage = MetricResult.not_applicable(
            "probability_coverage",
            EMPTY_SCOPE_REASON,
            n_bouts=0,
            n_cases=0,
            n_model_votes=0,
        )

    if not eligible:
        def unavailable(metric: str) -> MetricResult:
            return MetricResult.not_applicable(
                metric,
                NO_PROBABILITIES_REASON,
                n_bouts=0,
                n_cases=0,
                n_model_votes=0,
                n_human_votes=0,
                n_comparisons=0,
            )

        return ProbabilityMetrics(
            unavailable("brier_score"),
            unavailable("brier_score_individual"),
            unavailable("vote_share_mae"),
            unavailable("vote_share_mae_individual"),
            coverage,
        )

    def probabilities(case: AggregatedCase) -> tuple[float, ...]:
        values = case.original_probabilities
        assert values and all(value is not None for value in values)
        return values  # type: ignore[return-value]

    def target(case: AggregatedCase) -> int:
        return int(case.human_tally.majority == "player_1")

    def human_share(case: AggregatedCase) -> float:
        share = case.human_tally.player_1_share
        assert share is not None
        return share

    def metric(
        name: str, values: Iterable[float], *, individual: bool = False
    ) -> MetricResult:
        return _mean_result(
            name,
            eligible,
            values,
            n_model_votes=_model_vote_count(eligible),
            n_human_votes=3 * len(eligible),
            n_comparisons=(
                _model_vote_count(eligible) if individual else len(eligible)
            ),
        )

    return ProbabilityMetrics(
        metric(
            "brier_score",
            ((fmean(probabilities(case)) - target(case)) ** 2 for case in eligible),
        ),
        metric(
            "brier_score_individual",
            (
                fmean((value - target(case)) ** 2 for value in probabilities(case))
                for case in eligible
            ),
            individual=True,
        ),
        metric(
            "vote_share_mae",
            (
                abs(fmean(probabilities(case)) - human_share(case))
                for case in eligible
            ),
        ),
        metric(
            "vote_share_mae_individual",
            (
                fmean(abs(value - human_share(case)) for value in probabilities(case))
                for case in eligible
            ),
            individual=True,
        ),
        coverage,
    )


def _select(
    cases: Sequence[AggregatedCase],
    *,
    model_alias: str | None = None,
    bout_ids: Sequence[str],
) -> tuple[AggregatedCase, ...]:
    allowed = set(bout_ids)
    return tuple(
        case
        for case in cases
        if case.bout_id in allowed
        and (model_alias is None or case.model_alias == model_alias)
    )


def calculate_metrics(
    prepared: PreparedAnalysis, aggregated: AggregatedAnalysis
) -> CalculatedMetrics:
    """Calculate every stage-five metric on all mandated scopes."""

    model_results: list[ModelScopeMetrics] = []
    position_results: list[PositionScopeMetrics] = []
    for model_alias in prepared.model_aliases:
        for scope in (prepared.model_scope(model_alias), prepared.common_scope):
            cases = _select(
                aggregated.original_cases,
                model_alias=model_alias,
                bout_ids=scope.bout_ids,
            )
            scope_id = (
                scope.scope_id
                if scope.kind == "model"
                else f"common:{model_alias}"
            )
            model_results.append(
                ModelScopeMetrics(
                    model_alias=model_alias,
                    scope_id=scope_id,
                    bout_ids=scope.bout_ids,
                    agreement=agreement_metrics(cases),
                    individual_judge=individual_judge_agreement(cases),
                    repeatability=repeatability(cases),
                    probability=probability_metrics(cases),
                )
            )
        for scope in (
            prepared.position_scope(model_alias),
            prepared.position_common_scope,
        ):
            cases = _select(
                aggregated.position_cases,
                model_alias=model_alias,
                bout_ids=scope.bout_ids,
            )
            scope_id = (
                scope.scope_id
                if scope.kind == "position"
                else f"position:common:{model_alias}"
            )
            position_results.append(
                PositionScopeMetrics(
                    model_alias=model_alias,
                    scope_id=scope_id,
                    bout_ids=scope.bout_ids,
                    metrics=position_bias(cases),
                )
            )

    all_human_cases = _unique_human_cases(aggregated.case_diagnostics)
    human_scope_specs = [("all_humans", prepared.selected_bout_ids)]
    human_scope_specs.extend(
        (f"humans:model:{model_alias}", prepared.model_scope(model_alias).bout_ids)
        for model_alias in prepared.model_aliases
    )
    if len(prepared.model_aliases) > 1:
        human_scope_specs.append(("humans:common", prepared.common_scope.bout_ids))
    human_results = tuple(
        HumanScopeMetrics(
            scope_id=scope_id,
            bout_ids=tuple(bout_ids),
            metrics=human_human_agreement(
                _select(all_human_cases, bout_ids=bout_ids)
            ),
        )
        for scope_id, bout_ids in human_scope_specs
    )
    return CalculatedMetrics(
        model_scopes=tuple(model_results),
        position_scopes=tuple(position_results),
        human_scopes=human_results,
    )


__all__ = [
    "AgreementMetrics",
    "CalculatedMetrics",
    "HumanAgreementMetrics",
    "HumanScopeMetrics",
    "IndividualJudgeMetrics",
    "ModelScopeMetrics",
    "PositionBiasMetrics",
    "PositionScopeMetrics",
    "ProbabilityMetrics",
    "RepeatabilityMetrics",
    "agreement_by_collegium",
    "agreement_metrics",
    "agreement_rate",
    "calculate_metrics",
    "cohens_kappa",
    "human_human_agreement",
    "individual_judge_agreement",
    "macro_f1",
    "position_bias",
    "probability_metrics",
    "repeatability",
]
