"""Public constants for the offline metrics calculation pipeline."""

from __future__ import annotations

from typing import Final, Literal

from src.experiment.schemas import Collegium


# Increment this version whenever a change can alter reported numeric values,
# eligibility scopes, denominators, confidence intervals, or N/A semantics.
CALCULATION_RULES_VERSION: Final = "1.0.0"

COLLEGIA: Final[tuple[Collegium, ...]] = (
    "hiring",
    "negotiation",
    "ownership",
)
BOOTSTRAP_REPLICATIONS: Final = 10_000
BOOTSTRAP_SEED: Final = 42
CONFIDENCE_LEVEL: Final = 0.95
PERCENTILE_BOUNDS: Final[tuple[float, float]] = (2.5, 97.5)

# Hyndman-Fan type 7: linear interpolation between adjacent order statistics,
# equivalent to NumPy's ``method="linear"`` convention.
PERCENTILE_METHOD: Final[Literal["linear_type_7"]] = "linear_type_7"


__all__ = [
    "BOOTSTRAP_REPLICATIONS",
    "BOOTSTRAP_SEED",
    "CALCULATION_RULES_VERSION",
    "COLLEGIA",
    "CONFIDENCE_LEVEL",
    "PERCENTILE_BOUNDS",
    "PERCENTILE_METHOD",
]
