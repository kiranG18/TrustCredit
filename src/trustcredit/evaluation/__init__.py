"""Evaluation module for TrustCredit."""

from .metrics import (
    get_metrics,
    get_fairness_metrics,
    get_reject_inference_metrics,
    demographic_parity,
    equal_opportunity,
    average_odds,
    average_precision_value_difference,
    geometric_mean_accuracy,
    kickout,
    get_threshold_bad_rate,
    create_eod_scorer,
    create_fairness_scorer,
)

# Re-export ks_threshold for backward compatibility
from trustcredit.models.tuning import ks_threshold

__all__ = [
    "get_metrics",
    "get_fairness_metrics",
    "get_reject_inference_metrics",
    "demographic_parity",
    "equal_opportunity",
    "average_odds",
    "average_precision_value_difference",
    "geometric_mean_accuracy",
    "kickout",
    "ks_threshold",
    "get_threshold_bad_rate",
    "create_eod_scorer",
    "create_fairness_scorer",
]
