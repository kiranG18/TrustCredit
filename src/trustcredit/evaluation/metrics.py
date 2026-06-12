"""
Performance and fairness evaluation metrics for credit scoring models.

Performance metrics:
    get_metrics() -- AUC, Brier, Balanced Accuracy, Precision, Recall, F1

Fairness metrics:
    demographic_parity     -- difference in positive prediction rates across groups
    equal_opportunity      -- difference in true positive rates across groups
    average_odds           -- average of TPR and FPR differences
    geometric_mean_accuracy -- geometric mean of per-group accuracy

Reject inference evaluation:
    get_reject_inference_metrics() -- approval rate and kickout metric
    kickout()                       -- measures improvement on borderline cases
"""

from typing import Dict, Tuple, Any, Union, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)


# ──────────────────────────────────────────────
# Fairness metrics
# ──────────────────────────────────────────────

def false_positive_rate(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> float:
    """Compute the false positive rate (FPR).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.

    Returns
    -------
    float
        FPR = FP / (FP + TN).
    """
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def demographic_parity(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    z: Union[np.ndarray, pd.Series],
) -> float:
    """Compute the Demographic Parity Difference (DPD).

    Measures the difference in positive prediction rates between the
    privileged group (z=1) and the unprivileged group (z=0). A value
    near zero indicates parity.

    Parameters
    ----------
    y_true : array-like
        True labels (unused in calculation, kept for API consistency).
    y_pred : array-like
        Predicted binary labels.
    z : array-like
        Binary sensitive attribute (0 = unprivileged, 1 = privileged).

    Returns
    -------
    float
        P(Y_hat=1 | Z=1) - P(Y_hat=1 | Z=0)
    """
    return float(np.mean(y_pred[z == 1]) - np.mean(y_pred[z == 0]))


def equal_opportunity(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    z: Union[np.ndarray, pd.Series],
) -> float:
    """Compute the Equal Opportunity Difference (EOD).

    Measures the difference in true positive rates (recall) between
    the privileged and unprivileged groups. A value near zero means
    both groups receive the positive outcome at similar rates when
    they truly qualify.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    z : array-like
        Binary sensitive attribute.

    Returns
    -------
    float
        TPR(Z=1) - TPR(Z=0)
    """
    tpr_1 = recall_score(y_true[z == 1], y_pred[z == 1], zero_division=0)
    tpr_0 = recall_score(y_true[z == 0], y_pred[z == 0], zero_division=0)
    return float(tpr_1 - tpr_0)


def average_odds(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    z: Union[np.ndarray, pd.Series],
) -> float:
    """Compute the Average Odds Difference (AOD).

    The arithmetic mean of the Equal Opportunity Difference and the
    difference in false positive rates between groups.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    z : array-like
        Binary sensitive attribute.

    Returns
    -------
    float
        0.5 * (EOD + FPR_diff)
    """
    eod = equal_opportunity(y_true, y_pred, z)
    fpr_diff = (
        false_positive_rate(y_true[z == 1], y_pred[z == 1])
        - false_positive_rate(y_true[z == 0], y_pred[z == 0])
    )
    return 0.5 * (eod + fpr_diff)


def average_precision_value_difference(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    z: Union[np.ndarray, pd.Series],
) -> float:
    """Compute the Average Precision Value Difference (APVD).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    z : array-like
        Binary sensitive attribute.

    Returns
    -------
    float
        Average odds computed with y_true and y_pred swapped.
    """
    return average_odds(y_pred, y_true, z)


def geometric_mean_accuracy(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    z: Union[np.ndarray, pd.Series],
) -> float:
    """Compute the Geometric Mean Accuracy (GMA) across groups.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    z : array-like
        Binary sensitive attribute.

    Returns
    -------
    float
        sqrt(Acc(Z=1) * Acc(Z=0))
    """
    acc_1 = accuracy_score(y_true[z == 1], y_pred[z == 1])
    acc_0 = accuracy_score(y_true[z == 0], y_pred[z == 0])
    return float(np.sqrt(acc_1 * acc_0))


# ──────────────────────────────────────────────
# Scorer factories for hyperparameter tuning
# ──────────────────────────────────────────────

def create_eod_scorer(
    z: Union[np.ndarray, pd.Series],
    benefit_class: int = 1,
):
    """Create an Equal Opportunity Difference scorer for model selection.

    Parameters
    ----------
    z : array-like
        Sensitive attribute, aligned with the dataset's row order.
    benefit_class : int, optional
        Which label is the "benefit" class. By default 1.

    Returns
    -------
    callable
        Scorer function with signature (y_true, y_pred) -> float.
    """
    def eod_scorer(y_true, y_pred):
        y_t = (y_true == benefit_class).astype(float)
        y_p = (y_pred == benefit_class).astype(float)
        return equal_opportunity(y_t, y_p, z)
    return eod_scorer


def create_fairness_scorer(
    fairness_goal: float,
    z: Union[np.ndarray, pd.Series],
    M: int = 10,
    benefit_class: int = 1,
):
    """Create a penalized AUC scorer that enforces a fairness constraint.

    Returns full AUC if the fairness goal is achieved; otherwise applies
    a linear penalty proportional to the fairness violation.

    Parameters
    ----------
    fairness_goal : float
        Maximum acceptable EOD value.
    z : array-like
        Sensitive attribute.
    M : int, optional
        Penalty magnitude per unit of fairness violation. By default 10.
    benefit_class : int, optional
        Positive class label. By default 1.

    Returns
    -------
    callable
        Scorer with signature (y_true, y_pred, y_score) -> float.
    """
    def fairness_scorer(y_true, y_pred, y_score):
        y_t = (y_true == benefit_class).astype(float)
        y_p = (y_pred == benefit_class).astype(float)
        eod = abs(equal_opportunity(y_t, y_p, z))
        auc = roc_auc_score(y_true, y_score)
        if eod <= fairness_goal:
            return auc
        return auc - M * (eod - fairness_goal)
    return fairness_scorer


# ──────────────────────────────────────────────
# Main evaluation functions
# ──────────────────────────────────────────────

def get_metrics(
    name_model_dict: Dict[str, Any],
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate multiple models and return a performance metrics table.

    Parameters
    ----------
    name_model_dict : dict
        Keys are model names. Values are either a fitted model, or a
        list [model, threshold] to override the default threshold.
    X : array-like
        Feature matrix for evaluation.
    y : array-like
        True binary labels.
    threshold : float, optional
        Default classification threshold. By default 0.5.

    Returns
    -------
    pd.DataFrame
        Rows are models, columns are metrics:
        AUC, Brier Score, Balanced Accuracy, Accuracy, Precision, Recall, F1.
    """
    predictions = {}
    for name, model in name_model_dict.items():
        if isinstance(model, list):
            model_, thresh = model[0], model[1]
        else:
            model_, thresh = model, threshold

        if hasattr(model_, "predict_proba"):
            y_prob = model_.predict_proba(X)[:, 1]
            y_pred = (y_prob >= thresh).astype(int)
        else:
            y_prob = None
            y_pred = model_.predict(X)
        predictions[name] = (y_pred, y_prob)

    rows = []
    for name, (y_pred, y_prob) in predictions.items():
        row = {"model": name}
        row["AUC"] = roc_auc_score(y, y_prob) if y_prob is not None else None
        row["Brier Score"] = brier_score_loss(y, y_prob) if y_prob is not None else None
        row["Balanced Accuracy"] = balanced_accuracy_score(y, y_pred)
        row["Accuracy"] = accuracy_score(y, y_pred)
        row["Precision"] = precision_score(y, y_pred, zero_division=0)
        row["Recall"] = recall_score(y, y_pred, zero_division=0)
        row["F1"] = f1_score(y, y_pred, zero_division=0)
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def get_fairness_metrics(
    name_model_dict: Dict[str, Any],
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    z: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
    benefit_class: int = 1,
) -> pd.DataFrame:
    """Compute fairness metrics for multiple models.

    Parameters
    ----------
    name_model_dict : dict
        Model name -> fitted model (or list [model, threshold]).
    X : array-like
        Feature matrix.
    y : array-like
        True labels.
    z : array-like
        Binary sensitive attribute.
    threshold : float, optional
        Classification threshold. By default 0.5.
    benefit_class : int, optional
        Which label is the positive/benefit class. By default 1.

    Returns
    -------
    pd.DataFrame
        Rows are models, columns are fairness metrics:
        DPD, EOD, AOD, APVD, GMA, Balanced Accuracy.
    """
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(z, pd.Series):
        z = z.values

    y_true = (y == benefit_class).astype(int)

    rows = []
    for name, model in name_model_dict.items():
        if isinstance(model, list):
            model_, thresh = model[0], model[1]
        else:
            model_, thresh = model, threshold

        y_prob = model_.predict_proba(X)[:, 1]
        y_pred = (y_prob >= thresh).astype(int)
        y_pred_bin = (y_pred == benefit_class).astype(int)

        row = {
            "model": name,
            "DPD": demographic_parity(y_true, y_pred_bin, z),
            "EOD": equal_opportunity(y_true, y_pred_bin, z),
            "AOD": average_odds(y_true, y_pred_bin, z),
            "APVD": average_precision_value_difference(y_true, y_pred_bin, z),
            "GMA": geometric_mean_accuracy(y_true, y_pred_bin, z),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred_bin),
        }
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def kickout(
    y_true: Union[np.ndarray, pd.Series],
    y_pred_base: Union[np.ndarray, pd.Series],
    y_pred_reject: Union[np.ndarray, pd.Series],
) -> float:
    """Compute the kickout metric for reject inference evaluation.

    Measures how well a reject inference model corrects the base model's
    errors on actual defaulters (true positives missed by base model)
    relative to how much it incorrectly rejects good payers.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred_base : array-like
        Predictions from the base model.
    y_pred_reject : array-like
        Predictions from the reject inference model.

    Returns
    -------
    float
        Kickout = P(reject_correct | base_missed_bad) - P(reject_wrong | base_correct_good)
    """
    wrong_base = (y_true == 1) & (y_pred_base == 0)
    correct_base = (y_true == 0) & (y_pred_base == 0)
    kb = np.mean(y_pred_reject[wrong_base]) if wrong_base.sum() > 0 else 0.0
    kg = np.mean(y_pred_reject[correct_base]) if correct_base.sum() > 0 else 0.0
    return float(kb - kg)


def get_threshold_bad_rate(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    bad_rate: float = 0.25,
) -> float:
    """Find the probability threshold that keeps the false positive rate below bad_rate.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_probs : np.ndarray
        Predicted probabilities for the positive class.
    bad_rate : float, optional
        Maximum tolerable FPR. By default 0.25.

    Returns
    -------
    float
        Threshold value, defaulting to 0.5 if no valid threshold exists.
    """
    sorted_indices = np.argsort(y_probs)[::-1]
    y_true_sorted = y_true[sorted_indices]

    total_neg = np.sum(1 - y_true_sorted)
    if total_neg == 0:
        return 0.5

    fp = np.cumsum(1 - y_true_sorted)
    fpr = fp / total_neg
    valid = fpr <= bad_rate

    if not valid.any():
        return 0.5

    max_idx = np.max(np.where(valid)[0])
    return float(y_probs[sorted_indices[max_idx]])


def get_reject_inference_metrics(
    name_model_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    bad_rate: float = 0.5,
) -> pd.DataFrame:
    """Evaluate reject inference models using approval rate and kickout.

    Parameters
    ----------
    name_model_dict : dict
        Model name -> (y_probs_labeled, y_probs_unlabeled).
        Must include a "base" key for the baseline model.
    y : np.ndarray
        True labels for the labeled population.
    bad_rate : float, optional
        FPR constraint for threshold selection. By default 0.5.

    Returns
    -------
    pd.DataFrame
        Columns: model, approval_rate, balanced_accuracy, kickout.
    """
    assert "base" in name_model_dict, "Must include a 'base' key for the baseline model."

    y_probs_base, _ = name_model_dict["base"]
    base_threshold = get_threshold_bad_rate(y, y_probs_base, bad_rate=bad_rate)
    y_pred_base = (y_probs_base >= base_threshold).astype(int)

    rows = []
    for name, (y_probs, y_probs_unl) in name_model_dict.items():
        threshold = get_threshold_bad_rate(y, y_probs, bad_rate=bad_rate)
        y_pred = (y_probs >= threshold).astype(int)
        y_pred_unl = (y_probs_unl >= threshold).astype(int)
        rows.append({
            "model": name,
            "approval_rate": np.mean(np.concatenate([y_pred, y_pred_unl])),
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "kickout": kickout(y, y_pred_base, y_pred),
        })

    return pd.DataFrame(rows).set_index("model")
