"""Fairness-aware modeling strategies for TrustCredit.

This module implements three categories of algorithmic fairness interventions:

Pre-processing (applied before model training):
    Reweighing -- adjusts sample weights so that each demographic x outcome
    combination has its expected representation under independence.

In-processing (embedded in the training objective):
    FairGBM -- LightGBM variant that incorporates fairness constraints directly
    into the gradient boosting objective.

Post-processing (applied after model training):
    ThresholdOpt -- learns group-specific classification thresholds to satisfy
    a fairness constraint while maximizing a performance metric.

All classes follow a unified interface:
    fit(X, y, sensitive_attributes)  →  returns self
    predict(X)                        →  returns binary labels
    predict_proba(X)                  →  returns probability array
"""

from typing import Union, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, accuracy_score

from trustcredit.evaluation.metrics import demographic_parity, equal_opportunity


class FairModel(ClassifierMixin, BaseEstimator):
    """Abstract base class for all fairness-aware models in TrustCredit."""

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ):
        """Fit the model with fairness constraints.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Binary target labels.
        sensitive_attributes : array-like
            Protected attribute values (binary: 0 or 1).
        """
        raise NotImplementedError("Subclasses must implement fit().")


class PreProcessingFair(FairModel):
    """Base class for pre-processing fairness strategies.

    Pre-processing methods modify the training data (typically via sample
    reweighting) before passing it to any standard classifier.

    Parameters
    ----------
    estimator : ClassifierMixin
        Any sklearn-compatible classifier that accepts sample_weight in fit().
    """

    def __init__(self, estimator: ClassifierMixin):
        self.estimator = estimator

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> "PreProcessingFair":
        self.classes_ = np.unique(y)
        X_, y_, sample_weights = self._preprocess(X, y, sensitive_attributes)
        self.estimator.fit(X_, y_, sample_weight=sample_weights)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict binary labels.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        np.ndarray
            Binary predictions.
        """
        return self.estimator.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Class probabilities.
        """
        return self.estimator.predict_proba(X)

    def _preprocess(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> Tuple:
        """Compute sample weights or modify training data for fairness.

        Must be overridden by subclasses.
        """
        return X, y, np.ones_like(y, dtype=float)


class InProcessingFair(FairModel):
    """Base class for in-processing fairness methods."""
    pass


class PostProcessingFair(FairModel):
    """Base class for post-processing fairness methods.

    Post-processing methods fit a standard classifier first, then learn
    group-specific decision rules to enforce fairness on top of the model.

    Parameters
    ----------
    estimator : BaseEstimator
        Any sklearn-compatible classifier.
    """

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> "PostProcessingFair":
        self.classes_ = np.unique(y)
        self.estimator.fit(X, y)
        self._fit_postprocess(X, y, sensitive_attributes)
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """Predict with group-aware thresholds.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        sensitive_attributes : array-like
            Protected attribute values used to select the threshold.

        Returns
        -------
        np.ndarray
            Fairness-adjusted binary predictions.
        """
        proba = self.estimator.predict_proba(X)
        return self._apply_postprocess(proba, sensitive_attributes)

    def _fit_postprocess(self, X, y, sensitive_attributes):
        pass

    def _apply_postprocess(self, predictions, sensitive_attributes):
        return predictions


# ──────────────────────────────────────────────
# Concrete implementations
# ──────────────────────────────────────────────

class Reweighing(PreProcessingFair):
    """Reweighing pre-processing fairness method (Kamiran & Calders, 2012).

    Computes sample weights so that each combination of (sensitive group,
    label) appears with its expected frequency under statistical independence
    between the sensitive attribute and the target. This removes discriminatory
    patterns without changing the feature space.

    Weight(s, y) = P(S=s) * P(Y=y) / P(S=s, Y=y)
    """

    def _preprocess(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> Tuple:
        weights = {}
        for s_val in np.unique(sensitive_attributes):
            for y_val in np.unique(y):
                observed = np.mean((sensitive_attributes == s_val) & (y == y_val))
                expected = np.mean(sensitive_attributes == s_val) * np.mean(y == y_val)
                weights[(s_val, y_val)] = expected / observed if observed > 0 else 0.0

        weights_array = np.array([weights[(s, lbl)] for s, lbl in zip(sensitive_attributes, y)])
        return X, y, weights_array


class FairGBM(InProcessingFair):
    """In-processing fairness via FairGBM (constrained gradient boosting).

    Wraps the FairGBMClassifier which adds a fairness penalty directly to
    the LightGBM loss function during training. The constraint group
    corresponds to the sensitive attribute.

    Parameters
    ----------
    **fairgbm_params
        Any parameters accepted by fairgbm.FairGBMClassifier.
    """

    def __init__(self, **fairgbm_params):
        from fairgbm import FairGBMClassifier
        self._model = FairGBMClassifier(**fairgbm_params)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> "FairGBM":
        self.classes_ = np.unique(y)
        self._model.fit(X, y, constraint_group=sensitive_attributes)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self._model.predict_proba(X)


class ThresholdOpt(PostProcessingFair):
    """Post-processing fairness via group-specific threshold optimization.

    Searches over a grid of (threshold_group0, threshold_group1) combinations
    to find the pair that maximizes a performance metric while keeping a
    fairness metric below a specified constraint value.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted or unfitted sklearn-compatible classifier.
    perf_metric : str, optional
        Performance metric to maximize: "balanced_accuracy" or "accuracy".
        By default "balanced_accuracy".
    fair_metric : str, optional
        Fairness constraint to satisfy: "demographic_parity" or "equal_opportunity".
        By default "demographic_parity".
    constraint_value : float, optional
        Maximum allowed value for the fairness metric. By default 0.05.
    n_thresholds : int, optional
        Resolution of the threshold grid search. By default 25.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        perf_metric: str = "balanced_accuracy",
        fair_metric: str = "demographic_parity",
        constraint_value: float = 0.05,
        n_thresholds: int = 25,
    ):
        self.estimator = estimator
        self.constraint_value = constraint_value
        self.n_thresholds = n_thresholds
        self.perf_metric = balanced_accuracy_score if perf_metric == "balanced_accuracy" else accuracy_score
        self.fair_metric = demographic_parity if fair_metric == "demographic_parity" else equal_opportunity

    def _fit_postprocess(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        sensitive_attributes: Union[pd.Series, np.ndarray],
    ) -> None:
        preds = self.estimator.predict_proba(X)[:, 1]
        thresholds_g0 = np.quantile(preds, np.linspace(0, 1, self.n_thresholds))
        thresholds_g1 = np.quantile(preds, np.linspace(0, 1, self.n_thresholds))

        acc_matrix = np.zeros((self.n_thresholds, self.n_thresholds))
        fair_matrix = np.zeros((self.n_thresholds, self.n_thresholds))

        for t0, thresh_0 in enumerate(thresholds_g0):
            for t1, thresh_1 in enumerate(thresholds_g1):
                pred = np.where(sensitive_attributes == 0, preds > thresh_0, preds > thresh_1)
                acc_matrix[t0, t1] = self.perf_metric(y, pred)
                fair_matrix[t0, t1] = self.fair_metric(y, pred, sensitive_attributes)

        acc_matrix[fair_matrix > self.constraint_value] = 0.0
        best = np.unravel_index(np.argmax(acc_matrix), acc_matrix.shape)
        self.thresh0_ = thresholds_g0[best[0]]
        self.thresh1_ = thresholds_g1[best[1]]

    def _apply_postprocess(
        self,
        predictions: np.ndarray,
        sensitive_attributes: np.ndarray,
    ) -> np.ndarray:
        return np.where(
            sensitive_attributes == 0,
            predictions[:, 1] > self.thresh0_,
            predictions[:, 1] > self.thresh1_,
        )
