"""
Reject Inference methods for credit scoring pipelines.

When a bank only has outcome labels (default/no-default) for approved
applicants, any model trained on this data suffers from selection bias.
Reject inference is the set of techniques that attempt to infer what
would have happened to rejected applicants, and use that information
to train less biased models.

This module implements six reject inference strategies:

- RejectUpward: up-weights accepted samples that look like rejects
- RejectDownward: down-weights accepted samples that look easily accepted
- RejectSoftCutoff: bin-based reweighting of accepted samples
- FuzzyParcelling: duplicates rejected samples with soft labels
- RejectExtrapolation: hard-labels rejects and adds them to training data
- RejectSpreading: semi-supervised label propagation via graph spreading

All classes follow the sklearn API. The fit() method expects a feature matrix
where rejected (unlabeled) samples are marked with -1 in the target vector y.
"""

from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.semi_supervised import LabelSpreading


class RejectInference(ClassifierMixin, BaseEstimator):
    """Abstract base class for reject inference methods.

    All reject inference strategies inherit from this class. Subclasses
    must implement `_preprocess()` which updates the training set (X, y)
    or defines sample weights before the final model is fit.

    Parameters
    ----------
    base_estimator : BaseEstimator
        The final classifier trained on the (possibly augmented) labeled data.
    reject_estimator : BaseEstimator
        An auxiliary classifier used to estimate acceptance probabilities or
        infer labels for the rejected population.
    """

    def __init__(self, base_estimator: BaseEstimator, reject_estimator: BaseEstimator):
        self.base_estimator = base_estimator
        self.reject_estimator = reject_estimator

    def _wrap(self, X: np.ndarray) -> Union[np.ndarray, pd.DataFrame]:
        """Re-wrap a numpy array into a DataFrame if feature names are available.

        This prevents feature name warnings from LightGBM and sklearn.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix to wrap.

        Returns
        -------
        DataFrame or ndarray
            Wrapped data with original column names if available.
        """
        if getattr(self, "feature_names_in_", None) is not None:
            return pd.DataFrame(X, columns=self.feature_names_in_)
        return X

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> "RejectInference":
        """Fit the reject inference model.

        The feature matrix must contain both labeled (accepted) and unlabeled
        (rejected) samples in sequence. Labels for rejected samples must be -1.

        Parameters
        ----------
        X : array-like of shape (n_labeled + n_rejected, n_features)
            Combined feature matrix.
        y : array-like of shape (n_labeled + n_rejected,)
            Labels: 0/1 for accepted applicants, -1 for rejected applicants.

        Returns
        -------
        RejectInference
            Fitted model.
        """
        self.feature_names_in_ = X.columns.to_numpy() if isinstance(X, pd.DataFrame) else None
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        self.classes_ = np.unique(y_arr[y_arr != -1])

        X_unl = X_arr[y_arr == -1]
        X_lab = X_arr[y_arr != -1]
        y_lab = y_arr[y_arr != -1]

        X_updated, y_updated, sample_weights = self._preprocess(X_lab, y_lab, X_unl)
        self.base_estimator.fit(self._wrap(X_updated), y_updated, sample_weight=sample_weights)
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities using the fitted base estimator.

        Parameters
        ----------
        X : array-like
            Feature matrix for inference.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Class probabilities [P(0), P(1)].
        """
        return self.base_estimator.predict_proba(X)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict binary class labels using the fitted base estimator.

        Parameters
        ----------
        X : array-like
            Feature matrix for inference.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted binary labels.
        """
        return self.base_estimator.predict(X)

    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Prepare the updated training set and sample weights.

        Must be implemented by subclasses.

        Parameters
        ----------
        X : np.ndarray
            Labeled (accepted) feature matrix.
        y : np.ndarray
            Labels for accepted applicants (0 or 1).
        X_unl : np.ndarray
            Unlabeled (rejected) feature matrix.

        Returns
        -------
        (X_updated, y_updated, sample_weights)
        """
        raise NotImplementedError("Subclasses must implement _preprocess().")


class RejectUpward(RejectInference):
    """Upward reweighting strategy for reject inference.

    Trains an acceptance model on the combined accepted + rejected population,
    then up-weights accepted samples that are similar to rejected ones
    (i.e., those that were narrowly accepted). The intuition is that the
    model should pay more attention to borderline cases.

    Weight = 1 / P(accepted | x)
    """

    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_concat = np.concatenate([X, X_unl])
        y_accept = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])
        indices = np.random.permutation(X_concat.shape[0])
        self.reject_estimator.fit(self._wrap(X_concat[indices]), y_accept[indices])
        p_accept = self.reject_estimator.predict_proba(self._wrap(X))[:, 1]
        sample_weights = 1.0 / np.clip(p_accept, 1e-6, 1.0)
        return X, y, sample_weights


class RejectDownward(RejectInference):
    """Downward reweighting strategy for reject inference.

    The complement of RejectUpward: accepted samples that look very clearly
    accepted (high P(accepted)) are down-weighted, since they are least
    representative of the full applicant population.

    Weight = 1 - P(accepted | x) = P(rejected | x)
    """

    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_concat = np.concatenate([X, X_unl])
        y_accept = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])
        indices = np.random.permutation(X_concat.shape[0])
        self.reject_estimator.fit(self._wrap(X_concat[indices]), y_accept[indices])
        p_accept = self.reject_estimator.predict_proba(self._wrap(X))[:, 1]
        sample_weights = 1.0 - p_accept
        return X, y, sample_weights


class RejectSoftCutoff(RejectInference):
    """Soft cut-off reweighting strategy for reject inference.

    Groups all samples (accepted + rejected) into score bins by acceptance
    probability. Within each bin, the weight of each accepted sample is set
    to the inverse of the acceptance rate in that bin, correcting for the
    over-representation of accepted applicants in high-score bins.
    """

    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_concat = np.concatenate([X, X_unl])
        y_accept = np.concatenate([np.ones(X.shape[0]), np.zeros(X_unl.shape[0])])
        indices = np.random.permutation(X_concat.shape[0])
        self.reject_estimator.fit(self._wrap(X_concat[indices]), y_accept[indices])

        prob_accept = self.reject_estimator.predict_proba(self._wrap(X_concat))[:, 1]
        intervals = np.percentile(prob_accept, np.linspace(0, 100, num=100))
        sample_weights = np.ones(X_concat.shape[0])

        for i in range(len(intervals) - 1):
            in_interval = (prob_accept >= intervals[i]) & (prob_accept < intervals[i + 1])
            if in_interval.sum() > 0:
                accept_rate = np.mean(y_accept[in_interval])
                sample_weights[in_interval] = 1.0 / accept_rate if accept_rate > 0 else 1.0

        # Only accepted samples enter final training
        sample_weights = sample_weights[:X.shape[0]]
        return X, y, sample_weights


class FuzzyParcelling(RejectInference):
    """Fuzzy Parcelling strategy for reject inference.

    Each rejected applicant is duplicated: one copy is labeled as a good
    payer (0) and one as a bad payer (1). The good copy receives a weight
    equal to P(good payer | x) and the bad copy receives P(bad payer | x).
    Accepted samples keep weight 1. This softly incorporates rejected
    applicants without committing to a hard label assignment.
    """

    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Fit reject estimator on accepted data to predict reject labels
        self.reject_estimator.fit(self._wrap(X), y)
        prob_good = self.reject_estimator.predict_proba(self._wrap(X_unl))[:, 1]

        X_aug = np.concatenate([X, X_unl, X_unl])
        y_aug = np.concatenate([y, np.zeros(X_unl.shape[0]), np.ones(X_unl.shape[0])])
        weights = np.concatenate([np.ones(X.shape[0]), 1.0 - prob_good, prob_good])
        return X_aug, y_aug, weights


class RejectExtrapolation(RejectInference):
    """Hard-label extrapolation strategy for reject inference.

    Also known as augmentation or simple parcelling. A model is first fit
    on accepted applicants, then used to assign hard (0/1) labels to the
    rejected population. Selected rejected samples (based on the `mode`
    argument) are added to the training set with equal weight and a new
    model is trained on the combined data.

    Parameters
    ----------
    base_estimator : BaseEstimator
        Final classifier.
    reject_estimator : BaseEstimator
        Auxiliary classifier used to infer labels for rejected samples.
    mode : str, optional
        Which rejected samples to include:
        - "positive": only those predicted as good payers (prob > 0.5)
        - "all": include all rejected samples
        - "confident": only high-confidence predictions (prob > 0.8 or < 0.15)
        By default "positive".
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        reject_estimator: BaseEstimator,
        mode: str = "positive",
    ):
        super().__init__(base_estimator, reject_estimator)
        self.mode = mode

    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.reject_estimator.fit(self._wrap(X), y)
        prob_good = self.reject_estimator.predict_proba(self._wrap(X_unl))[:, 1]
        y_unl = (prob_good > 0.5).astype(int)

        if self.mode == "positive":
            selected = prob_good > 0.5
        elif self.mode == "all":
            selected = np.ones(len(prob_good), dtype=bool)
        elif self.mode == "confident":
            selected = (prob_good > 0.8) | (prob_good < 0.15)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Use 'positive', 'all', or 'confident'.")

        X_new = np.concatenate([X, X_unl[selected]])
        y_new = np.concatenate([y, y_unl[selected]])
        weights = np.ones(X_new.shape[0])
        return X_new, y_new, weights


class RejectSpreading(RejectInference):
    """Label Spreading strategy for reject inference.

    A semi-supervised, graph-based technique. Rejected samples receive a
    temporary label of -1 and are concatenated with accepted samples. The
    sklearn LabelSpreading algorithm propagates labels through the graph,
    assigning labels to rejected samples based on their neighborhood
    structure. The result is a fully labeled dataset for final model training.
    """

    def _preprocess(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_unl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_concat = np.concatenate([X, X_unl])
        y_concat = np.concatenate([y, -1 * np.ones(X_unl.shape[0])])

        label_spreading = LabelSpreading(kernel="knn")
        label_spreading.fit(X_concat, y_concat)

        y_new = label_spreading.transduction_
        weights = np.ones(X_concat.shape[0])
        return X_concat, y_new, weights
