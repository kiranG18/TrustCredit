"""
Keras-backed MLP classifier with a scikit-learn compatible API.

Supports class weights and sample weights, making it compatible with
the reject inference and fairness modules in TrustCredit.
"""

from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class MLPClassifier(ClassifierMixin, BaseEstimator):
    """Neural network classifier built on Keras with the sklearn API.

    Implements a standard MLP for binary classification with support for
    class-level and sample-level weighting. The architecture is a stack of
    fully-connected ReLU layers with L2 regularization and a sigmoid output.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, optional
        Number of units per hidden layer. By default (100,).
    batch_size : int, optional
        Mini-batch size for training. By default 32.
    learning_rate_init : float, optional
        Initial learning rate for the Adam optimizer. By default 0.1.
    learning_rate_decay_rate : float, optional
        Inverse-time decay rate. Set to 1.0 for constant learning rate. By default 0.1.
    alpha : float, optional
        L2 regularization strength. By default 0.0001.
    epochs : int, optional
        Number of full passes through the training data. By default 100.
    class_weight : str or None, optional
        If "balanced", inverse class frequencies are used as weights. By default None.
    random_state : int or None, optional
        Seed for reproducibility. By default None.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        batch_size: int = 32,
        learning_rate_init: float = 0.1,
        learning_rate_decay_rate: float = 0.1,
        alpha: float = 0.0001,
        epochs: int = 100,
        class_weight: Optional[str] = None,
        random_state: Optional[int] = None,
    ):
        self._random_state = random_state
        self._seed_everything(random_state)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.alpha = alpha
        self.epochs = epochs
        self.class_weight = class_weight

    @property
    def random_state(self) -> Optional[int]:
        return self._random_state

    @random_state.setter
    def random_state(self, value: Optional[int]) -> None:
        self._random_state = value
        self._seed_everything(value)

    def _seed_everything(self, value: Optional[int]) -> None:
        """Set global random seeds for reproducibility."""
        if value is not None:
            try:
                import tensorflow as tf
                import keras
                np.random.seed(value)
                tf.random.set_seed(value)
                keras.utils.set_random_seed(value)
                tf.config.experimental.enable_op_determinism()
            except ImportError:
                pass

    def _build_model(self, X: Union[np.ndarray, pd.DataFrame]):
        """Construct and compile the Keras sequential model.

        Parameters
        ----------
        X : array-like
            Input data used to determine input dimensionality.

        Returns
        -------
        keras.Sequential
            Compiled Keras model.
        """
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.regularizers import l2
        from keras.optimizers.schedules import InverseTimeDecay
        from keras.optimizers import Adam

        model = Sequential()
        model.add(Dense(
            self.hidden_layer_sizes[0],
            input_dim=X.shape[1],
            activation="relu",
            kernel_regularizer=l2(self.alpha),
        ))
        for layer_size in self.hidden_layer_sizes[1:]:
            model.add(Dense(layer_size, activation="relu", kernel_regularizer=l2(self.alpha)))
        model.add(Dense(1, activation="sigmoid"))

        lr_schedule = InverseTimeDecay(
            self.learning_rate_init,
            decay_steps=self.epochs,
            decay_rate=self.learning_rate_decay_rate,
            staircase=False,
        )
        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=lr_schedule),
            metrics=["AUC"],
        )
        return model

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "MLPClassifier":
        """Train the MLP on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Binary target labels.
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights. If provided, overrides class_weight. By default None.

        Returns
        -------
        MLPClassifier
            Fitted model instance.
        """
        class_weight_dict = None
        if self.class_weight == "balanced" and sample_weight is None:
            class_weight_dict = {0: 1 / np.sum(y == 0), 1: 1 / np.sum(y == 1)}

        self.model_ = self._build_model(X)
        self.model_.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            class_weight=class_weight_dict,
            sample_weight=sample_weight,
            verbose=0,
        )
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Columns are [P(class=0), P(class=1)].
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        prob = self.model_(X_arr, training=False)
        return np.concatenate([1 - prob, prob], axis=1)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict binary class labels using a 0.5 threshold.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Binary predictions.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return (self.model_(X_arr, training=False) > 0.5).numpy().flatten()

    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> float:
        """Return the AUC score on the given data.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            True labels.

        Returns
        -------
        float
            AUC score.
        """
        return float(self.model_.evaluate(X, y, verbose=0)[1])
