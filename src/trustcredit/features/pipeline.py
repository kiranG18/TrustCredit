"""
Feature engineering and preprocessing pipeline for TrustCredit.

Contains:
- EBE (Evidence-Based Encoding): smoothed target encoding for high-cardinality categoricals
- create_pipeline(): builds a full sklearn preprocessing + classifier pipeline
"""

from typing import Optional, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler


class EBE(TransformerMixin, BaseEstimator):
    """Evidence-Based Encoding (EBE) for categorical features.

    Replaces each category with a smoothed estimate of the target mean for
    that category. The smoothing prevents overfitting on rare categories
    by blending the category mean with the global mean.

    Formula: (category_mean * count + k * global_mean) / (count + k)

    Parameters
    ----------
    k : int, optional
        Smoothing strength. Higher values pull estimates toward the global
        mean more aggressively. By default 1.
    """

    def __init__(self, k: int = 1):
        self.k = k

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EBE":
        """Compute per-category smoothed means from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Categorical feature columns to encode.
        y : pd.Series
            Binary target variable.

        Returns
        -------
        EBE
            Fitted encoder.
        """
        self.feature_names_in_ = []
        self.n_features = X.shape[1]
        self._aux_dict_main = {}
        self.mean = {}
        self.unique_values = {col: X[col].unique() for col in X.columns}

        for i in range(self.n_features):
            Xi = X.iloc[:, i]
            X_name = Xi.name
            y_series = pd.Series(y, index=X.index)
            aux_dict = y_series.groupby(Xi).agg(["mean", "count"]).to_dict()
            self._aux_dict_main[X_name] = aux_dict
            self.feature_names_in_.append(X_name)
            self.mean[X_name] = y_series.mean()
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Apply smoothed target encoding to categorical columns.

        Parameters
        ----------
        X : pd.DataFrame
            Categorical feature columns to encode.
        y : pd.Series, optional
            Unused; present for sklearn pipeline compatibility.

        Returns
        -------
        np.ndarray
            Encoded feature matrix.
        """
        Xt_list = []
        for X_name in self.feature_names_in_:
            Xi = X[X_name].copy()
            means_dict = self._aux_dict_main[X_name]["mean"]
            counts_dict = self._aux_dict_main[X_name]["count"]
            global_mean = self.mean[X_name]

            default_mean = pd.Series(list(means_dict.values())).mode()[0]
            group_ave = Xi.map(means_dict).fillna(default_mean).astype(float)
            group_count = Xi.map(counts_dict).fillna(0).astype(float)

            Xt = (group_ave * group_count + self.k * global_mean) / (group_count + self.k)
            Xt_list.append(Xt.values.reshape(-1, 1))

        return np.hstack(Xt_list)

    def get_feature_names_out(self, input_features: Union[pd.DataFrame, List] = None) -> List[str]:
        """Return feature names for sklearn pipeline compatibility.

        Parameters
        ----------
        input_features : ignored
            Present for sklearn API compatibility.

        Returns
        -------
        List[str]
            List of encoded feature names.
        """
        return list(self.feature_names_in_)


def create_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    classifier: Optional[BaseEstimator] = None,
    cat_cols: Union[str, List[str]] = "infer",
    onehot: bool = True,
    onehotdrop: bool = False,
    normalize: bool = True,
    do_EBE: bool = True,
    crit: int = 3,
) -> Pipeline:
    """Build a full preprocessing and classification pipeline.

    Handles missing values, ordinal/EBE/one-hot encoding, and scaling.
    Supports heterogeneous credit data with mixed numeric and categorical types.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (used to infer column types).
    y : pd.Series
        Binary target variable (used for EBE fitting).
    classifier : BaseEstimator, optional
        Any sklearn-compatible classifier. If None, pipeline ends after preprocessing.
    cat_cols : str or list, optional
        Categorical column names, or "infer" to detect automatically. By default "infer".
    onehot : bool, optional
        Apply one-hot encoding to low-cardinality categoricals. By default True.
    onehotdrop : bool, optional
        Drop one category per one-hot column to avoid multicollinearity. By default False.
    normalize : bool, optional
        Apply standard scaling to numeric columns. By default True.
    do_EBE : bool, optional
        Apply EBE encoding to high-cardinality categoricals. By default True.
    crit : int, optional
        Cardinality threshold to switch from OHE to EBE. By default 3.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Ready-to-fit preprocessing + classifier pipeline.
    """
    if cat_cols == "infer":
        num_cols = [col for col in X.columns if X[col].dtype.kind in ["b", "i", "u", "f", "c"]]
        cat_cols_all = [col for col in X.columns if X[col].dtype.kind in ["O", "S", "U"]]
    else:
        num_cols = X.columns.difference(cat_cols).tolist()
        cat_cols_all = list(cat_cols)

    ebe_cols = [col for col in cat_cols_all if X[col].nunique() >= crit and do_EBE]
    cat_cols_ohe = [col for col in cat_cols_all if col not in ebe_cols]

    # Step 1: Fill missing values
    fill_pipe = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="mean"))]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), cat_cols_ohe),
            ("ebe", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), ebe_cols),
        ],
        verbose_feature_names_out=False,
    )
    fill_pipe.set_output(transform="pandas")

    # Step 2: Ordinal encoding (low-card cats) + EBE (high-card cats)
    encoder_pipe = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            (
                "cat",
                OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
                cat_cols_ohe,
            ),
            (
                "ebe",
                EBE() if do_EBE else OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
                ebe_cols,
            ),
        ],
        verbose_feature_names_out=False,
    )
    encoder_pipe.set_output(transform="pandas")

    # Step 3: Standard scaling for numeric columns
    scaling_pipe = ColumnTransformer(
        [("scaler", StandardScaler(), num_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    scaling_pipe.set_output(transform="pandas")

    # Step 4: One-hot encoding for low-cardinality categoricals
    onehot_encoder = (
        OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore")
        if onehotdrop
        else OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    )
    onehot_pipe = ColumnTransformer(
        transformers=[("onehot_encoder", onehot_encoder, cat_cols_ohe)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    onehot_pipe.set_output(transform="pandas")

    steps = [
        ("fill", fill_pipe),
        ("le", encoder_pipe),
        ("ss", scaling_pipe if normalize else "passthrough"),
        ("hot", onehot_pipe if onehot else "passthrough"),
        ("classifier", classifier),
    ]

    return Pipeline(steps=steps)
