"""
Explainability tools for TrustCredit credit scoring pipelines.

Provides global and local explanation methods that work directly with
sklearn Pipeline objects, handling the preprocessing steps transparently.

Global explanations:
    PartialDependencePipeline -- PDP and ICE curves for any pipeline feature

Local explanations:
    ShapPipelineExplainer  -- SHAP permutation values for individual predictions
    LimePipelineExplainer  -- LIME local linear approximations

Counterfactual explanations (actionable recourse):
    MAPOCAM  -- MAPOCAM algorithm for minimal-cost counterfactuals
    Dice     -- DiCE genetic algorithm for diverse counterfactual generation
    display_cfs() -- formats counterfactual output as a comparison table
"""

from typing import List, Union, Dict, Optional, Any

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import partial_dependence
from sklearn.pipeline import Pipeline


class PartialDependencePipeline:
    """Compute Partial Dependence Plots (PDP) or ICE curves for a full pipeline.

    Handles the preprocessing steps inside the pipeline transparently,
    and returns values transformed back to the original feature scale.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted sklearn pipeline (preprocessing + classifier).
    grid_resolution : int, optional
        Number of grid points to evaluate each feature at. By default 20.
    kind : str, optional
        "average" for PDP, "individual" for ICE. By default "average".
    """

    def __init__(self, pipeline: Pipeline, grid_resolution: int = 20, kind: str = "average"):
        self.preprocess = pipeline[:-1]
        self.model = pipeline[-1]
        self.grid_resolution = grid_resolution
        self.kind = kind

        class _ModelWrapper(RegressorMixin, BaseEstimator):
            def __init__(self, model):
                self.model = model
                self.classes_ = [0, 1]

            def fit(self, X, y):
                return self

            def predict(self, X):
                return self.model.predict_proba(X)[:, 1]

            def predict_proba(self, X):
                return self.model.predict_proba(X)[:, 1]

            def __sklearn_is_fitted__(self):
                return True

        self.model_wrapper = _ModelWrapper(self.model)

    def __call__(self, X: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Compute partial dependence for the specified features.

        Parameters
        ----------
        X : pd.DataFrame
            Input data (in original feature space).
        features : list of str
            Feature names to compute PDP/ICE for. Must be numeric features.

        Returns
        -------
        dict with keys:
            "values" -- grid values in original scale
            "prediction" -- PDP/ICE values
            "deciles" -- feature deciles in original scale
        """
        X_pre = self.preprocess.transform(X)
        result = partial_dependence(
            self.model_wrapper, X_pre, features,
            kind=self.kind, grid_resolution=self.grid_resolution,
            percentiles=(0.05, 0.95), method="brute",
        )
        result["values"] = result["grid_values"]

        deciles = [np.percentile(X_pre[f], np.arange(4, 94, 2)) for f in features]

        # Inverse-transform scaled features back to original scale
        scaled_features = self.preprocess[2].transformers_[0][2]
        for i, feature in enumerate(features):
            if feature in scaled_features:
                idx = scaled_features.index(feature)
                scaler = self.preprocess[2].transformers_[0][1]
                mu, sigma = scaler.mean_[idx], scaler.scale_[idx]
                result["values"][i] = result["values"][i] * sigma + mu
                deciles[i] = deciles[i] * sigma + mu

        return {"values": result["values"], "prediction": result[self.kind], "deciles": deciles}


class ShapPipelineExplainer:
    """SHAP permutation explainer that wraps a full sklearn pipeline.

    Automatically handles the preprocessing steps and maps SHAP values
    back to the original feature representation.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted preprocessing + classifier pipeline.
    background_samples : pd.DataFrame
        Background dataset for the SHAP masker (typically training data or a subset).
    method_explain : str, optional
        "prob" to explain probability scores, "pred" for binary decisions. By default "prob".
    threshold : float, optional
        Decision threshold used when method_explain="pred". By default 0.5.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        background_samples: pd.DataFrame,
        method_explain: str = "prob",
        threshold: float = 0.5,
    ):
        import shap
        self.method_explain = method_explain
        self.threshold = threshold
        self.preprocess = pipeline[:3]
        self.model = pipeline[3:]
        self.categoric_features = (
            self.preprocess[1].transformers_[1][2].copy()
            + self.preprocess[1].transformers_[2][2].copy()
        )
        self.categories_mapping = [
            {i: str(v) for i, v in enumerate(x)}
            for x in self.preprocess[1].transformers_[1][1].categories_
        ] + [
            {i: str(v) for i, v in enumerate(x)}
            for x in self.preprocess[1].transformers_[2][1].unique_values.values()
        ]

        X_pre = self.preprocess.transform(background_samples)
        self.feature_names = X_pre.columns.tolist()

        wrap_model = (
            (lambda x: self.model.predict_proba(x)[:, 1])
            if method_explain == "prob"
            else (lambda x: self.model.predict_proba(x)[:, 1] > self.threshold)
        )
        self.explainer = shap.Explainer(wrap_model, masker=X_pre, algorithm="permutation")

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute SHAP values for the input samples.

        Parameters
        ----------
        X : pd.DataFrame
            Input samples in original feature space.

        Returns
        -------
        pd.DataFrame
            SHAP values with one column per preprocessed feature.
        """
        X_pre = self.preprocess.transform(X)
        shap_values = self.explainer(X_pre).values
        return pd.DataFrame(
            {f: shap_values[:, i] for i, f in enumerate(self.feature_names)},
            columns=self.feature_names,
        )

    def plot_explanation(self, X: pd.DataFrame) -> None:
        """Plot a SHAP waterfall explanation for a single sample.

        Parameters
        ----------
        X : pd.DataFrame
            Single-row input (one sample to explain).
        """
        X_pre = self.preprocess.transform(X)
        prob = self.model.predict_proba(X_pre)[0, 1]
        explanation = self(X)

        top_k = 7
        v = np.abs(explanation.values[0])
        important = explanation.columns[v.argsort()[::-1]][:top_k].tolist()[::-1]
        imp = explanation[important].values[0]
        importance_dict = dict(zip(important, imp))

        pos_color, neg_color = "#80b1d3", "#fccde5"
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle(f"Default Probability = {prob:.2%}", fontsize=13, fontweight="bold")

        axs[0].barh(important, imp, color=[neg_color if x < 0 else pos_color for x in imp])
        axs[0].axvline(0, color="#606060", lw=1)
        for sp in ["top", "right", "bottom"]:
            axs[0].spines[sp].set_visible(False)
        xrange = imp.max() - imp.min()
        pad = 0.23
        axs[0].set_xlim(imp.min() - pad * xrange, imp.max() + pad * xrange)
        axs[0].set_xticks([])
        axs[0].set_title("SHAP Values", fontweight="bold")
        for j, feat in enumerate(important):
            v_val = importance_dict[feat]
            axs[0].text(v_val, j, f"{v_val:.2f}",
                        ha="right" if v_val < 0 else "left", va="center", fontsize=11)

        # Feature value table
        values = X_pre[important[::-1]].values[0].tolist()
        scaled_features = self.preprocess[2].transformers_[0][2]
        for i, feat in enumerate(important[::-1]):
            if feat in scaled_features:
                scaler = self.preprocess[2].transformers_[0][1]
                idx = scaled_features.index(feat)
                values[i] = values[i] * scaler.scale_[idx] + scaler.mean_[idx]
            if feat in self.categoric_features:
                cidx = self.categoric_features.index(feat)
                text = self.categories_mapping[cidx].get(int(values[i]), str(values[i]))
                values[i] = text[:10] + "..." if len(text) > 10 else text

        values = [round(x, 2) if isinstance(x, float) else x for x in values]
        table = axs[1].table(
            cellText=np.array([values]).T,
            rowLabels=important[::-1],
            loc="center", colWidths=[0.3], fontsize=12,
        )
        axs[1].axis("off")
        for j, feat in enumerate(important[::-1]):
            c = neg_color if importance_dict[feat] < 0 else pos_color
            table[(j, 0)].set_facecolor(c)
            table[(j, -1)].set_facecolor(c)

        plt.tight_layout()


class LimePipelineExplainer:
    """LIME tabular explainer wrapping a full sklearn pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted preprocessing + classifier pipeline.
    background_samples : pd.DataFrame
        Dataset used to initialize the LIME explainer (typically training data).
    method_explain : str, optional
        "prob" for probability explanations, "pred" for binary. By default "prob".
    threshold : float, optional
        Threshold for binary predictions when method_explain="pred". By default 0.5.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        background_samples: pd.DataFrame,
        method_explain: str = "prob",
        threshold: float = 0.5,
    ):
        from lime import lime_tabular

        self.method_explain = method_explain
        self.threshold = threshold
        self.preprocess = pipeline[:3]
        self.model = pipeline[3:]
        X_pre = self.preprocess.transform(background_samples)
        self.categoric_features = (
            self.preprocess[1].transformers_[1][2].copy()
            + self.preprocess[1].transformers_[2][2].copy()
        )
        self.feature_names = X_pre.columns.tolist()
        self.categoric_features_idx = [
            i for i, f in enumerate(self.feature_names) if f in self.categoric_features
        ]
        self.categories_mapping = {}
        for i, idx in enumerate(self.categoric_features_idx):
            feat = self.feature_names[idx]
            if feat in self.preprocess[1].transformers_[1][2]:
                cats = self.preprocess[1].transformers_[1][1].categories_[i].tolist()
            else:
                cats = self.preprocess[1].transformers_[2][1].unique_values[feat].tolist()
            self.categories_mapping[idx] = [str(v) for v in cats]

        self.explainer = lime_tabular.LimeTabularExplainer(
            X_pre.values,
            feature_names=self.feature_names,
            class_names=["1"],
            mode="classification",
            categorical_features=self.categoric_features_idx,
            categorical_names=self.categories_mapping,
            discretize_continuous=False,
        )

    def __call__(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute LIME explanations for all input samples.

        Parameters
        ----------
        X : pd.DataFrame
            Input samples in original feature space.

        Returns
        -------
        pd.DataFrame
            LIME coefficient matrix (one row per sample, one column per feature).
        """
        def pred_fn(X_arr):
            X_df = pd.DataFrame(X_arr, columns=self.feature_names)
            return self.model.predict_proba(X_df)

        X_pre = self.preprocess.transform(X)
        n = X_pre.shape[0]
        explanation_dict = {f: np.zeros(n) for f in self.feature_names}
        for i in range(n):
            exp = self.explainer.explain_instance(
                X_pre.values[i, :].flatten(), pred_fn,
                num_features=len(self.feature_names),
            )
            for f, v in exp.as_list():
                key = f.split("=")[0] if "=" in f else f
                if key in explanation_dict:
                    explanation_dict[key][i] = v
        return pd.DataFrame(explanation_dict)


class MAPOCAM:
    """MAPOCAM: Multi-Objective Actionable Conterfactual Explanations.

    Finds minimal-cost counterfactual explanations by searching over a
    discretized action space. Only mutable features can be changed.

    Parameters
    ----------
    pipeline : Pipeline
        Fitted preprocessing + classifier pipeline.
    X : pd.DataFrame
        Training data (used to define feature bounds and action sets).
    mutable_features : list of str
        Features that can be changed to achieve the counterfactual.
    target : int
        Target prediction class (typically 1 = approved).
    max_changes : int, optional
        Maximum number of features to change. By default 3.
    criteria : str, optional
        Objective criteria: "percentile", "percentile_changes", or "non_dom".
        By default "percentile".
    step_size : float, optional
        Grid granularity for action search. By default 0.01.
    threshold : float, optional
        Classification threshold. By default 0.5.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        X: pd.DataFrame,
        mutable_features: List[str],
        target: int,
        max_changes: int = 3,
        criteria: str = "percentile",
        step_size: float = 0.01,
        threshold: float = 0.5,
    ):
        import cfmining.algorithms as alg
        import cfmining.criteria as crit
        from cfmining.action_set import ActionSet
        from sklearn.linear_model import LogisticRegression

        self.preprocess = pipeline[:2]
        self.model = pipeline[2:]
        self.all_features = self.preprocess.get_feature_names_out().tolist()
        self.mutable_features = mutable_features
        categoric_features = (
            pipeline[1].transformers_[1][2].copy()
            + pipeline[1].transformers_[2][2].copy()
        )
        for col in mutable_features:
            assert col not in categoric_features, f"Mutable feature '{col}' must be numeric."

        # Build feature importance from logistic coefficients if available
        feat_importance = [0.0] * len(self.all_features)
        is_logistic = isinstance(pipeline[-1], LogisticRegression)
        if is_logistic:
            model_features = self.model[-2].get_feature_names_out().tolist()
            coefs = pipeline[-1].coef_[0]
            for i, col in enumerate(self.all_features):
                if col in mutable_features:
                    idx = model_features.index(col)
                    feat_importance[i] = coefs[idx]

        X_pre = self.preprocess.transform(X)
        # Add tiny noise to zero-variance columns
        for col in X_pre.columns:
            if np.percentile(X_pre[col], 1) == np.percentile(X_pre[col], 99):
                X_pre[col] += np.random.normal(0, 1e-4, X_pre.shape[0])

        self.action_set = ActionSet(X=X_pre)
        for i, feat in enumerate(self.action_set):
            if feat.name not in self.mutable_features:
                feat.mutable = False
            direction = 1 if feat_importance[i] < 0 else -1
            feat.step_size = step_size
            feat.flip_direction = direction
            feat.step_direction = direction
            feat.update_grid()

        class _HelperClassifier:
            def __init__(self, pipeline, target, threshold):
                self.model = pipeline[2:]
                self.feature_names = pipeline[:2].get_feature_names_out().tolist()
                self.target = target
                self.threshold = threshold

            def predict_proba(self, X_arr):
                X_df = pd.DataFrame([X_arr], columns=self.feature_names)
                return self.model.predict_proba(X_df)[0, self.target]

        self.predictor = _HelperClassifier(pipeline, target, threshold)

        if criteria == "percentile":
            perc_calc = crit.PercentileCalculator(action_set=self.action_set)
            self.criteria_fn = lambda x: crit.PercentileCriterion(x, perc_calc)
        elif criteria == "percentile_changes":
            perc_calc = crit.PercentileCalculator(action_set=self.action_set)
            self.criteria_fn = lambda x: crit.PercentileChangesCriterion(x, perc_calc)
        elif criteria == "non_dom":
            self.criteria_fn = lambda x: crit.NonDomCriterion(
                [feat.flip_direction for feat in self.action_set]
            )
        else:
            raise ValueError(f"Invalid criteria '{criteria}'. Use 'percentile', 'percentile_changes', or 'non_dom'.")

        self._alg = alg
        self.max_changes = max_changes

    def fit(self, individual: Union[np.ndarray, pd.DataFrame]) -> List[np.ndarray]:
        """Search for counterfactual solutions for a given individual.

        Parameters
        ----------
        individual : array-like
            Single sample in the original feature space.

        Returns
        -------
        list of np.ndarray
            List of counterfactual solutions in the preprocessed feature space.
        """
        individual_ = self.preprocess.transform(individual).values.flatten()
        method = self._alg.MAPOCAM(
            self.action_set, individual_, self.predictor,
            max_changes=self.max_changes,
            compare=self.criteria_fn(individual_),
        )
        method.fit()
        return method.solutions


class Dice:
    """DiCE: Diverse Counterfactual Explanations via genetic optimization.

    Generates multiple diverse counterfactuals for a given input by
    framing the problem as a genetic optimization over the feature space.

    Parameters
    ----------
    X : pd.DataFrame
        Training data (preprocessed internally).
    Y : array-like
        Target labels.
    pipeline : Pipeline
        Fitted preprocessing + classifier pipeline.
    n_cfs : int
        Number of counterfactuals to generate per query.
    mutable_features : list of str
        Features that can be modified.
    sparsity_weight : float, optional
        Controls the sparsity trade-off in optimization. By default 0.2.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        Y: Union[np.ndarray, pd.Series],
        pipeline: Pipeline,
        n_cfs: int,
        mutable_features: List[str],
        sparsity_weight: float = 0.2,
    ):
        import dice_ml

        self.total_CFs = n_cfs
        self.sparsity_weight = sparsity_weight
        self.mutable_features = mutable_features
        X_pre = pipeline[:2].transform(X)
        self.features = X_pre.columns.tolist()
        self.categoric_features = (
            pipeline[1].transformers_[1][2].copy()
            + pipeline[1].transformers_[2][2].copy()
        )
        self.continuous_features = [c for c in self.features if c not in self.categoric_features]
        self.preprocess = pipeline[:2]
        self.model = pipeline[2:]

        class _DiceModelWrapper:
            def __init__(self, model):
                self.model = model
                self.classes_ = getattr(model, "classes_", [0, 1])
            def predict_proba(self, X):
                if isinstance(X, pd.DataFrame):
                    X = X.copy()
                    for col in X.columns:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                return self.model.predict_proba(X)
            def predict(self, X):
                if isinstance(X, pd.DataFrame):
                    X = X.copy()
                    for col in X.columns:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                return self.model.predict(X)

        dice_model = dice_ml.Model(model=_DiceModelWrapper(self.model), backend="sklearn", model_type="classifier")
        X_ext = X_pre.copy()
        X_ext["target"] = Y
        dice_data = dice_ml.Data(
            dataframe=X_ext,
            continuous_features=self.continuous_features,
            outcome_name="target",
        )
        self.exp = dice_ml.Dice(dice_data, dice_model, method="genetic")

    def fit(self, individual: Union[np.ndarray, pd.DataFrame]) -> List[Any]:
        """Generate counterfactuals for a single individual.

        Parameters
        ----------
        individual : array-like
            Single sample in the original feature space.

        Returns
        -------
        list
            List of counterfactual feature vectors.
        """
        if isinstance(individual, np.ndarray):
            individual = pd.DataFrame([individual], columns=self.features)
        individual_ = self.preprocess.transform(individual)
        dice_exp = self.exp.generate_counterfactuals(
            individual_,
            total_CFs=self.total_CFs,
            desired_class="opposite",
            sparsity_weight=self.sparsity_weight,
            features_to_vary=self.mutable_features,
        )
        solutions = json.loads(dice_exp.to_json())["cfs_list"][0]
        return [s[:-1] for s in solutions]


def display_cfs(
    individual: pd.DataFrame,
    cfs: List[Any],
    pipeline: Pipeline,
    show_change: bool = False,
) -> pd.DataFrame:
    """Format counterfactuals into a readable comparison table.

    Parameters
    ----------
    individual : pd.DataFrame
        The original sample (single row).
    cfs : list
        Counterfactual feature vectors returned by MAPOCAM or Dice.
    pipeline : Pipeline
        Preprocessing pipeline used to transform the individual.
    show_change : bool, optional
        If True, appends the change value in parentheses. By default False.

    Returns
    -------
    pd.DataFrame
        Transposed comparison table: features as rows, cases as columns.
        Only features that differ from the original are shown.
    """
    preprocess = pipeline[:2]
    individual_ = preprocess.transform(individual).values.flatten()
    feature_names = preprocess.get_feature_names_out().tolist()
    df = pd.DataFrame(
        [individual_] + list(cfs),
        columns=feature_names,
        index=["Original"] + [f"CF {i}" for i in range(len(cfs))],
    )

    altered_cols = [c for c in df.columns if df[c].max() != df[c].min()]
    df = df[altered_cols].copy()

    for col in altered_cols:
        df[col + "_delta"] = df[col] - df[col].iloc[0]

    for col in df.columns:
        df[col] = df[col].round(3) if df[col].max() < 10 else df[col].round(0).astype(int)

    if show_change:
        for col in altered_cols:
            def fmt_change(x):
                if x > 0:
                    return "+"
                elif x < 0:
                    return "-"
                return ""
            sign = df[col + "_delta"].apply(fmt_change)
            df[col] = df[col].astype(str) + " (" + sign + df[col + "_delta"].astype(str) + ")"

    n_changes = (df[altered_cols] != df[altered_cols].iloc[0]).sum(axis=1)
    df["n_changes"] = n_changes
    df = df.sort_values("n_changes").drop(columns="n_changes")

    df = df[altered_cols].astype(str)
    for i in range(1, len(cfs) + 1):
        for col in altered_cols:
            if df.at[df.index[i], col + "_delta"] == "0" if col + "_delta" in df else False:
                df.iat[i, df.columns.get_loc(col)] = "---"

    return df[altered_cols].T
