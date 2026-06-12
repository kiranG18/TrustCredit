"""
Hyperparameter optimization using Optuna for TrustCredit models.

Provides:
- optimize_model(): full pipeline optimization via cross-validation or validation set
- optimize_model_fast(): faster tuning on pre-preprocessed data
- ks_threshold(): KS-statistic optimal threshold finder
- hyperparam_spaces: default search spaces for common classifiers
"""

from typing import Optional, Tuple, List, Dict, Any, Union, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
# optuna and TPESampler are imported lazily within functions to avoid heavy import-time dependencies.

from trustcredit.features.pipeline import create_pipeline


# Default hyperparameter search spaces for common classifiers
hyperparam_spaces: Dict[str, Dict[str, Any]] = {
    "LogisticRegression": {
        "C": {"low": 1e-3, "high": 1e3, "log": True, "type": "float"},
        "tol": {"low": 1e-6, "high": 1e-3, "log": True, "type": "float"},
        "class_weight": {"choices": [None, "balanced"], "type": "categorical"},
        "max_iter": {"low": 1000, "high": 1000, "step": 1, "type": "int"},
        "penalty": {"choices": ["l1", "l2"], "type": "categorical"},
        "solver": {"choices": ["liblinear"], "type": "categorical"},
    },
    "RandomForestClassifier": {
        "n_estimators": {"low": 5, "high": 250, "type": "int"},
        "max_depth": {"low": 2, "high": 12, "type": "int"},
        "criterion": {"choices": ["gini", "entropy"], "type": "categorical"},
        "min_samples_split": {"low": 2, "high": 256, "type": "int"},
        "min_samples_leaf": {"low": 1, "high": 256, "type": "int"},
        "max_features": {"low": 0.1, "high": 1.0, "type": "float"},
        "class_weight": {"choices": [None, "balanced"], "type": "categorical"},
    },
    "LGBMClassifier": {
        "n_estimators": {"low": 5, "high": 250, "type": "int"},
        "learning_rate": {"low": 0.05, "high": 1.0, "type": "float"},
        "num_leaves": {"low": 4, "high": 256, "type": "int"},
        "max_depth": {"low": 2, "high": 12, "type": "int"},
        "min_child_samples": {"low": 1, "high": 256, "type": "int"},
        "colsample_bytree": {"low": 0.1, "high": 1.0, "type": "float"},
        "reg_alpha": {"low": 1e-3, "high": 1e3, "log": True, "type": "float"},
        "reg_lambda": {"low": 1e-3, "high": 1e3, "log": True, "type": "float"},
        "class_weight": {"choices": [None, "balanced"], "type": "categorical"},
        "verbose": {"choices": [-1], "type": "categorical"},
    },
    "MLPClassifier": {
        "hidden_layer_sizes": {
            "choices": [[32], [64], [128], [32, 16], [64, 32], [64, 32, 16]],
            "type": "categorical",
        },
        "learning_rate_init": {"low": 0.001, "high": 0.1, "type": "float", "log": True},
        "learning_rate_decay_rate": {"low": 0.1, "high": 1, "type": "float"},
        "alpha": {"low": 1e-3, "high": 1e3, "type": "float", "log": True},
        "epochs": {"low": 10, "high": 100, "type": "int", "step": 10},
        "class_weight": {"choices": [None, "balanced"], "type": "categorical"},
        "batch_size": {"low": 128, "high": 128, "type": "int"},
    },
}


def _sample_params(trial: "optuna.trial.Trial", param_space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample hyperparameters from a search space using an Optuna trial.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Active Optuna trial object.
    param_space : dict
        Search space definition with 'type' keys and bounds/choices.

    Returns
    -------
    dict
        Sampled hyperparameter dictionary.
    """
    params = {}
    for name, values in param_space.items():
        ptype = values["type"]
        values_cp = {k: v for k, v in values.items() if k != "type"}
        if ptype == "int":
            params[name] = trial.suggest_int(name, **values_cp)
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, **values_cp)
        elif ptype == "float":
            params[name] = trial.suggest_float(name, **values_cp)
    return params


def objective(
    trial: "optuna.trial.Trial",
    model_class: BaseEstimator,
    pipeline_params: Dict[str, Any],
    fit_params: Dict[str, Any],
    score_func: Union[str, Callable],
    param_space: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[pd.Series],
    cv: int,
    seed_number: Optional[int],
) -> float:
    """Optuna objective for pipeline-level hyperparameter optimization.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Active Optuna trial.
    model_class : BaseEstimator class
        Classifier class to instantiate (not instance).
    pipeline_params : dict
        Keyword arguments forwarded to create_pipeline().
    fit_params : dict
        Keyword arguments forwarded to pipeline.fit().
    score_func : str or callable
        Scoring function. Use "roc_auc" or a callable with signature (y_true, y_score).
    param_space : dict
        Hyperparameter search space.
    X_train, y_train : array-like
        Training data.
    X_val, y_val : array-like or None
        Validation data. If None, uses cross-validation.
    cv : int
        Number of CV folds when no validation set is provided.
    seed_number : int or None
        Random seed for reproducibility.

    Returns
    -------
    float
        Validation score.
    """
    params = _sample_params(trial, param_space)
    params["random_state"] = seed_number

    if score_func == "roc_auc":
        score_func = roc_auc_score

    if X_val is not None and y_val is not None:
        model = create_pipeline(X_train, y_train, model_class(**params), **pipeline_params)
        model.fit(X_train, y_train, **fit_params)
        predictions = model.predict_proba(X_val)[:, 1]
        return score_func(y_val, predictions)
    else:
        if fit_params:
            raise ValueError("fit_params can only be used with an explicit validation set.")
        scores = []
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed_number)
        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = create_pipeline(X_tr, y_tr, model_class(**params), **pipeline_params)
            model.fit(X_tr, y_tr)
            scores.append(score_func(y_v, model.predict_proba(X_v)[:, 1]))
        return float(np.mean(scores))


def optimize_model(
    model_class: BaseEstimator,
    param_space: Union[Dict[str, Dict[str, Any]], str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    cv: int = 5,
    pipeline_params: Dict[str, Any] = {},
    fit_params: Dict[str, Any] = {},
    score_func: str = "roc_auc",
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
    seed_number: Optional[int] = None,
    n_jobs: int = 1,
) -> Tuple[Dict[str, Any], Pipeline]:
    """Optimize model hyperparameters via full pipeline tuning with Optuna.

    Searches for the best hyperparameters by rebuilding and re-fitting the
    complete pipeline on each trial. Supports both explicit validation sets
    and stratified cross-validation.

    Parameters
    ----------
    model_class : class
        Sklearn-compatible classifier class.
    param_space : dict or "suggest"
        Search space, or "suggest" to use built-in defaults.
    X_train, y_train : DataFrame/Series
        Training data.
    X_val, y_val : DataFrame/Series, optional
        Validation data. If omitted, uses CV.
    cv : int
        Cross-validation folds, used when no validation set is given.
    pipeline_params : dict
        Extra kwargs for create_pipeline().
    fit_params : dict
        Extra kwargs for pipeline.fit().
    score_func : str
        Metric to maximize ("roc_auc" supported by default).
    n_trials : int, optional
        Number of Optuna trials.
    timeout : int, optional
        Maximum optimization time in seconds.
    seed_number : int, optional
        Global random seed.
    n_jobs : int
        Parallel jobs for Optuna.

    Returns
    -------
    (dict, Pipeline)
        Best hyperparameters and trained pipeline with those params.
    """
    if param_space == "suggest":
        name = model_class.__name__
        if name not in hyperparam_spaces:
            raise ValueError(f"No default search space for '{name}'. Provide param_space explicitly.")
        param_space = hyperparam_spaces[name]

    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed_number))
    study.optimize(
        lambda trial: objective(
            trial, model_class, pipeline_params, fit_params, score_func,
            param_space, X_train, y_train, X_val, y_val, cv, seed_number,
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        n_jobs=n_jobs,
    )

    best_params = study.best_params
    model = create_pipeline(
        X_train, y_train,
        model_class(random_state=seed_number, **best_params),
        **pipeline_params,
    )
    model.fit(X_train, y_train)
    return best_params, model


def optimize_model_fast(
    model_class: BaseEstimator,
    param_space: Union[Dict[str, Dict[str, Any]], str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    pipeline_params: Dict[str, Any] = {},
    fit_params: Dict[str, Any] = {},
    score_func: str = "roc_auc",
    n_trials: Optional[int] = None,
    timeout: Optional[int] = None,
    seed_number: Optional[int] = None,
    n_jobs: int = 1,
) -> Tuple[Dict[str, Any], Pipeline]:
    """Faster hyperparameter tuning by pre-processing data once before optimization.

    Preprocesses the training and validation data once using create_pipeline,
    then runs Optuna only on the raw classifier (skipping pipeline rebuild on
    each trial). Significantly faster for large datasets or many trials.

    Parameters
    ----------
    model_class : class
        Sklearn-compatible classifier class.
    param_space : dict or "suggest"
        Search space, or "suggest" to use built-in defaults.
    X_train, y_train : DataFrame/Series
        Training data.
    X_val, y_val : DataFrame/Series
        Validation data (required for this fast variant).
    pipeline_params : dict
        Extra kwargs for create_pipeline().
    fit_params : dict
        Extra kwargs for classifier.fit() (without pipeline prefix).
    score_func : str or callable
        Metric to maximize.
    n_trials : int, optional
        Number of Optuna trials.
    timeout : int, optional
        Maximum optimization time in seconds.
    seed_number : int, optional
        Random seed.
    n_jobs : int
        Parallel jobs.

    Returns
    -------
    (dict, Pipeline)
        Best params and final trained pipeline.
    """
    if param_space == "suggest":
        name = model_class.__name__
        if name not in hyperparam_spaces:
            raise ValueError(f"No default search space for '{name}'.")
        param_space = hyperparam_spaces[name]

    # Pre-process once
    preprocess = create_pipeline(X_train, y_train, None, **pipeline_params)
    preprocess.fit(X_train, y_train)
    X_tr_pre = preprocess[:-1].transform(X_train)
    X_val_pre = preprocess[:-1].transform(X_val)

    # Strip pipeline__ prefix from fit_params
    fit_params_clean = {k.split("__")[-1]: v for k, v in fit_params.items()}

    def _objective_fast(trial):
        params = _sample_params(trial, param_space)
        try:
            params["random_state"] = seed_number
            model = model_class(**params)
        except TypeError:
            del params["random_state"]
            model = model_class(**params)
        model.fit(X_tr_pre, y_train, **fit_params_clean)
        y_score = model.predict_proba(X_val_pre)[:, 1]
        if score_func == "roc_auc":
            return roc_auc_score(y_val, y_score)
        y_pred = y_score > ks_threshold(y_val, y_score)
        return score_func(y_val, y_pred, y_score)

    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed_number))
    study.optimize(_objective_fast, n_trials=n_trials, timeout=timeout, show_progress_bar=True, n_jobs=n_jobs)

    best_params = study.best_params
    try:
        best_params["random_state"] = seed_number
        model = create_pipeline(X_train, y_train, model_class(random_state=seed_number, **best_params), **pipeline_params)
        model.fit(X_train, y_train, **fit_params)
    except TypeError:
        del best_params["random_state"]
        model = create_pipeline(X_train, y_train, model_class(**best_params), **pipeline_params)
        model.fit(X_train, y_train, **fit_params)

    return best_params, model


def ks_threshold(
    y_true: Union[np.ndarray, pd.Series],
    y_score: Union[np.ndarray, pd.Series],
) -> float:
    """Find the classification threshold that maximizes the KS statistic.

    The Kolmogorov-Smirnov statistic measures the maximum separation between
    the cumulative distributions of scores for positive and negative classes.
    This threshold is commonly used in credit scoring as an alternative to 0.5.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probability scores for the positive class.

    Returns
    -------
    float
        Threshold value that maximizes (TPR - FPR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return float(thresholds[np.argmax(tpr - fpr)])
