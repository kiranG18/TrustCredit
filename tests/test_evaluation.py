import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from trustcredit.evaluation.metrics import (
    demographic_parity,
    equal_opportunity,
    average_odds,
    average_precision_value_difference,
    geometric_mean_accuracy,
    kickout,
    get_threshold_bad_rate,
    get_metrics,
    get_fairness_metrics,
    get_reject_inference_metrics,
    create_eod_scorer,
    create_fairness_scorer
)
from trustcredit.models.tuning import ks_threshold

def test_fairness_metrics_basic():
    y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Privileged (z=1): pred is [0, 0, 1, 1], mean is 0.5
    # Unprivileged (z=0): pred is [0, 1, 0, 1], mean is 0.5
    # DPD should be P(Y_hat=1|Z=1) - P(Y_hat=1|Z=0) = 0.5 - 0.5 = 0.0
    dp = demographic_parity(y_true, y_pred, z)
    assert pytest.approx(dp) == 0.0
    
    # Privileged (z=1): true [1, 1], pred [1, 1] -> TPR = 1.0
    # Unprivileged (z=0): true [1, 1], pred [0, 1] -> TPR = 0.5
    # EOD should be TPR(Z=1) - TPR(Z=0) = 1.0 - 0.5 = 0.5
    eo = equal_opportunity(y_true, y_pred, z)
    assert pytest.approx(eo) == 0.5
    
    ao = average_odds(y_true, y_pred, z)
    assert isinstance(ao, float)
    
    apvd = average_precision_value_difference(y_true, y_pred, z)
    assert isinstance(apvd, float)
    
    gma = geometric_mean_accuracy(y_true, y_pred, z)
    assert isinstance(gma, float)

def test_kickout():
    y_true = np.array([1, 1, 0, 0])
    y_pred_base = np.array([0, 0, 0, 0])
    y_pred_reject = np.array([1, 0, 1, 0])
    
    # base missed bad: y_true == 1 and y_pred_base == 0 -> indices [0, 1]
    # reject correct on those: y_pred_reject[[0, 1]] -> [1, 0] -> mean is 0.5
    # base correct good: y_true == 0 and y_pred_base == 0 -> indices [2, 3]
    # reject wrong (rejections): y_pred_reject[[2, 3]] -> [1, 0] -> mean is 0.5
    # kickout should be 0.5 - 0.5 = 0.0
    ko = kickout(y_true, y_pred_base, y_pred_reject)
    assert pytest.approx(ko) == 0.0

def test_get_threshold_bad_rate():
    y_true = np.array([0, 0, 1, 1])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9])
    
    thresh = get_threshold_bad_rate(y_true, y_probs, bad_rate=0.5)
    assert isinstance(thresh, float)
    assert thresh >= 0.0 and thresh <= 1.0

def test_ks_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.3, 0.8, 0.9])
    
    thresh = ks_threshold(y_true, y_score)
    assert isinstance(thresh, float)
    assert thresh >= 0.0 and thresh <= 1.0

def test_get_metrics():
    np.random.seed(42)
    X = np.random.randn(20, 2)
    y = np.random.randint(0, 2, size=20)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    models = {"lr": model}
    df_metrics = get_metrics(models, X, y)
    assert isinstance(df_metrics, pd.DataFrame)
    assert "AUC" in df_metrics.columns
    assert "F1" in df_metrics.columns

def test_get_fairness_metrics():
    np.random.seed(42)
    X = np.random.randn(20, 2)
    y = np.random.randint(0, 2, size=20)
    z = np.random.randint(0, 2, size=20)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    models = {"lr": model}
    df_fair = get_fairness_metrics(models, X, y, z)
    assert isinstance(df_fair, pd.DataFrame)
    assert "DPD" in df_fair.columns
    assert "EOD" in df_fair.columns

def test_get_reject_inference_metrics():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    y_probs_base = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9])
    y_probs_unl = np.array([0.15, 0.25, 0.75, 0.85, 0.15, 0.25, 0.75, 0.85])
    
    model_dict = {
        "base": (y_probs_base, y_probs_unl),
        "model1": (y_probs_base * 0.9 + 0.05, y_probs_unl * 0.9 + 0.05)
    }
    
    df_metrics = get_reject_inference_metrics(model_dict, y, bad_rate=0.5)
    assert isinstance(df_metrics, pd.DataFrame)
    assert "approval_rate" in df_metrics.columns
    assert "kickout" in df_metrics.columns

def test_scorer_factories():
    z = np.array([0, 0, 1, 1])
    eod_scorer = create_eod_scorer(z)
    fair_scorer = create_fairness_scorer(fairness_goal=0.1, z=z)
    
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    
    eod_val = eod_scorer(y_true, y_pred)
    fair_val = fair_scorer(y_true, y_pred, y_score)
    
    assert isinstance(eod_val, float)
    assert isinstance(fair_val, float)
