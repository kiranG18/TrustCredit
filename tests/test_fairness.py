import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from trustcredit.fairness.models import Reweighing, ThresholdOpt, FairGBM

@pytest.fixture
def synthetic_fairness_data():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, size=100)
    # Sensitive attribute (e.g., gender, binary)
    sensitive_attributes = np.random.randint(0, 2, size=100)
    
    # Intentionally correlate sensitive attribute with y for bias
    y[sensitive_attributes == 1] = np.random.choice([0, 1], size=np.sum(sensitive_attributes == 1), p=[0.2, 0.8])
    y[sensitive_attributes == 0] = np.random.choice([0, 1], size=np.sum(sensitive_attributes == 0), p=[0.8, 0.2])
    
    return X, y, sensitive_attributes

def test_reweighing(synthetic_fairness_data):
    X, y, sensitive_attributes = synthetic_fairness_data
    base = LogisticRegression(random_state=42)
    reweighing_model = Reweighing(estimator=base)
    
    reweighing_model.fit(X, y, sensitive_attributes)
    
    # Check that it fits and predicts
    probs = reweighing_model.predict_proba(X[:10])
    preds = reweighing_model.predict(X[:10])
    assert probs.shape == (10, 2)
    assert preds.shape == (10,)
    
    # Verify weights are computed
    _, _, weights = reweighing_model._preprocess(X, y, sensitive_attributes)
    assert weights.shape == (100,)
    assert np.all(weights >= 0)

def test_threshold_opt(synthetic_fairness_data):
    X, y, sensitive_attributes = synthetic_fairness_data
    base = DecisionTreeClassifier(random_state=42, max_depth=3)
    
    threshold_model = ThresholdOpt(
        estimator=base,
        perf_metric="balanced_accuracy",
        fair_metric="demographic_parity",
        constraint_value=0.1,
        n_thresholds=10
    )
    
    threshold_model.fit(X, y, sensitive_attributes)
    
    # Check thresholds are selected
    assert hasattr(threshold_model, "thresh0_")
    assert hasattr(threshold_model, "thresh1_")
    
    # Test predict
    preds = threshold_model.predict(X[:10], sensitive_attributes[:10])
    assert preds.shape == (10,)
    assert np.issubdtype(preds.dtype, np.bool_) or np.issubdtype(preds.dtype, np.integer)

def test_fairgbm(synthetic_fairness_data):
    # Skip if fairgbm is not installed or cannot be loaded
    try:
        import fairgbm
    except Exception:
        pytest.skip("fairgbm is not available or cannot be loaded on this platform")
    
    X, y, sensitive_attributes = synthetic_fairness_data
    
    fairgbm_model = FairGBM(n_estimators=5, random_state=42)
    fairgbm_model.fit(X, y, sensitive_attributes)
    
    probs = fairgbm_model.predict_proba(X[:10])
    preds = fairgbm_model.predict(X[:10])
    
    assert probs.shape == (10, 2)
    assert preds.shape == (10,)
