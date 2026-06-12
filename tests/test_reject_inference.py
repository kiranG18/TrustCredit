import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from trustcredit.reject_inference.methods import (
    RejectUpward,
    RejectDownward,
    RejectSoftCutoff,
    FuzzyParcelling,
    RejectExtrapolation,
    RejectSpreading
)

@pytest.fixture
def synthetic_reject_data():
    np.random.seed(42)
    # Labeled data
    X_lab = np.random.randn(60, 4)
    y_lab = np.random.randint(0, 2, size=60)
    # Unlabeled data (marked -1)
    X_unl = np.random.randn(30, 4)
    y_unl = -1 * np.ones(30, dtype=int)
    
    X = np.concatenate([X_lab, X_unl])
    y = np.concatenate([y_lab, y_unl])
    
    # Also create as DataFrame to test feature name handling
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
    y_series = pd.Series(y)
    
    return X, y, X_df, y_series

@pytest.mark.parametrize("strategy_class", [
    RejectUpward,
    RejectDownward,
    RejectSoftCutoff,
    FuzzyParcelling,
    RejectSpreading
])
def test_reject_inference_strategies(synthetic_reject_data, strategy_class):
    X, y, X_df, y_series = synthetic_reject_data
    base = LogisticRegression(random_state=42)
    reject = LogisticRegression(random_state=42)
    
    strategy = strategy_class(base_estimator=base, reject_estimator=reject)
    
    # Test numpy array
    strategy.fit(X, y)
    probs = strategy.predict_proba(X[:10])
    preds = strategy.predict(X[:10])
    assert probs.shape == (10, 2)
    assert preds.shape == (10,)
    
    # Test pandas DataFrame
    strategy.fit(X_df, y_series)
    probs_df = strategy.predict_proba(X_df.iloc[:10])
    assert probs_df.shape == (10, 2)

@pytest.mark.parametrize("mode", ["positive", "all", "confident"])
def test_reject_extrapolation(synthetic_reject_data, mode):
    X, y, X_df, y_series = synthetic_reject_data
    base = LogisticRegression(random_state=42)
    reject = LogisticRegression(random_state=42)
    
    strategy = RejectExtrapolation(base_estimator=base, reject_estimator=reject, mode=mode)
    strategy.fit(X, y)
    probs = strategy.predict_proba(X[:10])
    assert probs.shape == (10, 2)
    
    # Test invalid mode
    with pytest.raises(ValueError, match="Unknown mode"):
        invalid = RejectExtrapolation(base_estimator=base, reject_estimator=reject, mode="invalid")
        invalid.fit(X, y)
