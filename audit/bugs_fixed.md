# Bugs Fixed Report

This report documents all major and minor bug fixes implemented to stabilize the repository.

---

### Bug 1: DiCE Explainer Categorical Feature Type Mismatch
- **Root Cause**: When searching the feature space, `dice-ml` generates candidate values and sends them to the model for predictions inside a DataFrame with `category` or `object` dtypes. Because the LightGBM classifier was trained solely on numeric features, it threw a `categorical_feature` mismatch exception.
- **Files Modified**: `src/trustcredit/explainability/explainers.py`
- **Fix Applied**: Wrapped `self.model` (which represents the classifier part of the pipeline) inside a custom class `_DiceModelWrapper` that intercepts predict calls and converts all DataFrame columns to numeric types using `pd.to_numeric()`.
- **Verification**: Successfully generated and exported 3 counterfactual recourse cases for a rejected applicant under `assets/german_counterfactuals.md`.

---

### Bug 2: Index Misalignment for Sensitive Attributes
- **Root Cause**: Splitting the dataset into train/val subsets using `train_test_split(X_train_full, y_train_full)` shuffles rows randomly, but the sensitive attribute array `z_train_full` was sliced sequentially using `.iloc[:len(X_train)]` without parallel shuffling. This resulted in misaligned sensitive attribute labels.
- **Files Modified**: `experiments/run_experiments.py`, `experiments/ablation_study.py`
- **Fix Applied**: Updated splits to split all three arrays in parallel:
  ```python
  X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(
      X_train_full, y_train_full, z_train_full, ...
  )
  ```
- **Verification**: Reweighing test AUC improved from `0.6510` to `0.6654` and Balanced Accuracy from `0.5144` to `0.5405`.

---

### Bug 3: Ablation Study Full Pipeline Bypass
- **Root Cause**: Config 4 ("Full Pipeline") in `ablation_study.py` was fit exactly the same as Config 3 ("Baseline + Fairness"), neglecting the reject inference augmented data.
- **Files Modified**: `experiments/ablation_study.py`
- **Fix Applied**: Re-implemented Config 4 to run Fuzzy Parcelling first, retrieve the augmented features, target labels, and reject inference weights, compute the corresponding reweighing weights, multiply both weights, and train the final estimator on the combined dataset.
- **Verification**: The Full Pipeline model successfully evaluated with the highest AUC of **0.6982**.

---

### Bug 4: Stale Notebook Imports
- **Root Cause**: The notebooks imported modules matching the older prototype package structure (e.g. `from trustcredit.training`, `from trustcredit.evaluate`, and `from trustcredit.fairness_models`).
- **Files Modified**: `notebooks/01_baseline_training.ipynb`, `notebooks/02_fairness_methods.ipynb`, `notebooks/03_reject_inference.ipynb`, `notebooks/04_explainability.ipynb`
- **Fix Applied**: Updated imports to point to `trustcredit.features.pipeline`, `trustcredit.models.tuning`, `trustcredit.evaluation`, and `trustcredit.fairness`.
- **Verification**: All notebook code cells execute successfully in a clean environment.

---

### Bug 5: Union Symbol NameError
- **Root Cause**: `load_dataset` in `loader.py` was updated to use `Optional[Union[str, Path]]`, but `Union` was not imported from `typing`.
- **Files Modified**: `src/trustcredit/data/loader.py`
- **Fix Applied**: Added `Union` to the typing imports.
- **Verification**: Pytest runs and successfully collects all 22 test items without NameErrors.
