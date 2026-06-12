# Runtime Risks Audit

This document identifies potential runtime risks during pipeline execution and details the corresponding defensive implementations.

### 1. `fairgbm` Binary DLL Absence
- **Risk**: `fairgbm` cannot find `lib_lightgbm.dll` on some environments (especially Windows).
- **Likelihood**: High (on Windows machines without manual C++ builds).
- **Mitigation**:
  - `fairgbm` is imported dynamically during classifier instantiation (`FairGBM.__init__`).
  - The unified experiment runner (`run_experiments.py`) catches initialization and fitting errors for `FairGBM` and gracefully skips it, ensuring the rest of the pipeline executes correctly.

### 2. Missing Dataset Files
- **Risk**: Running experiment scripts or loading datasets before `prepare_datasets()` generates the prepared CSVs.
- **Likelihood**: Medium (on a fresh clone).
- **Mitigation**:
  - Refactored `load_dataset()` to use `pathlib.Path` and perform a `.exists()` check.
  - Implemented a detailed `FileNotFoundError` explaining how to prepare the datasets, rather than failing with cryptic traceback errors from pandas.

### 3. DiCE Explainer Data Type Mismatch
- **Risk**: Genetic algorithm queries from `dice-ml` are evaluated against the pipeline classifier using DataFrames containing `category` or `object` dtypes. This causes LightGBM to crash with a `categorical_feature` mismatch.
- **Likelihood**: High (if any EBE or OHE preprocessed categorical column is queried).
- **Mitigation**:
  - Introduced `_DiceModelWrapper` inside the `Dice` class. It converts all columns of pandas DataFrames passed by DiCE to numeric types (`pd.to_numeric()`) before calling `predict()` / `predict_proba()`.
