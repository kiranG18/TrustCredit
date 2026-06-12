# Test Report

This report summarizes the execution of the unit test suite after the package modernization and lazy import upgrades.

---

## Execution Summary

- **Tests Discovered / Collected**: 22
- **Tests Passed**: 21
- **Tests Failed**: 0
- **Tests Skipped**: 1 (skipped on platforms where `fairgbm` cannot find `lib_lightgbm.dll`)
- **Status**: **PASSING**

---

## Test Collection Breakdown

### 1. Data Layer (`tests/test_data.py`)
- **Collected**: 3
- **Passed**: 3
- **Coverage**: Verifies encoding-aware CSV reading and dataset configuration mapping.

### 2. Evaluation Metrics (`tests/test_evaluation.py`)
- **Collected**: 8
- **Passed**: 8
- **Coverage**: Verifies Demographics Parity, Equal Opportunity, Average Odds, kickout, and custom scorer factories.

### 3. Fairness Interventions (`tests/test_fairness.py`)
- **Collected**: 3
- **Passed**: 2
- **Skipped**: 1 (`test_fairgbm`, due to missing DLL on Windows wheel)
- **Coverage**: Verifies Reweighing and ThresholdOpt.

### 4. Reject Inference (`tests/test_reject_inference.py`)
- **Collected**: 8
- **Passed**: 8
- **Coverage**: Verifies all 6 reject inference algorithms (Upward, Downward, SoftCutoff, FuzzyParcelling, Extrapolation, Spreading).
