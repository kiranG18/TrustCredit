# Dependency Report

This report documents the modernized installation configuration specified in `pyproject.toml`.

---

## 1. Core Dependencies (Minimal Installation)
The minimal install contains only the standard scientific computing and machine learning libraries:
- **`pandas`**: Data manipulation and DataFrame structure.
- **`numpy`**: Multidimensional array math.
- **`scikit-learn`**: Pipeline building, scaling, metric calculations, and base estimators.
- **`lightgbm`**: Core gradient boosting engine (used for baseline and reject inference).
- **`matplotlib`**: Base plotting engine.
- **`gdown`**: Automatic dataset downloader from Google Drive.
- **`chardet`**: Automatic character encoding detection for raw CSVs.
- **`xxhash`**: Hashing functions.

To install the core package only, run:
```bash
pip install -e .
```

---

## 2. Optional Dependencies (Advanced Extensions)
Advanced extensions are categorized into dedicated installation groups:

### A. `tuning`
- **Packages**: `optuna`
- **Purpose**: Hyperparameter optimization search.
- **Install command**:
  ```bash
  pip install -e ".[tuning]"
  ```

### B. `fairness`
- **Packages**: `fairgbm`
- **Purpose**: In-processing fairness models.
- **Install command**:
  ```bash
  pip install -e ".[fairness]"
  ```

### C. `xai`
- **Packages**: `shap`, `lime`, `dice-ml`
- **Purpose**: Local and global explainability and counterfactual recourse.
- **Install command**:
  ```bash
  pip install -e ".[xai]"
  ```

### D. `dev`
- **Packages**: `pytest`, `black`, `ruff`, `jupyterlab`
- **Purpose**: Running unit tests, linting, formatting checks, and notebook development.
- **Install command**:
  ```bash
  pip install -e ".[dev]"
  ```

### Full Installation
To install the complete repository with all dependencies, run:
```bash
pip install -e ".[dev,tuning,fairness,xai]"
```
