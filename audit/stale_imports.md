# Stale and Heavy Imports Audit

This audit identifies stale, redundant, or heavy import-time dependencies that caused performance or test suite collection risks.

1. **`optuna` imported at module-level in `tuning.py`**
   - **Type**: Heavy import-time dependency.
   - **Root Cause**: `optuna` and its samplers were imported at the top-level of `src/trustcredit/models/tuning.py`. Because `src/trustcredit/evaluation/__init__.py` re-exported `ks_threshold` by importing it from `trustcredit.models.tuning`, importing *any* evaluation metric transitively loaded `optuna` and its dependencies (`tqdm`, `sqlalchemy`, `alembic`, etc.).
   - **Impact**: Led to slow test collection and execution, and forced the requirement of Optuna even for minimal code verification.
   - **Resolution**: Converted `optuna` and `TPESampler` to lazy imports inside the methods that require them (`optimize_model` and `optimize_model_fast`). Used string literals (`"optuna.trial.Trial"`) for type annotations to bypass import-time reference checks.

2. **Jupyter Notebook Stale Imports**
   - **Type**: Stale references to deprecated package names.
   - **Root Cause**: The notebooks originally referenced the prototype package structure (e.g. `credit_pipeline.data`, `credit_pipeline.training`, etc.).
   - **Impact**: The notebooks could not be executed end-to-end in a clean environment.
   - **Resolution**: Updated all notebook cells to import from `trustcredit` matching the modernized structure.
