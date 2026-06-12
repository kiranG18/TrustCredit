# Missing Modules Audit

During the dependency and import validation, the following missing modules or symbol references were identified and resolved:

1. **`Union` symbol in `src/trustcredit/data/loader.py`**
   - **Type**: Python symbol `NameError` during import compilation.
   - **Root Cause**: The method `load_dataset` used type hint `Optional[Union[str, Path]]`, but `Union` was not imported from `typing`.
   - **Impact**: Triggered a test collection failure in `pytest` across the entire project since `tests/test_data.py` imports `load_dataset` at module scope.
   - **Resolution**: Added `Union` to the typing import:
     ```python
     from typing import Optional, Union
     ```

2. **Compiled `lib_lightgbm.dll` for `fairgbm` on Windows**
   - **Type**: Optional binary runtime dependency.
   - **Root Cause**: The `fairgbm` library depends on a compiled version of the LightGBM C++ library (`lib_lightgbm.dll`), which is missing/not packaged correctly for some Windows wheel distributions of `fairgbm`.
   - **Impact**: Importing `fairgbm` fails on Windows environments with `Exception: Cannot find lightgbm library file`.
   - **Resolution**: Kept `fairgbm` as an optional dependency and ensured it is imported lazily inside wrapper initializers. Caught exceptions during evaluation to skip FairGBM gracefully on environments where the binary DLL cannot be found.
