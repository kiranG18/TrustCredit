# TrustCredit Dependency Graph

The package features a clean modular structure. By migrating heavy library calls to lazy imports, we've untangled import-time dependencies so that minimal components can run without optional modules.

```mermaid
graph TD
    %% Package entry point
    TC[trustcredit] --> TC_Data[trustcredit.data]
    TC --> TC_Features[trustcredit.features]
    TC --> TC_Models[trustcredit.models]
    TC --> TC_RI[trustcredit.reject_inference]
    TC --> TC_Fairness[trustcredit.fairness]
    TC --> TC_Eval[trustcredit.evaluation]
    TC --> TC_XAI[trustcredit.explainability]

    %% Internal module dependencies
    TC_RI -->|inherits sklearn base| SK[scikit-learn]
    TC_Fairness -->|evaluation metrics| TC_Eval
    TC_Eval -->|ks_threshold| TC_Models

    %% Lazy Imports
    TC_Models -.->|lazy import| Optuna[optuna]
    TC_Models -.->|lazy import| TF[tensorflow/keras]
    
    TC_Fairness -.->|lazy import| FairGBM[fairgbm]
    
    TC_XAI -.->|lazy import| SHAP[shap]
    TC_XAI -.->|lazy import| LIME[lime]
    TC_XAI -.->|lazy import| DiCE[dice-ml]
    TC_XAI -.->|lazy import| CFM[cfmining]

    classDef lazy fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,stroke-dasharray: 5 5;
    class Optuna,TF,FairGBM,SHAP,LIME,DiCE,CFM lazy;
```

### Dependency Decoupling

- **Core Dependencies (Minimal Install)**:
  - `pandas` / `numpy` / `scikit-learn` / `matplotlib` / `gdown` / `chardet` / `xxhash`
  - Installs instantly and provides the dataset loader, preprocessors, metrics, and standard estimators.

- **Optional Features**:
  - `dev`: `pytest`, `black`, `ruff`, `jupyterlab`
  - `tuning`: `optuna` (loaded dynamically only during search)
  - `fairness`: `fairgbm` (loaded dynamically inside in-processing classifiers)
  - `xai`: `shap`, `lime`, `dice-ml` (loaded dynamically inside explainers)
