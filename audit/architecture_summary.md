# Architecture Summary

TrustCredit implements a modular, extensible, and responsible credit risk modeling pipeline. 

Here is a summary of the data flow and package layer operations:

---

## 1. Data Layer (`trustcredit.data`)
- **Responsibilities**: Automatic character encoding detection for CSVs, dataset downloading (from Google Drive), dataset cleaning, schema casting, and loading.
- **Dtypes**: Restores string categorical indicators into pandas `Categorical` format on load to make column transformers robust.

---

## 2. Preprocessing & Features (`trustcredit.features`)
- **Pipeline Builder**: `create_pipeline` structures a standard sklearn `Pipeline` containing:
  - **Imputation**: Mean imputation for numerical columns; mode imputation for categorical columns.
  - **Label Encoding**: Numerical encoding of categoricals.
  - **EBE (Evidence-Based Encoding)**: High-cardinality categoricals are target-encoded using a smoothed target mean to avoid overfitting:
    $$\text{Encoded}(x) = \frac{\text{Mean}(y|x) \times \text{Count}(x) + k \times \text{GlobalMean}(y)}{\text{Count}(x) + k}$$
  - **Scaling**: Standard scaling for numeric columns.
  - **One-Hot Encoding**: Low-cardinality categoricals are one-hot encoded.

---

## 3. Models Layer (`trustcredit.models`)
- **Estimaors**: Wraps standard classifiers (like Keras-backed MLPs or LightGBM) with sklearn-compliant interfaces.
- **Tuning**: Optimizes model hyperparameters via TPESampler-driven Optuna search, fitting either the full pipeline (`optimize_model`) or preprocessed arrays (`optimize_model_fast`).
- **KS Threshold**: Finds the optimal decision threshold that maximizes the Kolmogorov-Smirnov (KS) statistic.

---

## 4. Reject Inference Layer (`trustcredit.reject_inference`)
- **Responsibilities**: Alleviates selection bias by inferring outcome labels or weights for rejected (unlabeled) credit applicants and augmenting the training set.
- **Algorithms**:
  - `RejectUpward` / `RejectDownward`: Up/down-weights borderline accepted applicants based on acceptance probability.
  - `RejectSoftCutoff`: Bins applicants by acceptance score and applies interval-based weights.
  - `FuzzyParcelling`: Duplicates rejected applicants, assigning one copy a good label and the other a bad label, weighted by predicted probability.
  - `RejectExtrapolation`: Generates hard labels using a model trained on accepted data.
  - `RejectSpreading`: propagates labels to rejected applicants using semi-supervised graph label propagation.

---

## 5. Fairness Layer (`trustcredit.fairness`)
- **Responsibilities**: Enforces demographic parity or equal opportunity constraints to prevent discriminatory credit decisions.
- **Interventions**:
  - **Pre-processing (Reweighing)**: Reweights training samples to eliminate dependencies between protected groups and outcome labels.
  - **In-processing (FairGBM)**: penalizes group-level performance violations directly inside the boosting loss.
  - **Post-processing (ThresholdOpt)**: Searches grid combinations of group-specific decision thresholds to maximize accuracy subject to fairness limits.

---

## 6. Explainability Layer (`trustcredit.explainability`)
- **Responsibilities**: Provides transparency and recourse for applicants.
- **Components**:
  - `PartialDependencePipeline`: Computes pipeline-aware PDP and ICE curves.
  - `ShapPipelineExplainer` / `LimePipelineExplainer`: Maps local feature importances back to original feature scales.
  - `Dice` / `MAPOCAM`: Generates actionable counterfactual recommendations (e.g. lowering loan duration to turn a rejection into an approval).

---

## 7. Evaluation Layer (`trustcredit.evaluation`)
- **Responsibilities**: Evaluates model performance and fairness.
- **Metrics**: DPD (Demographic Parity Difference), EOD (Equal Opportunity Difference), AOD (Average Odds Difference), GMA (Geometric Mean Accuracy), and Kickout metric for reject inference.
