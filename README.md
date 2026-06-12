# TrustCredit: Responsible Credit Risk Modeling Pipeline

I built TrustCredit to address a critical, often ignored challenge in credit risk scoring: selection bias. 

When financial institutions build machine learning models to predict loan defaults, they typically train them only on approved applicants whose repayment outcomes are known. This creates a severe selection bias because the model never learns from the population of rejected applicants. 

TrustCredit is a comprehensive credit risk modeling pipeline that corrects selection bias using reject inference, enforces fairness constraints to prevent discrimination, and uses explainable AI (XAI) tools to make credit decisions transparent and actionable.

---

## The Selection Bias Problem

When a model is trained exclusively on accepted applications, it learns on a biased sample. If a bank uses this biased model to make future decisions, it can lead to suboptimal outcomes, such as missing creditworthy applicants or mispricing risk on borderline applicants.

### How Reject Inference Fixes It
Reject inference is the process of inferring the credit outcomes of rejected applicants using their feature distributions, allowing them to be safely reintroduced into the training population. TrustCredit implements six reject inference strategies:

* **RejectUpward:** Up-weights accepted applicants who lie close to the decision boundary (narrowly approved) to represent the missing rejected population.
* **RejectDownward:** Down-weights accepted applicants who are highly likely to be accepted, focusing the model's capacity on borderline cases.
* **RejectSoftCutoff:** Uses score bins to reweight accepted samples based on the acceptance rate of their respective score intervals.
* **FuzzyParcelling:** Duplicates rejected applicants, assigning one copy a good label and the other a bad label, weighted by their predicted probabilities.
* **RejectExtrapolation:** Predicts hard labels for the rejected population using a model trained on accepted data, then trains a new model on the combined data.
* **RejectSpreading:** Uses graph-based semi-supervised label spreading to propagate labels from accepted applicants to rejected applicants.

---

## Fairness-Aware Learning

Credit models must be fair and compliant with regulatory standards. TrustCredit integrates three stages of algorithmic fairness interventions:

* **Pre-processing (Reweighing):** Adjusts sample weights prior to training to eliminate statistical dependencies between sensitive attributes (like Gender or Age) and credit outcomes.
* **In-processing (FairGBM):** Integrates fairness constraints directly into the LightGBM objective function, balancing performance and demographic parity or equal opportunity.
* **Post-processing (ThresholdOpt):** Optimizes group-specific classification thresholds on a pre-trained model to meet strict fairness constraints.

---

## Explainability and Recourse

Credit decisions require transparency. TrustCredit provides:

* **Global and Local Explanations:** Full pipeline integration for SHAP and LIME, mapping feature importances back to the original scales.
* **Actionable Recourse (DiCE & MAPOCAM):** Generates minimal-cost counterfactual explanations showing exactly what a rejected applicant needs to change (like reducing loan duration or credit amount) to turn a rejection into an approval.

---

## Pipeline Architecture

Here is the block diagram of the TrustCredit decision pipeline:

```
[Raw Credit Data] ──> [ColumnTransformer] ──> [Reject Inference] ──> [Fairness Model] ──> [KS Threshold] ──> [XAI Explanations]
                     - Imputers               (Fuzzy Parcelling)     (Reweighing)        (Optimal Split)      - SHAP Waterfall
                     - Ordinal/One-Hot                                                                        - DiCE Recourse
                     - EBE Encoder
```

---

## Project Structure

* `src/trustcredit/`: Main Python package containing source modules:
  * `data/`: Data loading and preparation tools.
  * `features/`: Pipeline builder and Evidence-Based Encoding (EBE).
  * `models/`: Keras MLP wrapper and Optuna hyperparameter tuning.
  * `reject_inference/`: 6 reject inference algorithms.
  * `fairness/`: Reweighing, FairGBM, and ThresholdOpt.
  * `evaluation/`: Performance, fairness, and reject inference evaluation metrics.
  * `explainability/`: PDP, SHAP, LIME, MAPOCAM, and DiCE explainer wrappers.
* `notebooks/`: Structured Jupyter notebooks demonstrating baseline training, fairness, reject inference, and explainability.
* `config/`: Centralized configuration.
* `experiments/`: Unified experiment runner, ablation study framework, and artifact generator.
* `results/`: Directory for experiment tables and comparison metrics.
* `assets/`: Saved visual charts and diagrams.

---

## How to Run

### Installation

First, clone and install the required `cfmining` dependency from GitHub:
```bash
pip install git+https://github.com/visual-ds/cfmining.git
```

Then install the `trustcredit` package and its dependencies in editable mode:
```bash
pip install -e .
```

### Run Experiments
To train and compare baseline, reject inference, and fairness models:
```bash
python experiments/run_experiments.py --dataset german --seed 42
```

### Run Ablation Study
To run the ablation study measuring components' individual contributions:
```bash
python experiments/ablation_study.py --dataset german --seed 42
```

### Generate Artifacts
To generate all visual assets, SHAP waterfall plots, and recourse tables:
```bash
python experiments/generate_artifacts.py
```
