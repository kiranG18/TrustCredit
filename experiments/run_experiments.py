"""
TrustCredit Unified Experiment Runner
======================================

Trains and compares three model variants on a credit dataset:
  1. Baseline          -- LightGBM tuned with Optuna, no corrections
  2. Reject Inference  -- Fuzzy Parcelling on accepted + rejected data
  3. Fairness-Aware    -- Reweighing (pre-proc) + FairGBM (in-proc)

Results are saved to results/experiment_results.csv and plots to assets/.

Usage:
    python experiments/run_experiments.py --dataset german --seed 42
    python experiments/run_experiments.py --dataset taiwan --seed 42 --n_trials 30
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Make the src directory importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

from trustcredit.data.loader import load_dataset
from trustcredit.features.pipeline import create_pipeline
from trustcredit.models.tuning import optimize_model_fast, ks_threshold, hyperparam_spaces
from trustcredit.reject_inference.methods import FuzzyParcelling
from trustcredit.fairness.models import Reweighing, FairGBM
from trustcredit.evaluation.metrics import get_metrics, get_fairness_metrics


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def get_dataset_config(cfg: dict, dataset_name: str) -> dict:
    for ds in cfg["datasets"]:
        if ds["name"] == dataset_name:
            return ds
    raise ValueError(f"Dataset '{dataset_name}' not found in config.")


def encode_sensitive(df: pd.DataFrame, col: str, positive_val: str) -> pd.Series:
    """Encode a sensitive attribute as binary (1 = privileged group)."""
    return (df[col] == positive_val).astype(int)


def simulate_reject_population(
    X: pd.DataFrame,
    y: pd.Series,
    reject_fraction: float,
    seed: int,
) -> tuple:
    """Split labeled data to simulate accepted + rejected populations."""
    n_rejected = int(len(X) * reject_fraction)
    idx_rejected = np.random.choice(len(X), size=n_rejected, replace=False)
    mask_rejected = np.zeros(len(X), dtype=bool)
    mask_rejected[idx_rejected] = True

    X_accepted = X[~mask_rejected].reset_index(drop=True)
    y_accepted = y[~mask_rejected].reset_index(drop=True)
    X_rejected = X[mask_rejected].reset_index(drop=True)

    return X_accepted, y_accepted, X_rejected


def save_results_table(results: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    results.to_csv(path)
    print(f"\nResults saved to: {path}")
    print(results.round(4).to_string())


# ──────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────

def plot_roc_curves(models_dict: dict, X_test, y_test, save_path: str) -> None:
    from sklearn.metrics import roc_curve, auc

    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")

    for (name, model), color in zip(models_dict.items(), colors):
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — TrustCredit Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curves saved to: {save_path}")


def plot_metrics_comparison(perf_df: pd.DataFrame, fair_df: pd.DataFrame, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#ffffff")
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

    # Performance
    metrics_cols = ["AUC", "Precision", "Recall", "F1"]
    perf_plot = perf_df[metrics_cols].copy()
    x = np.arange(len(metrics_cols))
    width = 0.8 / len(perf_plot)
    ax = axes[0]
    ax.set_facecolor("#f8f9fa")
    for i, (model_name, row) in enumerate(perf_plot.iterrows()):
        offset = (i - len(perf_plot) / 2 + 0.5) * width
        bars = ax.bar(x + offset, row.values, width, label=model_name,
                      color=colors[i % len(colors)], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_cols, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Performance Metrics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # Fairness
    fair_cols = ["DPD", "EOD", "AOD"]
    fair_plot = fair_df[fair_cols].abs().copy()
    x2 = np.arange(len(fair_cols))
    ax2 = axes[1]
    ax2.set_facecolor("#f8f9fa")
    for i, (model_name, row) in enumerate(fair_plot.iterrows()):
        offset = (i - len(fair_plot) / 2 + 0.5) * width
        ax2.bar(x2 + offset, row.values, width, label=model_name,
                color=colors[i % len(colors)], alpha=0.85, edgecolor="white")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["Demo. Parity\n(|DPD|)", "Equal Opp.\n(|EOD|)", "Avg. Odds\n(|AOD|)"], fontsize=10)
    ax2.set_title("Fairness Metrics (lower = fairer)", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.axhline(0.05, color="red", ls="--", lw=1.5, alpha=0.7, label="Fairness threshold")
    ax2.grid(True, axis="y", alpha=0.3)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    plt.suptitle("TrustCredit — Performance vs Fairness Trade-off", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


# ──────────────────────────────────────────────────────────
# Main experiment pipeline
# ──────────────────────────────────────────────────────────

def run_experiments(dataset_name: str, seed: int, n_trials: int, config_path: str) -> None:
    cfg = load_config(config_path)
    ds_cfg = get_dataset_config(cfg, dataset_name)
    set_seed(seed)

    print(f"\n{'='*60}")
    print(f"  TrustCredit Experiment Runner")
    print(f"  Dataset : {dataset_name}")
    print(f"  Seed    : {seed}")
    print(f"  Trials  : {n_trials}")
    print(f"{'='*60}\n")

    os.makedirs("results", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    # ── Load data ──
    df = load_dataset(dataset_name, data_path=cfg["paths"]["data_prepared"])
    target_col = ds_cfg["target_col"]
    sensitive_col = ds_cfg["sensitive_col"]

    X = df.drop(columns=[target_col, sensitive_col])
    y = df[target_col].astype(int)
    z = encode_sensitive(df, sensitive_col, ds_cfg["sensitive_positive"])

    X_train_full, X_test, y_train_full, y_test, z_train_full, z_test = train_test_split(
        X, y, z, test_size=cfg["split"]["test_size"], stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(
        X_train_full, y_train_full, z_train_full,
        test_size=cfg["split"]["val_size"],
        stratify=y_train_full, random_state=seed,
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── 1. Baseline ──
    print("\n[1/3] Training Baseline (LightGBM + Optuna)...")
    param_space = {k: v for k, v in hyperparam_spaces["LGBMClassifier"].items() if k != "verbose"}
    best_params, baseline_model = optimize_model_fast(
        LGBMClassifier,
        param_space,
        X_train, y_train, X_val, y_val,
        pipeline_params={"normalize": False, "do_EBE": True, "onehot": False},
        score_func="roc_auc",
        n_trials=n_trials,
        seed_number=seed,
    )
    print(f"   Best params: {best_params}")

    # ── 2. Reject Inference ──
    print("\n[2/3] Training Reject Inference (Fuzzy Parcelling)...")
    X_acc, y_acc, X_rej = simulate_reject_population(
        X_train, y_train, ds_cfg["reject_fraction"], seed=seed
    )

    preprocess = create_pipeline(X_acc, y_acc, None, normalize=False, do_EBE=True, onehot=False)
    preprocess.fit(X_acc, y_acc)
    X_acc_pre = preprocess[:-1].transform(X_acc)
    X_rej_pre = preprocess[:-1].transform(X_rej)

    base_lgbm = LGBMClassifier(random_state=seed, verbose=-1)
    reject_lgbm = LGBMClassifier(random_state=seed, verbose=-1)
    ri_model = FuzzyParcelling(base_estimator=base_lgbm, reject_estimator=reject_lgbm)

    y_combined = np.concatenate([y_acc.values, -1 * np.ones(len(X_rej_pre))])
    X_combined = pd.concat([X_acc_pre, X_rej_pre], ignore_index=True)
    ri_model.fit(X_combined, y_combined)
    print("   Fuzzy Parcelling trained.")

    # Wrap models to use the same preprocessor
    class _PipelineModelWrapper:
        def __init__(self, preprocess, model):
            self.preprocess = preprocess
            self.model = model
        def predict_proba(self, X):
            return self.model.predict_proba(self.preprocess[:-1].transform(X))
        def predict(self, X):
            return self.model.predict(self.preprocess[:-1].transform(X))

    ri_wrapper = _PipelineModelWrapper(preprocess, ri_model)

    # ── 3. Fairness-Aware ──
    print("\n[3/3] Training Fairness-Aware models...")
    # 3a. Reweighing
    rw_base = LGBMClassifier(random_state=seed, verbose=-1)
    rw_model = Reweighing(estimator=rw_base)
    X_train_pre = preprocess[:-1].transform(X_train)
    rw_model.fit(X_train_pre, y_train, sensitive_attributes=z_train.values)
    rw_wrapper = _PipelineModelWrapper(preprocess, rw_model)
    print("   Reweighing trained.")

    # 3b. FairGBM
    try:
        fgbm_model = FairGBM(
            constraint_type=cfg["fairgbm"]["constraint_type"],
            multiplier_learning_rate=cfg["fairgbm"]["multiplier_learning_rate"],
            n_estimators=cfg["fairgbm"]["n_estimators"],
            verbose=-1,
        )
        fgbm_model.fit(X_train_pre, y_train, sensitive_attributes=z_train.values)
        fgbm_wrapper = _PipelineModelWrapper(preprocess, fgbm_model)
        print("   FairGBM trained.")
    except Exception as e:
        print(f"   FairGBM skipped: {e}")
        fgbm_model = None

    # ── Evaluate ──
    print("\nEvaluating all models on test set...")
    X_test_pre = preprocess[:-1].transform(X_test)
    z_test_vals = z_test.values

    models = {
        "Baseline": baseline_model,
        "Reject Inference": ri_wrapper,
        "Reweighing": rw_wrapper,
    }
    if fgbm_model is not None:
        models["FairGBM"] = fgbm_wrapper

    # Performance table
    perf_results = get_metrics(models, X_test, y_test)
    fair_results = get_fairness_metrics(models, X_test, y_test, z_test_vals)

    combined = perf_results.join(fair_results[["DPD", "EOD", "AOD", "GMA"]])

    # Save
    save_results_table(combined, f"results/{dataset_name}_experiment_results.csv")
    perf_results.to_csv(f"results/{dataset_name}_performance.csv")
    fair_results.to_csv(f"results/{dataset_name}_fairness.csv")

    # ── Plots ──
    # ROC curves (only models with consistent preprocessor)
    plot_roc_curves(
        {"Baseline": baseline_model, "Reject Inference": ri_wrapper, "Reweighing": rw_wrapper},
        X_test, y_test,
        f"assets/{dataset_name}_roc_curves.png",
    )
    plot_metrics_comparison(
        perf_results.loc[["Baseline", "Reject Inference", "Reweighing"]],
        fair_results.loc[["Baseline", "Reject Inference", "Reweighing"]],
        f"assets/{dataset_name}_metrics_comparison.png",
    )

    print(f"\nDone! All results saved to results/ and assets/")


# ──────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrustCredit Experiment Runner")
    parser.add_argument("--dataset", type=str, default="german",
                        choices=["german", "taiwan", "homecredit"],
                        help="Dataset to run experiments on.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to config YAML file.")
    args = parser.parse_args()

    run_experiments(
        dataset_name=args.dataset,
        seed=args.seed,
        n_trials=args.n_trials,
        config_path=args.config,
    )
