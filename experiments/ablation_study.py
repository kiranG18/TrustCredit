"""
TrustCredit Ablation Study
============================

Systematically measures the contribution of each pipeline component:

  Config 1: Baseline only           (LightGBM, no corrections)
  Config 2: Baseline + Reject Inf.  (adds Fuzzy Parcelling)
  Config 3: Baseline + Fairness     (adds Reweighing)
  Config 4: Full Pipeline           (Reject Inf. + Reweighing)

Generates a comparison table and a visual ablation plot saved to assets/.

Usage:
    python experiments/ablation_study.py --dataset german --seed 42
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
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
from trustcredit.models.tuning import ks_threshold
from trustcredit.reject_inference.methods import FuzzyParcelling
from trustcredit.fairness.models import Reweighing
from trustcredit.evaluation.metrics import get_metrics, get_fairness_metrics


ABLATION_CONFIGS = [
    "Baseline Only",
    "Baseline + Reject Inference",
    "Baseline + Fairness",
    "Full Pipeline",
]

COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def encode_sensitive(df, col, pos_val):
    return (df[col] == pos_val).astype(int)


def simulate_rejects(X, y, z, fraction, seed):
    n = int(len(X) * fraction)
    idx = np.random.choice(len(X), size=n, replace=False)
    mask = np.zeros(len(X), dtype=bool)
    mask[idx] = True
    
    X_acc = X[~mask].reset_index(drop=True)
    y_acc = y[~mask].reset_index(drop=True)
    z_acc = z[~mask].reset_index(drop=True)
    X_rej = X[mask].reset_index(drop=True)
    z_rej = z[mask].reset_index(drop=True)
    
    return X_acc, y_acc, z_acc, X_rej, z_rej


def run_ablation(dataset_name: str, seed: int, config_path: str) -> None:
    cfg = load_config(config_path)
    ds_cfg = next(d for d in cfg["datasets"] if d["name"] == dataset_name)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  TrustCredit Ablation Study")
    print(f"  Dataset: {dataset_name}  |  Seed: {seed}")
    print(f"{'='*60}\n")

    os.makedirs("results", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    df = load_dataset(dataset_name, data_path=cfg["paths"]["data_prepared"])
    X = df.drop(columns=[ds_cfg["target_col"], ds_cfg["sensitive_col"]])
    y = df[ds_cfg["target_col"]].astype(int)
    z = encode_sensitive(df, ds_cfg["sensitive_col"], ds_cfg["sensitive_positive"])

    X_train_full, X_test, y_train_full, y_test, z_train_full, z_test = train_test_split(
        X, y, z, test_size=cfg["split"]["test_size"], stratify=y, random_state=seed
    )

    X_acc, y_acc, z_acc, X_rej, z_rej = simulate_rejects(
        X_train_full, y_train_full, z_train_full, ds_cfg["reject_fraction"], seed
    )

    # Shared preprocessor (fit on accepted data)
    preprocess = create_pipeline(X_acc, y_acc, None, normalize=False, do_EBE=True, onehot=False)
    preprocess.fit(X_acc, y_acc)
    X_acc_pre = preprocess[:-1].transform(X_acc)
    X_rej_pre = preprocess[:-1].transform(X_rej)
    X_test_pre = preprocess[:-1].transform(X_test)

    results_list = []

    # ── Config 1: Baseline Only ──
    print("[1/4] Baseline Only...")
    baseline = LGBMClassifier(random_state=seed, verbose=-1, n_estimators=100)
    baseline.fit(X_acc_pre, y_acc)

    # ── Config 2: Baseline + Reject Inference ──
    print("[2/4] Baseline + Reject Inference...")
    base_ri = LGBMClassifier(random_state=seed, verbose=-1, n_estimators=100)
    reject_est = LGBMClassifier(random_state=seed, verbose=-1, n_estimators=50)
    ri_model = FuzzyParcelling(base_estimator=base_ri, reject_estimator=reject_est)
    y_comb = np.concatenate([y_acc.values, -1 * np.ones(len(X_rej_pre))])
    X_comb = pd.concat([X_acc_pre, X_rej_pre], ignore_index=True)
    ri_model.fit(X_comb, y_comb)

    # ── Config 3: Baseline + Fairness (Reweighing) ──
    print("[3/4] Baseline + Fairness...")
    base_rw = LGBMClassifier(random_state=seed, verbose=-1, n_estimators=100)
    rw_model = Reweighing(estimator=base_rw)
    rw_model.fit(X_acc_pre, y_acc, sensitive_attributes=z_acc.values)

    # ── Config 4: Full Pipeline (RI + Reweighing) ──
    print("[4/4] Full Pipeline...")
    # Step 1: Reject Inference via Fuzzy Parcelling
    base_ri_full = LGBMClassifier(random_state=seed, verbose=-1, n_estimators=100)
    reject_est_full = LGBMClassifier(random_state=seed, verbose=-1, n_estimators=50)
    ri_full = FuzzyParcelling(base_estimator=base_ri_full, reject_estimator=reject_est_full)
    ri_full.fit(X_comb, y_comb)
    
    # Step 2: Extract Fuzzy Parcelling augmented dataset and sample weights
    X_aug, y_aug, weights_ri = ri_full._preprocess(X_acc_pre.values, y_acc.values, X_rej_pre.values)
    
    # Build corresponding sensitive attributes for augmented dataset [z_acc, z_rej, z_rej]
    z_aug = np.concatenate([z_acc.values, z_rej.values, z_rej.values])
    
    # Step 3: Compute Reweighing weights on the augmented dataset
    rw_full_prep = Reweighing(estimator=LGBMClassifier())
    _, _, weights_rw = rw_full_prep._preprocess(X_aug, y_aug, z_aug)
    
    # Step 4: Multiply reject inference weights by fairness weights
    final_weights = weights_ri * weights_rw
    
    # Step 5: Fit the final estimator on the augmented dataset with combined weights
    rw_full = LGBMClassifier(random_state=seed, verbose=-1, n_estimators=100)
    rw_full.fit(X_aug, y_aug, sample_weight=final_weights)

    # ── Evaluate all configs ──
    configs = {
        "Baseline Only": baseline,
        "Baseline + Reject Inference": ri_model.base_estimator,
        "Baseline + Fairness": rw_model,
        "Full Pipeline": rw_full,
    }

    for name, model in configs.items():
        try:
            y_prob = model.predict_proba(X_test_pre)[:, 1]
            thresh = ks_threshold(y_test.values, y_prob)
            y_pred = (y_prob >= thresh).astype(int)

            from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score
            from trustcredit.evaluation.metrics import demographic_parity, equal_opportunity

            z_test_vals = z_test.values
            row = {
                "Configuration": name,
                "AUC": roc_auc_score(y_test, y_prob),
                "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred, zero_division=0),
                "|DPD|": abs(demographic_parity(y_test.values, y_pred, z_test_vals)),
                "|EOD|": abs(equal_opportunity(y_test.values, y_pred, z_test_vals)),
            }
            results_list.append(row)
            print(f"   {name}: AUC={row['AUC']:.3f}, |DPD|={row['|DPD|']:.3f}")
        except Exception as e:
            print(f"   {name} failed: {e}")

    results_df = pd.DataFrame(results_list).set_index("Configuration")
    results_df.to_csv(f"results/{dataset_name}_ablation.csv")
    print(f"\nAblation results saved to: results/{dataset_name}_ablation.csv")
    print(results_df.round(4).to_string())

    # ── Ablation Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(f"TrustCredit Ablation Study — {dataset_name.capitalize()} Dataset",
                 fontsize=14, fontweight="bold")

    metrics_to_plot = [("AUC", True), ("F1", True), ("|DPD|", False)]
    for ax, (metric, higher_better) in zip(axes, metrics_to_plot):
        ax.set_facecolor("#f8f9fa")
        values = results_df[metric].values
        bars = ax.bar(
            range(len(ABLATION_CONFIGS)), values,
            color=COLORS[:len(ABLATION_CONFIGS)], alpha=0.85, edgecolor="white", width=0.6,
        )
        ax.set_xticks(range(len(ABLATION_CONFIGS)))
        ax.set_xticklabels(
            [c.replace("Baseline + ", "Baseline\n+ ").replace("Full Pipeline", "Full\nPipeline")
             for c in ABLATION_CONFIGS],
            fontsize=9,
        )
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        direction = "Higher = Better" if higher_better else "Lower = Fairer"
        ax.set_title(f"{metric}\n({direction})", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(True, axis="y", alpha=0.3)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    plt.tight_layout()
    out_path = f"assets/{dataset_name}_ablation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Ablation plot saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrustCredit Ablation Study")
    parser.add_argument("--dataset", type=str, default="german",
                        choices=["german", "taiwan"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    run_ablation(args.dataset, args.seed, args.config)
