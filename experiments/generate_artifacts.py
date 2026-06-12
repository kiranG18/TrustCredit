"""
TrustCredit Artifacts Generator
===============================

Generates visual and tabular artifacts for the TrustCredit repository:
1. Pipeline Architecture Diagram (matplotlib box-and-arrow chart)
2. SHAP Local Explanation (waterfall bar chart for a rejected applicant)
3. Actionable Recourse (counterfactual comparison table using DiCE)

Saves all outputs to the assets/ directory.

Usage:
    python experiments/generate_artifacts.py
"""

import os
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Make the src directory importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trustcredit.data.loader import load_dataset
from trustcredit.features.pipeline import create_pipeline
from trustcredit.models.tuning import ks_threshold
from trustcredit.explainability.explainers import ShapPipelineExplainer, Dice, display_cfs
from lightgbm import LGBMClassifier


def draw_pipeline_architecture(save_path: str) -> None:
    """Draw a clean, professional architecture block diagram for the TrustCredit pipeline."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    
    # Define box styles
    box_blue = dict(boxstyle="round,pad=0.5", fc="#e2f0d9", ec="#385723", lw=1.5)
    box_green = dict(boxstyle="round,pad=0.5", fc="#deebf7", ec="#203764", lw=1.5)
    box_orange = dict(boxstyle="round,pad=0.5", fc="#fce4d6", ec="#c65911", lw=1.5)
    box_gray = dict(boxstyle="round,pad=0.5", fc="#f2f2f2", ec="#7f7f7f", lw=1.5)
    
    # Draw boxes
    ax.text(0.1, 0.8, "Raw Credit Data\n(German / Taiwan)", ha="center", va="center", bbox=box_blue, fontsize=11, fontweight="bold")
    ax.text(0.4, 0.8, "Pre-processing &\nFeature Engineering\n- Imputers\n- Ordinal/One-Hot\n- EBE Target Encoding", ha="center", va="center", bbox=box_blue, fontsize=10)
    
    ax.text(0.75, 0.8, "Reject Inference\n(Fuzzy Parcelling)\n- Infer unobserved\n  rejected outcomes", ha="center", va="center", bbox=box_green, fontsize=10)
    ax.text(0.75, 0.5, "Fairness Intervention\n- Pre-proc: Reweighing\n- In-proc: FairGBM\n- Post-proc: ThresholdOpt", ha="center", va="center", bbox=box_green, fontsize=10)
    
    ax.text(0.4, 0.5, "Model Training\n- LightGBM / MLP\n- Optuna Tuning\n- KS-Thresholding", ha="center", va="center", bbox=box_orange, fontsize=10, fontweight="bold")
    
    ax.text(0.1, 0.5, "Explainable AI\n- SHAP Local/Global\n- DiCE Counterfactuals\n- PDP/ICE Curves", ha="center", va="center", bbox=box_gray, fontsize=10)
    
    # Draw arrows
    arrow_args = dict(arrowstyle="->", lw=1.5, color="#2c3e50")
    
    # Raw Data -> Preprocessing
    ax.annotate("", xy=(0.26, 0.8), xytext=(0.21, 0.8), arrowprops=arrow_args)
    # Preprocessing -> Reject Inference
    ax.annotate("", xy=(0.62, 0.8), xytext=(0.54, 0.8), arrowprops=arrow_args)
    # Reject Inference -> Fairness Intervention
    ax.annotate("", xy=(0.75, 0.61), xytext=(0.75, 0.69), arrowprops=arrow_args)
    # Fairness Intervention -> Model Training
    ax.annotate("", xy=(0.54, 0.5), xytext=(0.62, 0.5), arrowprops=arrow_args)
    # Model Training -> XAI
    ax.annotate("", xy=(0.22, 0.5), xytext=(0.3, 0.5), arrowprops=arrow_args)
    
    # Add title and descriptions
    plt.title("TrustCredit: Responsible Credit Risk Pipeline Architecture", fontsize=14, fontweight="bold", y=0.98)
    
    # Footer note
    ax.text(0.5, 0.2, "* Designed by Kiran, IIT Madras. Implements unified Reject Inference, Algorithmic Fairness, and XAI.",
            ha="center", va="center", fontsize=10, style="italic", color="#555555")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Pipeline architecture diagram saved to: {save_path}")


def generate_local_explanations() -> None:
    """Train a pipeline on German credit dataset and generate SHAP and DiCE explanations."""
    print("Loading German Credit dataset...")
    df = load_dataset("german", data_path="data/prepared")
    
    # Train test split
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["DEFAULT", "Gender"])
    y = df["DEFAULT"].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Fit base LightGBM pipeline
    print("Fitting baseline LightGBM pipeline...")
    clf = LGBMClassifier(n_estimators=50, max_depth=5, random_state=42, verbose=-1)
    pipeline = create_pipeline(X_train, y_train, clf, normalize=True, do_EBE=True, onehot=True)
    pipeline.fit(X_train, y_train)
    
    # Find a sample predicted as default (e.g. y_prob > 0.6)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    default_indices = np.where((y_prob > 0.6) & (y_test == 1))[0]
    
    if len(default_indices) == 0:
        default_indices = np.where(y_prob > 0.5)[0]
        
    sample_idx = default_indices[0]
    sample_x = X_test.iloc[[sample_idx]]
    
    # --- 1. SHAP Explanation ---
    print("Generating SHAP explanation...")
    try:
        explainer = ShapPipelineExplainer(pipeline, X_train.iloc[:50], method_explain="prob")
        explainer.plot_explanation(sample_x)
        shap_path = "assets/german_shap_explanation.png"
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"SHAP explanation waterfall plot saved to: {shap_path}")
    except Exception as e:
        print(f"Failed to generate SHAP explanation: {e}")
        
    # --- 2. DiCE Counterfactuals ---
    print("Generating DiCE counterfactuals...")
    try:
        # Choose continuous mutable features
        mutable_features = ["Duration", "CreditAmount", "Age"]
        # Filter mutable features that exist in preprocessed space
        # Preprocessed columns:
        X_pre = pipeline[:2].transform(X_train)
        valid_mutables = [c for c in X_pre.columns if any(m in c for m in mutable_features)]
        
        dice_explainer = Dice(
            X_train.iloc[:50],
            y_train.iloc[:50],
            pipeline,
            n_cfs=3,
            mutable_features=valid_mutables,
            sparsity_weight=0.5
        )
        
        cfs = dice_explainer.fit(sample_x)
        cf_df = display_cfs(sample_x, cfs, pipeline, show_change=True)
        
        cf_path = "assets/german_counterfactuals.md"
        with open(cf_path, "w") as f:
            f.write("# Counterfactual Explanations (Actionable Recourse)\n\n")
            f.write("The table below shows how the rejected applicant can modify their application ")
            f.write("attributes (e.g., lower credit amount or loan duration) to change the model's outcome ")
            f.write("from **Default (Rejected)** to **Approved**.\n\n")
            f.write(cf_df.to_markdown())
            f.write("\n")
            
        print(f"DiCE counterfactuals saved to: {cf_path}")
        print("\nCounterfactual Recourse Table:")
        print(cf_df.to_string())
    except Exception as e:
        print(f"Failed to generate DiCE counterfactuals: {e}")


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    draw_pipeline_architecture("assets/pipeline_architecture.png")
    generate_local_explanations()
