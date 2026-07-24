"""
============================================================
Compare All Models
Trustworthy Mental Distress Classification Framework
============================================================

This script compares the performance of all trained models.

Models:
1. Logistic Regression
2. Linear SVM
3. Random Forest
4. DistilBERT
5. RoBERTa

Outputs:
- Final comparison table
- CSV file
- Performance visualizations

Author: Rehana N
============================================================
"""

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Metric Files
# ============================================================

METRIC_FILES = {
    "Logistic Regression": "logistic_regression_metrics.json",
    "Linear SVM": "linear_svm_metrics.json",
    "Random Forest": "random_forest_metrics.json",
    "DistilBERT": "distilbert-base-uncased_metrics.json",
    "RoBERTa": "roberta-base_metrics.json",
}


# ============================================================
# Load Metrics
# ============================================================

def load_metrics():

    comparison_data = []

    print("=" * 60)
    print("Loading Model Metrics")
    print("=" * 60)

    for model_name, filename in METRIC_FILES.items():

        file_path = METRICS_DIR / filename

        print(file_path, file_path.exists())

        if not file_path.exists():
            print(f"Missing metrics file: {filename}")
            continue

        with open(file_path, "r") as f:
            metrics = json.load(f)

        # Support both classical ML and transformer metric formats
        accuracy = metrics.get("accuracy", metrics.get("eval_accuracy"))
        precision = metrics.get("precision", metrics.get("eval_precision"))
        recall = metrics.get("recall", metrics.get("eval_recall"))
        f1 = metrics.get("f1_score", metrics.get("eval_f1"))

        comparison_data.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        })

        print(f"Loaded: {model_name}")

    print()

    return comparison_data
# ============================================================
# Create Comparison DataFrame
# ============================================================

def create_comparison_dataframe(comparison_data):

    df = pd.DataFrame(comparison_data)

    df = df.sort_values(
        by="Accuracy",
        ascending=False
    ).reset_index(drop=True)

    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)

    print(df)

    print()

    return df
# ============================================================
# Save Comparison Table
# ============================================================

def save_comparison_table(df):

    output_file = TABLES_DIR / "all_models_comparison.csv"

    df.to_csv(output_file, index=False)

    print(f"Comparison table saved to:\n{output_file}\n")


# ============================================================
# Plot Metric
# ============================================================

def plot_metric(df, metric, filename):

    plt.figure(figsize=(8, 5))

    bars = plt.bar(df["Model"], df[metric])

    plt.title(f"{metric} Comparison", fontsize=14)
    plt.xlabel("Models")
    plt.ylabel(metric)

    plt.ylim(0, 1)

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()

    output_path = FIGURES_DIR / filename

    plt.savefig(output_path, dpi=300)

    plt.close()

    print(f"{metric} figure saved: {output_path}")


# ============================================================
# Generate All Figures
# ============================================================

def generate_figures(df):

    print("=" * 60)
    print("Generating Comparison Figures")
    print("=" * 60)

    plot_metric(df, "Accuracy", "accuracy_comparison.png")
    plot_metric(df, "Precision", "precision_comparison.png")
    plot_metric(df, "Recall", "recall_comparison.png")
    plot_metric(df, "F1 Score", "f1_score_comparison.png")

    print()


# ============================================================
# Print Best Model
# ============================================================

def print_best_model(df):

    best_model = df.iloc[0]

    print("=" * 60)
    print("Best Performing Model")
    print("=" * 60)

    print(f"Model      : {best_model['Model']}")
    print(f"Accuracy   : {best_model['Accuracy']:.4f}")
    print(f"Precision  : {best_model['Precision']:.4f}")
    print(f"Recall     : {best_model['Recall']:.4f}")
    print(f"F1 Score   : {best_model['F1 Score']:.4f}")

    print()


# ============================================================
# Main
# ============================================================

def main():

    comparison_data = load_metrics()

    if len(comparison_data) == 0:
        print("No metrics files found.")
        return

    comparison_df = create_comparison_dataframe(
        comparison_data
    )

    save_comparison_table(comparison_df)

    generate_figures(comparison_df)

    print_best_model(comparison_df)

    print("=" * 60)
    print("Comparison Completed Successfully.")
    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()