"""
============================================================
Train Classical Machine Learning Models
============================================================

This script trains and evaluates classical machine learning
models for multiclass mental distress classification.

Supported Models
----------------
1. Logistic Regression
2. Linear SVM
3. Random Forest

Pipeline
--------
Load Dataset
    ↓
TF-IDF Feature Extraction
    ↓
Train Model
    ↓
Evaluate Model
    ↓
Save Model
    ↓
Save Metrics
    ↓
Generate Comparison Table

Author : Rehana N
============================================================
"""

# ============================================================
# Imports
# ============================================================

from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"

METRICS_DIR = RESULTS_DIR / "metrics"

TABLES_DIR = RESULTS_DIR / "tables"

REPORTS_DIR = RESULTS_DIR / "reports"

PREDICTIONS_DIR = RESULTS_DIR / "predictions"

CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"

FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load Datasets
# ============================================================

def load_datasets():
    """
    Load the train, validation and test datasets.

    Returns
    -------
    tuple
        train_df, validation_df, test_df
    """

    train_df = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    validation_df = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    test_df = pd.read_csv(
        DATA_DIR / "test.csv"
    )

    return (
        train_df,
        validation_df,
        test_df,
    )

# ============================================================
# Dataset Summary
# ============================================================

def dataset_summary(
    train_df,
    validation_df,
    test_df,
):
    """
    Display information about the datasets.
    """

    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    print(f"Training Samples   : {len(train_df)}")
    print(f"Validation Samples : {len(validation_df)}")
    print(f"Testing Samples    : {len(test_df)}")

    print("\nColumns:")

    print(train_df.columns.tolist())

    print("=" * 60)

    print("\nLabel Distribution (Training Set)\n")

    print(
        train_df["label"].value_counts()
    )

    print("=" * 60)
    # ============================================================
# TF-IDF Feature Extraction
# ============================================================

def create_tfidf_features(
    train_df,
    validation_df,
    test_df,
):
    """
    Convert text into TF-IDF feature vectors.

    Returns
    -------
    tuple
        X_train
        X_validation
        X_test
        y_train
        y_validation
        y_test
        vectorizer
    """

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
    )

    X_train = vectorizer.fit_transform(
        train_df["text"]
    )

    X_validation = vectorizer.transform(
        validation_df["text"]
    )

    X_test = vectorizer.transform(
        test_df["text"]
    )

    y_train = train_df["label_id"]

    y_validation = validation_df["label_id"]

    y_test = test_df["label_id"]

    print("TF-IDF feature extraction completed.\n")

    print(f"Training Features   : {X_train.shape}")
    print(f"Validation Features : {X_validation.shape}")
    print(f"Testing Features    : {X_test.shape}")

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        vectorizer,
    )


# ============================================================
# Train Model
# ============================================================

def train_model(
    model,
    model_name,
    X_train,
    y_train,
):
    """
    Train a machine learning model.

    Parameters
    ----------
    model
        Scikit-learn model

    model_name
        Name of the model

    X_train
        Training features

    y_train
        Training labels
    """

    print("\n" + "=" * 60)
    print(f"Training {model_name}")
    print("=" * 60)

    model.fit(
        X_train,
        y_train,
    )

    print(f"{model_name} training completed.")

    return model
# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name,
):
    """
    Evaluate a trained machine learning model.

    Parameters
    ----------
    model
        Trained scikit-learn model.

    X_test
        Testing features.

    y_test
        Testing labels.

    model_name
        Name of the model.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    print("\n" + "=" * 60)
    print(f"Evaluating {model_name}")
    print("=" * 60)

    predictions = model.predict(
        X_test,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    print("\nTest Results")
    print("-" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print("-" * 60)

    report = classification_report(
    y_test,
    predictions,
    target_names=[
        "Anxious",
        "Depressed",
        "Frustrated",
        "Others",
        "Suicidal",
    ],
    zero_division=0,
)

    print(report)

    metrics = {

        "model": model_name,

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

    }

    return metrics, report, predictions
# ============================================================
# Save Model
# ============================================================

def save_model(
    model,
    vectorizer,
    model_name,
):
    """
    Save the trained model and TF-IDF vectorizer.
    """

    filename = (
        model_name.lower()
        .replace(" ", "_")
    )

    model_path = MODEL_DIR / f"{filename}.pkl"

    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        vectorizer,
        vectorizer_path,
    )

    print("\nModel saved successfully.")
    print(f"Model      : {model_path}")
    print(f"Vectorizer : {vectorizer_path}")


# ============================================================
# Save Metrics
# ============================================================

def save_metrics(
    metrics,
    model_name,
):
    """
    Save evaluation metrics as a JSON file.
    """

    filename = (
        model_name.lower()
        .replace(" ", "_")
    )

    metrics_path = METRICS_DIR / f"{filename}_metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print(f"\nMetrics saved : {metrics_path}")

# ============================================================
# Save Classification Report
# ============================================================

def save_classification_report(
    report,
    model_name,
):
    """
    Save the classification report as a text file.
    """

    filename = (
        model_name.lower()
        .replace(" ", "_")
    )

    report_path = REPORTS_DIR / f"{filename}_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(report)

    print(f"\nClassification report saved: {report_path}")
    # ============================================================
# Save Predictions
# ============================================================

def save_predictions(
    test_df,
    predictions,
    model_name,
):
    """
    Save model predictions to a CSV file.
    """

    filename = (
        model_name.lower()
        .replace(" ", "_")
    )

    prediction_path = (
        PREDICTIONS_DIR /
        f"{filename}_predictions.csv"
    )

    prediction_df = test_df.copy()

    prediction_df["predicted_label_id"] = predictions

    label_mapping = {
        0: "Anxious",
        1: "Depressed",
        2: "Frustrated",
        3: "Others",
        4: "Suicidal",
    }

    prediction_df["predicted_label"] = (
        prediction_df["predicted_label_id"]
        .map(label_mapping)
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    print(
        f"\nPredictions saved: {prediction_path}"
    )
    # ============================================================
# Save Confusion Matrix
# ============================================================

def save_confusion_matrix(
    y_true,
    predictions,
    model_name,
):
    """
    Save the confusion matrix as a PNG image.
    """

    labels = [
        "Anxious",
        "Depressed",
        "Frustrated",
        "Others",
        "Suicidal",
    ]

    cm = confusion_matrix(
        y_true,
        predictions,
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation="nearest", cmap="Blues")

    plt.title(f"{model_name} Confusion Matrix")

    plt.colorbar()

    tick_marks = range(len(labels))

    plt.xticks(
        tick_marks,
        labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        tick_marks,
        labels,
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
    )

    save_path = (
        CONFUSION_DIR /
        f"{filename}_cm.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"\nConfusion matrix saved: {save_path}")
# ============================================================
# Generate Comparison Table
# ============================================================

def save_comparison_table(
    comparison_results,
):
    """
    Save comparison table for all classical models.
    """

    comparison_df = pd.DataFrame(
        comparison_results
    )

    comparison_path = (
        TABLES_DIR /
        "classical_models_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("Classical Model Comparison")
    print("=" * 60)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print(f"\nComparison table saved to:")
    print(comparison_path)

    return comparison_df

# ============================================================
# Save Comparison Figures
# ============================================================

def save_comparison_figures(comparison_df):
    """
    Generate publication-quality comparison figures.
    """

    # ---------------- Accuracy ----------------

    plt.figure(figsize=(8, 5))

    plt.bar(
        comparison_df["Model"],
        comparison_df["Accuracy"],
    )

    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison of Classical Models")

    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------- F1 ----------------

    plt.figure(figsize=(8, 5))

    plt.bar(
        comparison_df["Model"],
        comparison_df["F1 Score"],
    )

    plt.ylabel("Weighted F1 Score")
    plt.title("F1 Score Comparison of Classical Models")

    plt.ylim(0, 1)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "f1_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nComparison figures saved.")
# ============================================================
# Run Complete Model Pipeline
# ============================================================

def run_model_pipeline(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    test_df,
    vectorizer,
):
    """
    Train, evaluate and save a machine learning model.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    model = train_model(
        model,
        model_name,
        X_train,
        y_train,
    )

    metrics, report, predictions = evaluate_model(
        model,
        X_test,
        y_test,
        model_name,
    )

    save_model(
        model,
        vectorizer,
        model_name,
    )

    save_metrics(
        metrics,
        model_name,
    )

    save_classification_report(
        report,
        model_name,
    )
    save_predictions(
    test_df,
    predictions,
    model_name,
)
    save_confusion_matrix(
    y_test,
    predictions,
    model_name,
)

    return metrics
# ============================================================
# Main Function
# ============================================================

def main():

    # ========================================================
    # Load Dataset
    # ========================================================

    train_df, validation_df, test_df = load_datasets()

    dataset_summary(
        train_df,
        validation_df,
        test_df,
    )

    # ========================================================
    # TF-IDF Feature Extraction
    # ========================================================

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        vectorizer,
    ) = create_tfidf_features(
        train_df,
        validation_df,
        test_df,
    )

    # ========================================================
    # Classical Machine Learning Models
    # ========================================================

    models = [

        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),

        (
            "Linear SVM",
            LinearSVC(
                random_state=42,
            ),
        ),

        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),

    ]

    comparison_results = []

    # ========================================================
    # Train, Evaluate and Save
    # ========================================================

    for model_name, model in models:

        metrics = run_model_pipeline(
    model,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    test_df,
    vectorizer,
)

        comparison_results.append(
            {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
            }
        )

    # ========================================================
    # Comparison Table
    # ========================================================

    comparison_df = save_comparison_table(
    comparison_results,
)

    save_comparison_figures(
    comparison_df,
)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()