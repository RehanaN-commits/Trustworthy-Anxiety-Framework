"""
============================================================
Trustworthy Mental Distress Classification Framework

File: train.py

Purpose:
    Fine-tune transformer models for mental distress
    classification.

Supported Models
----------------
- distilbert-base-uncased
- roberta-base
- microsoft/deberta-v3-base
============================================================
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import evaluate
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

RESULT_DIR = PROJECT_ROOT / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_MODELS = [

    "distilbert-base-uncased",

    "roberta-base",

    "microsoft/deberta-v3-base",

]

MAX_LENGTH = 256

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

EPOCHS = 3

# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--model",

        required=True,

        choices=SUPPORTED_MODELS,

    )

    return parser.parse_args()

# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    train_df = pd.read_csv(
        DATA_DIR / "train.csv"
    )

    validation_df = pd.read_csv(
        DATA_DIR / "validation.csv"
    )

    with open(
        DATA_DIR / "label_mapping.json",
        "r",
        encoding="utf-8",
    ) as file:

        label_mapping = json.load(file)

    num_labels = len(label_mapping)

    print("\nDataset Loaded")

    print(f"Train Samples      : {len(train_df)}")

    print(f"Validation Samples : {len(validation_df)}")

    print(f"Classes            : {num_labels}")

    return (
        train_df,
        validation_df,
        num_labels,
    )

# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer(model_name):

    print(f"\nLoading Tokenizer : {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    return tokenizer
# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples, tokenizer):

    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


# ============================================================
# PREPARE DATASETS
# ============================================================

def prepare_datasets(
    train_df,
    validation_df,
    tokenizer,
):

    print("\nPreparing Hugging Face datasets...")
    train_df = train_df.rename(columns={"label_id": "labels"})
    validation_df = validation_df.rename(columns={"label_id": "labels"})

    train_dataset = Dataset.from_pandas(
        train_df
    )

    validation_dataset = Dataset.from_pandas(
        validation_df
    )

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    validation_dataset = validation_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )

    train_dataset.set_format(
        type="torch",
       columns=[
    "input_ids",
    "attention_mask",
    "labels",
]
    )

    validation_dataset.set_format(
        type="torch",
        columns=[
    "input_ids",
    "attention_mask",
    "labels",
]
    )

    print(train_dataset.column_names)
    print(validation_dataset.column_names)

    return (
        train_dataset,
        validation_dataset,
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_name,
    num_labels,
):

    print(f"\nLoading Model : {model_name}")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    return model


# ============================================================
# METRICS
# ============================================================

accuracy_metric = evaluate.load("accuracy")

f1_metric = evaluate.load("f1")

precision_metric = evaluate.load("precision")

recall_metric = evaluate.load("recall")


def compute_metrics(eval_prediction):

    logits, labels = eval_prediction.predictions, eval_prediction.label_ids

    predictions = np.argmax(
        logits,
        axis=1,
    )

    accuracy = accuracy_metric.compute(
        predictions=predictions,
        references=labels,
    )["accuracy"]

    precision = precision_metric.compute(
        predictions=predictions,
        references=labels,
        average="weighted",
    )["precision"]

    recall = recall_metric.compute(
        predictions=predictions,
        references=labels,
        average="weighted",
    )["recall"]

    f1 = f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="weighted",
    )["f1"]

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

    }
# ============================================================
# TRAINING ARGUMENTS
# ============================================================

from transformers import EarlyStoppingCallback


def create_training_arguments(model_name):

    output_dir = MODEL_DIR / model_name.replace("/", "_")

    training_args = TrainingArguments(

        output_dir=str(output_dir),

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,

        per_device_eval_batch_size=BATCH_SIZE,

        learning_rate=LEARNING_RATE,

        weight_decay=0.01,

        eval_strategy="epoch",

        save_strategy="epoch",

        logging_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="f1",

        greater_is_better=True,

        save_total_limit=2,

        report_to="none",

        seed=42,

    )

    return training_args


# ============================================================
# TRAIN MODEL
# ============================================================


def train_model(

    model,

    tokenizer,

    train_dataset,

    validation_dataset,

    training_args,

):

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        compute_metrics=compute_metrics,

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],

    )

    trainer.train()

    return trainer


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(

    trainer,

    model_name,

):

    metrics = trainer.evaluate()

    metric_path = (
        RESULT_DIR
        / f"{model_name.replace('/', '_')}_metrics.json"
    )

    with open(
        metric_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\nValidation Metrics")

    for key, value in metrics.items():

        print(f"{key} : {value}")

    trainer.save_model()

    print("\nBest model saved successfully.")

    print(f"\nMetrics saved to:\n{metric_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    train_df, validation_df, num_labels = load_data()

    tokenizer = load_tokenizer(args.model)

    train_dataset, validation_dataset = prepare_datasets(

        train_df,

        validation_df,

        tokenizer,

    )

    model = load_model(

        args.model,

        num_labels,

    )

    training_args = create_training_arguments(

        args.model,

    )

    trainer = train_model(

        model,

        tokenizer,

        train_dataset,

        validation_dataset,

        training_args,

    )

    save_results(

        trainer,

        args.model,

    )

    print("\nTraining Completed Successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
