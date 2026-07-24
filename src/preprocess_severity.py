"""
============================================================
Trustworthy Mental Distress Classification Framework

File: preprocess_severity.py

Purpose:
    Prepare the MentalDistress Severity dataset
    for transformer training.

Outputs:
    data/processed/
        ├── train_severity.csv
        ├── validation_severity.csv
        ├── test_severity.csv
        ├── severity_mapping.json
        └── severity_preprocessing_report.json
============================================================
"""

from pathlib import Path
import json
import random
import re
import unicodedata

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TRAIN_SIZE = 0.80
VALID_SIZE = 0.10
TEST_SIZE = 0.10

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    dataset_path = (
        RAW_DIR /
        "MentalDistress" /
        "Mental_Distress_Dataset-labeled.csv"
    )

    dataframe = pd.read_csv(dataset_path)

    print(f"\nLoaded dataset: {dataset_path.name}")
    print(f"Samples : {len(dataframe)}")

    return dataframe

# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(dataframe):

    required_columns = [
        "text",
        "severity"
    ]

    for column in required_columns:

        if column not in dataframe.columns:

            raise ValueError(f"Missing column: {column}")

    dataframe = dataframe.dropna(
        subset=["text", "severity"]
    )

    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["severity"] = dataframe["severity"].astype(str)

    dataframe = dataframe[
        dataframe["text"].str.strip() != ""
    ]

    print(f"Valid Samples : {len(dataframe)}")

    return dataframe

# ============================================================
# CLEAN TEXT
# ============================================================

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")

def clean_text(text):

    text = unicodedata.normalize("NFKC", text)

    text = URL_PATTERN.sub("", text)
    text = EMAIL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()

def preprocess_text(dataframe):

    dataframe["text"] = dataframe["text"].apply(clean_text)

    dataframe = dataframe[
        dataframe["text"].str.strip() != ""
    ]

    dataframe = dataframe.reset_index(drop=True)

    return dataframe

# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(dataframe):

    dataframe = dataframe.drop_duplicates(
        subset=["text"]
    )

    dataframe = dataframe.reset_index(drop=True)

    return dataframe

# ============================================================
# ENCODE LABELS
# ============================================================

def encode_labels(dataframe):

    encoder = LabelEncoder()

    dataframe["severity_id"] = encoder.fit_transform(
        dataframe["severity"]
    )

    mapping = {
        label: int(index)
        for index, label in enumerate(
            encoder.classes_
        )
    }

    with open(
        PROCESSED_DIR / "severity_mapping.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            mapping,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nSeverity Mapping")

    for label, idx in mapping.items():

        print(idx, ":", label)

    return dataframe

# ============================================================
# REPORT
# ============================================================

def generate_report(dataframe):

    report = {

        "samples": int(len(dataframe)),

        "classes": int(
            dataframe["severity"].nunique()
        )

    }

    with open(
        PROCESSED_DIR /
        "severity_preprocessing_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(report, file, indent=4)

# ============================================================
# SPLIT
# ============================================================

def split_dataset(dataframe):

    train_df, temp_df = train_test_split(

        dataframe,

        train_size=TRAIN_SIZE,

        stratify=dataframe["severity_id"],

        random_state=RANDOM_SEED,

        shuffle=True,

    )

    valid_ratio = VALID_SIZE / (VALID_SIZE + TEST_SIZE)

    validation_df, test_df = train_test_split(

        temp_df,

        train_size=valid_ratio,

        stratify=temp_df["severity_id"],

        random_state=RANDOM_SEED,

        shuffle=True,

    )

    return (

        train_df.reset_index(drop=True),

        validation_df.reset_index(drop=True),

        test_df.reset_index(drop=True),

    )

# ============================================================
# SAVE
# ============================================================

def save_datasets(

    train_df,

    validation_df,

    test_df,

):

    train_df.to_csv(
        PROCESSED_DIR /
        "train_severity.csv",
        index=False
    )

    validation_df.to_csv(
        PROCESSED_DIR /
        "validation_severity.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_DIR /
        "test_severity.csv",
        index=False
    )

    print("\nSeverity datasets saved successfully.")

# ============================================================
# MAIN
# ============================================================

def main():

    dataframe = load_dataset()

    dataframe = validate_dataset(dataframe)

    dataframe = preprocess_text(dataframe)

    dataframe = remove_duplicates(dataframe)

    dataframe = encode_labels(dataframe)

    generate_report(dataframe)

    train_df, validation_df, test_df = split_dataset(
        dataframe
    )

    save_datasets(
        train_df,
        validation_df,
        test_df,
    )

    print("\nSeverity preprocessing completed.")

if __name__ == "__main__":
    main()