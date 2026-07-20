"""
============================================================
Trustworthy Mental Distress Classification Framework

File: preprocess.py

Purpose:
    Prepare the MentalDistress dataset for transformer training.

Outputs:
    data/processed/
        ├── train.csv
        ├── validation.csv
        ├── test.csv
        ├── label_mapping.json
        └── preprocessing_report.json
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

# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(dataset_name: str):

    dataset_folder = RAW_DIR / dataset_name

    csv_files = list(dataset_folder.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError("Dataset CSV not found.")

    dataframe = pd.read_csv(csv_files[0])

    print(f"\nLoaded dataset: {csv_files[0].name}")
    print(f"Samples : {len(dataframe)}")

    return dataframe

# ============================================================
# VALIDATE DATASET
# ============================================================

TEXT_COLUMNS = [
    "text",
    "sentence",
    "content",
    "post",
    "tweet",
]

LABEL_COLUMNS = [
    "label",
    "class",
    "category",
]

def validate_dataset(dataframe):

    lower_columns = {
        c.lower().strip(): c
        for c in dataframe.columns
    }

    text_column = None
    label_column = None

    for c in TEXT_COLUMNS:

        if c in lower_columns:
            text_column = lower_columns[c]
            break

    for c in LABEL_COLUMNS:

        if c in lower_columns:
            label_column = lower_columns[c]
            break

    if text_column is None:
        raise ValueError("Text column not found.")

    if label_column is None:
        raise ValueError("Label column not found.")

    dataframe = dataframe.rename(
        columns={
            text_column: "text",
            label_column: "label",
        }
    )

    dataframe = dataframe.dropna(
        subset=["text", "label"]
    )

    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["label"] = dataframe["label"].astype(str)

    dataframe = dataframe[
        dataframe["text"].str.strip() != ""
    ]

    print(f"Valid Samples : {len(dataframe)}")
    print(f"Classes       : {dataframe['label'].nunique()}")

    return dataframe
# ============================================================
# TEXT CLEANING
# ============================================================

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Basic transformer-friendly text cleaning.
    """

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs
    text = URL_PATTERN.sub("", text)

    # Remove Emails
    text = EMAIL_PATTERN.sub("", text)

    # Remove @mentions
    text = MENTION_PATTERN.sub("", text)

    # Replace new lines and tabs
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    # Normalize whitespace
    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


# ============================================================
# CLEAN DATASET
# ============================================================

def preprocess_text(dataframe):

    print("\nCleaning text...")

    dataframe["text"] = dataframe["text"].apply(clean_text)

    # Remove empty samples after cleaning
    dataframe = dataframe[
        dataframe["text"].str.strip() != ""
    ]

    dataframe = dataframe.reset_index(drop=True)

    print(f"Remaining Samples : {len(dataframe)}")

    return dataframe


# ============================================================
# REMOVE DUPLICATE TEXTS
# ============================================================

def remove_duplicates(dataframe):

    print("\nRemoving duplicate texts...")

    before = len(dataframe)

    dataframe = dataframe.drop_duplicates(
        subset=["text"]
    )

    dataframe = dataframe.reset_index(drop=True)

    removed = before - len(dataframe)

    print(f"Removed Duplicates : {removed}")
    print(f"Final Samples      : {len(dataframe)}")

    return dataframe


# ============================================================
# LABEL ENCODING
# ============================================================

def encode_labels(dataframe):

    print("\nEncoding labels...")

    encoder = LabelEncoder()

    dataframe["label_id"] = encoder.fit_transform(
        dataframe["label"]
    )

    label_mapping = {
        label: int(index)
        for index, label in enumerate(
            encoder.classes_
        )
    }

    with open(
        PROCESSED_DIR / "label_mapping.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            label_mapping,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nLabel Mapping")

    for label, index in label_mapping.items():

        print(f"{index} : {label}")

    return dataframe


# ============================================================
# PREPROCESSING REPORT
# ============================================================

def generate_report(dataframe):

    report = {

        "samples":
            int(len(dataframe)),

        "classes":
            int(dataframe["label"].nunique()),

        "average_words":
            round(
                dataframe["text"]
                .str.split()
                .apply(len)
                .mean(),
                2,
            ),

        "minimum_words":
            int(
                dataframe["text"]
                .str.split()
                .apply(len)
                .min()
            ),

        "maximum_words":
            int(
                dataframe["text"]
                .str.split()
                .apply(len)
                .max()
            ),

    }

    with open(
        PROCESSED_DIR / "preprocessing_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print("\nPreprocessing Summary")

    for key, value in report.items():

        print(f"{key} : {value}")

    return report
# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_dataset(dataframe):

    print("\nCreating train/validation/test split...")

    train_df, temp_df = train_test_split(
        dataframe,
        train_size=TRAIN_SIZE,
        stratify=dataframe["label_id"],
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    valid_ratio = VALID_SIZE / (VALID_SIZE + TEST_SIZE)

    validation_df, test_df = train_test_split(
        temp_df,
        train_size=valid_ratio,
        stratify=temp_df["label_id"],
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    print(f"Train Samples      : {len(train_df)}")
    print(f"Validation Samples : {len(validation_df)}")
    print(f"Test Samples       : {len(test_df)}")

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ============================================================
# SAVE DATASETS
# ============================================================

def save_datasets(
    train_df,
    validation_df,
    test_df,
):

    train_path = PROCESSED_DIR / "train.csv"
    validation_path = PROCESSED_DIR / "validation.csv"
    test_path = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nSaved Processed Files")

    print(train_path)
    print(validation_path)
    print(test_path)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    dataset_name = "MentalDistress"

    print("=" * 60)
    print("Mental Distress Dataset Preprocessing")
    print("=" * 60)

    dataframe = load_dataset(dataset_name)

    dataframe = validate_dataset(dataframe)

    dataframe = preprocess_text(dataframe)

    dataframe = remove_duplicates(dataframe)

    dataframe = encode_labels(dataframe)

    generate_report(dataframe)

    (
        train_df,
        validation_df,
        test_df,
    ) = split_dataset(dataframe)

    save_datasets(
        train_df,
        validation_df,
        test_df,
    )

    print("\nPreprocessing Completed Successfully.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()