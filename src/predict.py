"""
predict.py
-----------
RoBERTa inference module for
Trustworthy Mental Distress Classification Framework.

Author: Rehana
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "roberta-base"

# ==========================================================
# Load Model
# ==========================================================

print("[INFO] Loading RoBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("[INFO] RoBERTa loaded successfully.")

# ==========================================================
# Label Mapping
# ==========================================================

LABELS = {
    0: "Anxious",
    1: "Depressed",
    2: "Suicidal",
    3: "Frustrated",
    4: "Others",
}

# ==========================================================
# Prediction
# ==========================================================


def predict_distress(text: str):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(probabilities, dim=1).item()

    confidence = probabilities[0][prediction].item()

    probability_dict = {}

    for i, label in LABELS.items():

        probability_dict[label] = round(
            probabilities[0][i].item() * 100,
            2,
        )

    result = {

        "prediction": prediction,

        "label": LABELS[prediction],

        "confidence": round(confidence, 4),

        "probabilities": probability_dict,

    }

    return result


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    sample = (
        "I feel hopeless and I don't know "
        "what to do anymore."
    )

    output = predict_distress(sample)

    print("\nPrediction Result\n")

    print(output)