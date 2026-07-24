
"""
train_severity.py
Train a RoBERTa model for severity classification.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "severity_roberta"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train_severity.csv")
val_df = pd.read_csv(DATA_DIR / "validation_severity.csv")
test_df = pd.read_csv(DATA_DIR / "test_severity.csv")

label_col = "severity_id"

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=False, max_length=256)

train_ds = Dataset.from_pandas(train_df[["text",label_col]].rename(columns={label_col:"labels"})).map(tokenize, batched=True)
val_ds = Dataset.from_pandas(val_df[["text",label_col]].rename(columns={label_col:"labels"})).map(tokenize, batched=True)
test_ds = Dataset.from_pandas(test_df[["text",label_col]].rename(columns={label_col:"labels"})).map(tokenize, batched=True)

cols=["input_ids","attention_mask","labels"]
train_ds.set_format(type="torch", columns=cols)
val_ds.set_format(type="torch", columns=cols)
test_ds.set_format(type="torch", columns=cols)

num_labels = train_df[label_col].nunique()

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    p,r,f,_ = precision_recall_fscore_support(labels,preds,average="weighted",zero_division=0)
    acc = accuracy_score(labels,preds)
    return {
        "accuracy":acc,
        "precision":p,
        "recall":r,
        "f1":f
    }

args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

pred = trainer.predict(test_ds)
preds = np.argmax(pred.predictions, axis=1)
labels = pred.label_ids

metrics = compute_metrics((pred.predictions, labels))

with open(RESULTS_DIR/"severity_metrics.json","w") as f:
    json.dump(metrics,f,indent=4)

with open(RESULTS_DIR/"severity_classification_report.txt","w") as f:
    f.write(classification_report(labels,preds))

np.savetxt(
    RESULTS_DIR/"severity_confusion_matrix.csv",
    confusion_matrix(labels,preds),
    delimiter=",",
    fmt="%d"
)

print("Training complete.")
print(metrics)
