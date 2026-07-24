from pathlib import Path
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "models" / "roberta-base"

print("Loading tokenizer from Hugging Face...")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

print("Saving tokenizer to:")
print(MODEL_DIR)

tokenizer.save_pretrained(MODEL_DIR)

print("\nDone!")