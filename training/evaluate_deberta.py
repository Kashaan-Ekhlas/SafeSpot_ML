import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    multilabel_confusion_matrix,
    classification_report
)

NUM_LABELS   = 5
BATCH_SIZE   = 32
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_NAMES = ["S1_harassment", "S2_hate", "S3_violence", "S4_sexual", "S8_safe"]

print("device:", DEVICE)

# Use test split — val was seen during training, test is truly held out
dataset = load_dataset("json", data_files={"test": "../processed/test.jsonl"})

tokenizer = AutoTokenizer.from_pretrained("../models/deberta_lora/full_model")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset = dataset["test"]

model = AutoModelForSequenceClassification.from_pretrained("../models/deberta_lora/full_model").to(DEVICE)
model.eval()
print("model loaded\n")

all_preds  = []
all_labels = []

with torch.no_grad():
    for i in range(0, len(test_dataset), BATCH_SIZE):
        batch         = test_dataset[i:i+BATCH_SIZE]
        input_ids     = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels        = batch["labels"].cpu().numpy()

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = torch.sigmoid(logits)
        preds  = (probs > 0.5).int().cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels)

all_preds  = np.vstack(all_preds)
all_labels = np.vstack(all_labels)

print("metrics")
print("accuracy:        ", accuracy_score(all_labels, all_preds))
print("micro_f1:        ", f1_score(all_labels, all_preds, average="micro", zero_division=0))
print("macro_f1:        ", f1_score(all_labels, all_preds, average="macro", zero_division=0))
print("precision_macro: ", precision_score(all_labels, all_preds, average="macro", zero_division=0))
print("recall_macro:    ", recall_score(all_labels, all_preds, average="macro", zero_division=0))

print("\nconfusion matrices")
for i, cm in enumerate(multilabel_confusion_matrix(all_labels, all_preds)):
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  {LABEL_NAMES[i]}")
    print(f"  tn: {tn}  fp: {fp}")
    print(f"  fn: {fn}  tp: {tp}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"  precision: {precision:.3f}  recall: {recall:.3f}")

print("\nclassification report")
print(classification_report(all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0))