import torch
import numpy as np
import torch.nn as nn
from datasets import load_dataset, Features, Sequence, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, precision_score, recall_score

MODEL_NAME  = "microsoft/deberta-v3-base"
NUM_LABELS  = 5
LABEL_NAMES = ["S1_harassment", "S2_hate", "S3_violence", "S4_sexual", "S8_safe"]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

dataset = load_dataset(
    "json",
    data_files={
        "train":      "../processed/train.jsonl",
        "validation": "../processed/val.jsonl"
    }
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)

features = dataset["train"].features.copy()
features["labels"] = Sequence(Value("float32"))
dataset = dataset.cast(features)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

labels_array  = np.array(dataset["train"]["labels"])
label_counts  = labels_array.sum(axis=0)
total_samples = len(labels_array)

pos_weights = np.clip(
    (total_samples - label_counts) / (label_counts + 1e-6),
    a_min=1.0,
    a_max=10.0
)

print("Class counts:", label_counts)
print("Class weights:", pos_weights)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_proj", "value_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.to(DEVICE)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs > 0.5).int().numpy()

    macro_f1  = f1_score(labels, preds, average="macro", zero_division=0)
    micro_f1  = f1_score(labels, preds, average="micro", zero_division=0)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall    = recall_score(labels, preds, average="macro", zero_division=0)

    # Per-label F1 so you can see which labels are struggling each epoch
    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
    per_label_metrics = {
        f"f1_{name}": float(per_label_f1[i])
        for i, name in enumerate(LABEL_NAMES)
    }

    return {
        "macro_f1":       macro_f1,
        "micro_f1":       micro_f1,
        "precision_macro": precision,
        "recall_macro":   recall,
        **per_label_metrics
    }


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        loss_fct = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weights, dtype=torch.float32).to(model.device)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir="../models/deberta_lora",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,          # gradient clipping — prevents loss explosion
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=200,
    bf16=True,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    report_to="none"
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train(resume_from_checkpoint=True)
print("Saving merged full model...")
merged = trainer.model.merge_and_unload()
merged.save_pretrained("../models/deberta_lora/full_model")
tokenizer.save_pretrained("../models/deberta_lora/full_model")
print("Training complete.")