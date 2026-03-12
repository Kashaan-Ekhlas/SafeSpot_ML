import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_NAMES = ["S1_harassment", "S2_hate", "S3_violence", "S4_sexual", "S8_safe"]

print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained("../models/deberta_lora/full_model")
model = AutoModelForSequenceClassification.from_pretrained("../models/deberta_lora/full_model").to(DEVICE)
model.eval()

print("Model ready.\n")

while True:
    text = input("Enter text (or type 'exit'): ")

    if text.lower() == "exit":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]

    print("\nProbabilities:")
    for label, prob in zip(LABEL_NAMES, probs):
        print(f"  {label:<20} {prob:.4f}")

    predicted = [label for label, prob in zip(LABEL_NAMES, probs) if prob > 0.5]
    print("\nPredicted labels (threshold=0.5):")
    for label in predicted:
        print(f"  ✓ {label}")
    if not predicted:
        print("  ✓ No violation detected")

    print("-" * 40)