from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/deberta_lora/full_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_NAMES = ["S1_harassment", "S2_hate", "S3_violence", "S4_sexual", "S8_safe"]

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print("Model loaded.")

def load_wordlist(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        }

FILTER_WORDS = load_wordlist("data/en.txt")

class Content(BaseModel):
    text: str
    context: str | None = None  # i was gonna have llama analyse the context but it got lost in the shufflin of development so maybe later

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/api/moderate")
def moderate(payload: Content):
    text = payload.text.strip()

    if not text:
        return {
            "action": "takedown",
            "reason": "empty_content",
            "matched_count": 0
        }
    tokens = text.lower().split()
    matched = [word for word in tokens if word in FILTER_WORDS]

    if matched:
        return {
            "action": "takedown",
            "reason": "blocked_terms_found",
            "matched_count": len(matched)
        }
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        probs = torch.sigmoid(model(**inputs).logits)[0].cpu().numpy()

    predictions = [
    label for label, prob in zip(LABEL_NAMES, probs) if prob > 0.6
]

    # Remove S8_safe from violations
    violations = [p for p in predictions if p != "S8_safe"]

    if violations:
        return {
        "action": "takedown",
        "reason": violations[0],
        "matched_count": len(violations)
         }

    return {
        "action": "allow",
        "reason": "safe",
        "matched_count": 0
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)