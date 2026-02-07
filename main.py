from fastapi import FastAPI 
from pydantic import BaseModel # Structured data validation / Strict typing
from fastapi.middleware.cors import CORSMiddleware

class Content(BaseModel): 
    text: str

def load_wordlist(path: str) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        }

FILTER_WORDS = load_wordlist("data/en.txt")

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/api/moderate")
def moderate(payload: Content):
    if not payload.text.strip():
        return {
            "action": "takedown",
            "reason": "empty_content",
            "matched_count": 0
        }
    text = payload.text.lower()
    tokens = text.split()
    matched = [word for word in tokens if word in FILTER_WORDS]
    if matched:
        return {
            "action": "takedown",
            "reason": "blocked_terms_found",
            "matched_count": len(matched)
        }

    return {
        "action": "allow",
        "reason": "no_blocked_terms_found",
        "matched_count": 0
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
