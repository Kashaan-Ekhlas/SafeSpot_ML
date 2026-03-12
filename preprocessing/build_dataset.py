import pandas as pd
from pathlib import Path
from datasets import load_dataset

POLICIES = [
    "S1_harassment",
    "S2_hate",
    "S3_violence",
    "S4_sexual",
    "S8_safe",
]
NUM_LABELS = 5


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip()


def print_label_distribution(df: pd.DataFrame, split_name: str):
    print(f"\n--- {split_name} ({len(df)} total rows) ---")
    labels_matrix = pd.DataFrame(df["labels"].tolist(), columns=POLICIES)
    for col in POLICIES:
        count = int(labels_matrix[col].sum())
        print(f"  {col:<20} {count:>7}  ({100 * count / len(df):.1f}%)")


def load_jigsaw(path: Path) -> list[dict]:
    df = pd.read_csv(path, encoding="utf-8")
    records = []
    for _, row in df.iterrows():
        text = clean_text(row["comment_text"])
        s1 = max(row["toxic"], row["severe_toxic"], row["insult"])
        s2 = row["identity_hate"]
        s3 = row["threat"]
        s4 = row["obscene"]
        s8 = int((s1 + s2 + s3 + s4) == 0)
        records.append({"text": text, "labels": [s1, s2, s3, s4, s8]})
    return records


def load_hatespeech(path: Path) -> list[dict]:
    # class 0 = hate speech, class 1 = offensive, class 2 = neither
    df = pd.read_csv(path, encoding="utf-8")
    records = []
    for _, row in df.iterrows():
        s1 = s2 = s3 = s4 = s8 = 0
        if row["class"] == 0:
            s2 = 1
        elif row["class"] == 1:
            s1 = 1
        else:
            s8 = 1
        records.append({"text": clean_text(row["tweet"]), "labels": [s1, s2, s3, s4, s8]})
    return records


def load_ucberkeley() -> list[dict]:
    # Raw rows (no averaging) — dedup by text to prevent train/val leakage
    # from the same comment appearing across splits.
    # First row per unique text is kept, preserving original annotator scores.
    # Thresholds match your original SQL: hatespeech > 1, violence > 2
    print("Loading ucberkeley-dlab/measuring-hate-speech...")
    df = load_dataset("ucberkeley-dlab/measuring-hate-speech")["train"].to_pandas()
    print(f"  Raw rows: {len(df)}")

    df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    print(f"  After text dedup: {len(df)}")

    is_hate     = df["hatespeech"] > 1
    is_violence = df["violence"] > 2
    print(f"  hatespeech > 1: {is_hate.sum()}  |  violence > 2: {is_violence.sum()}")

    selected = df[is_hate | is_violence].reset_index(drop=True)

    records = []
    for _, row in selected.iterrows():
        s1 = s2 = s3 = s4 = s8 = 0
        if row["hatespeech"] > 1:
            s2 = 1
        if row["violence"] > 2:
            s3 = 1
        records.append({"text": clean_text(row["text"]), "labels": [s1, s2, s3, s4, s8]})
    return records


def load_civil_comments() -> list[dict]:
    # Taking sexual_explicit, obscene -> S4
    #         identity_attack          -> S2 (hate supplement)
    #         threat                   -> S3 (violence supplement)
    # No comment_id column, dedup by text
    print("Loading google/civil_comments...")
    df = load_dataset("google/civil_comments")["train"].to_pandas()
    print(f"  Total rows: {len(df)}")

    mask_sexual   = df["sexual_explicit"] >= 0.5
    mask_obscene  = df["obscene"] >= 0.5
    mask_threat   = df["threat"] >= 0.5
    mask_identity = df["identity_attack"] >= 0.5

    selected = df[
        mask_sexual | mask_obscene | mask_threat | mask_identity
    ].drop_duplicates(subset="text").reset_index(drop=True)

    print(f"  Selected: {len(selected)}")

    records = []
    for _, row in selected.iterrows():
        s2 = int(row["identity_attack"] >= 0.5)
        s3 = int(row["threat"] >= 0.5)
        s4 = int(row["sexual_explicit"] >= 0.5 or row["obscene"] >= 0.5)
        records.append({"text": clean_text(row["text"]), "labels": [0, s2, s3, s4, 0]})
    return records


def main():
    base = Path("../datasets")

    print("=" * 60)
    data = []
    data += load_jigsaw(base / "jigsaw" / "train.csv")
    data += load_hatespeech(base / "hatespeech" / "labeled_data.csv")
    data += load_ucberkeley()
    data += load_civil_comments()

    df = pd.DataFrame(data)
    print_label_distribution(df, "Full dataset (pre-balancing)")

    SAFE_INDEX = 4
    safe_rows   = df[df["labels"].apply(lambda x: x[SAFE_INDEX] == 1)]
    unsafe_rows = df[df["labels"].apply(lambda x: x[SAFE_INDEX] == 0)]
    print(f"\nSafe (raw): {len(safe_rows)}  |  Unsafe (raw): {len(unsafe_rows)}")

    safe_rows = safe_rows.sample(n=min(len(safe_rows), len(unsafe_rows)), random_state=42)
    print(f"Safe (after 1:1 cap): {len(safe_rows)}")

    df = pd.concat([safe_rows, unsafe_rows]).sample(frac=1, random_state=42).reset_index(drop=True)
    print_label_distribution(df, "Full dataset (post-balancing)")

    n     = len(df)
    train = df.iloc[: int(0.8 * n)]
    val   = df.iloc[int(0.8 * n) : int(0.9 * n)]
    test  = df.iloc[int(0.9 * n) :]

    print_label_distribution(train, "Train split")
    print_label_distribution(val,   "Val split")
    print_label_distribution(test,  "Test split")

    Path("../processed").mkdir(exist_ok=True)
    train.to_json("../processed/train.jsonl", orient="records", lines=True)
    val.to_json("../processed/val.jsonl",     orient="records", lines=True)
    test.to_json("../processed/test.jsonl",   orient="records", lines=True)

    print("\nSaved processed datasets.")
    print("=" * 60)


if __name__ == "__main__":
    main()