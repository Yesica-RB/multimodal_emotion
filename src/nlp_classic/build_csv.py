# src/nlp_classic/build_csv.py
# Builds a clean CSV file from the raw MVSA-Single dataset.
# Each row contains: id, label, text (caption), image_path.
#
# The raw label file can have two annotations per item (e.g. "positive,neutral").
# When annotators disagree, we keep the first label (text annotator).

import os
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────
RAW_DIR    = "data/raw/data"        # folder with .jpg and .txt pairs
LABELS_TXT = "data/raw/labelResultAll.txt"  # raw annotation file
OUT_CSV    = "data/processed/labels.csv"    # output clean CSV

os.makedirs("data/processed", exist_ok=True)


def resolve_label(raw_label: str) -> str:
    """Resolve annotation conflicts.

    MVSA-Single has two annotators per item.
    - If both agree  → use that label.
    - If they differ → use the first label (text annotator).
    """
    parts = [p.strip().lower() for p in raw_label.split(",")]
    if len(parts) == 1:
        return parts[0]
    if parts[0] == parts[1]:
        return parts[0]   # both agree
    return parts[0]       # disagreement: keep text annotator label


# ── Read labels ─────────────────────────────────────────────────
rows = []
with open(LABELS_TXT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        item_id = parts[0].strip()
        label   = resolve_label(parts[1])
        if label not in ["positive", "negative", "neutral"]:
            continue   # skip invalid labels
        rows.append({"id": item_id, "label": label})

df_labels = pd.DataFrame(rows)

# ── Read captions and image paths ───────────────────────────────
captions, img_paths = [], []

for _, row in df_labels.iterrows():
    txt_path = os.path.join(RAW_DIR, f"{row['id']}.txt")
    jpg_path = os.path.join(RAW_DIR, f"{row['id']}.jpg")

    # Read tweet caption from .txt file
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            caption = f.read().strip().replace("\n", " ")
    except FileNotFoundError:
        caption = ""

    captions.append(caption)
    img_paths.append(jpg_path if os.path.exists(jpg_path) else "")

df_labels["text"]       = captions
df_labels["image_path"] = img_paths

# ── Filter out incomplete rows ───────────────────────────────────
# Remove rows with missing caption or missing image
df_labels = df_labels[df_labels["text"]       != ""]
df_labels = df_labels[df_labels["image_path"] != ""]

# ── Save ────────────────────────────────────────────────────────
df_labels.to_csv(OUT_CSV, index=False)
print(f"CSV saved to {OUT_CSV}")
print(df_labels["label"].value_counts())
print(f"Total rows: {len(df_labels)}")
