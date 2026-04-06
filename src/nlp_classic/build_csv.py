# src/nlp_classic/build_csv.py
import os
import pandas as pd

RAW_DIR    = "data/raw/data"
LABELS_TXT = "data/raw/labelResultAll.txt"
OUT_CSV    = "data/processed/labels.csv"

os.makedirs("data/processed", exist_ok=True)

def resolve_label(raw_label):
    """Si hay dos etiquetas separadas por coma, resuelve el conflicto."""
    parts = [p.strip().lower() for p in raw_label.split(",")]
    if len(parts) == 1:
        return parts[0]
    if parts[0] == parts[1]:
        return parts[0]       # coinciden → etiqueta clara
    # No coinciden → usamos la etiqueta del texto (primera columna)
    return parts[0]

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
            continue
        rows.append({"id": item_id, "label": label})

df_labels = pd.DataFrame(rows)

captions, img_paths = [], []
for _, row in df_labels.iterrows():
    txt_path = os.path.join(RAW_DIR, f"{row['id']}.txt")
    jpg_path = os.path.join(RAW_DIR, f"{row['id']}.jpg")
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            caption = f.read().strip().replace("\n", " ")
    except FileNotFoundError:
        caption = ""
    captions.append(caption)
    img_paths.append(jpg_path if os.path.exists(jpg_path) else "")

df_labels["text"]       = captions
df_labels["image_path"] = img_paths

df_labels = df_labels[df_labels["text"] != ""]
df_labels = df_labels[df_labels["image_path"] != ""]

df_labels.to_csv(OUT_CSV, index=False)
print(f"CSV guardado en {OUT_CSV}")
print(df_labels["label"].value_counts())
print(f"Total filas: {len(df_labels)}")