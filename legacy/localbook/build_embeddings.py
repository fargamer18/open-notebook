import sqlite3
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DB = Path("/run/media/faraz/DATA/LocalBook/indexes/linear_equations.db")

INDEX_DIR = Path("/run/media/faraz/DATA/LocalBook/indexes/semantic")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH = INDEX_DIR / "pages.faiss"
META_PATH = INDEX_DIR / "pages.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME, device="cpu")

con = sqlite3.connect(DB)
con.row_factory = sqlite3.Row

rows = con.execute("""
    SELECT id, pdf, page, text
    FROM pages
    WHERE text IS NOT NULL
      AND length(trim(text)) >= 30
    ORDER BY id
""").fetchall()

con.close()

print(f"Embedding {len(rows)} pages...")

texts = []
metadata = []

for row in rows:
    text = " ".join((row["text"] or "").split())

    texts.append(text)

    metadata.append({
        "id": row["id"],
        "pdf": row["pdf"],
        "page": row["page"]
    })

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.asarray(embeddings, dtype="float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, str(FAISS_PATH))

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print()
print("Done.")
print(f"Vectors: {index.ntotal}")
print(f"Dimensions: {dimension}")
print(f"FAISS: {FAISS_PATH}")
print(f"Metadata: {META_PATH}")
