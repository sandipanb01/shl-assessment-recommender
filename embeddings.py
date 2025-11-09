# embeddings.py — Build Embeddings for SHL Assessment Catalog
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
OUTPUT_FILE = "embeddings.pkl"
INPUT_CSV = "shl_catalogue.csv"   #scraped or cleaned SHL catalog
TEXT_COLUMNS = ["TestName", "description"]

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=["TestName", "description"]).reset_index(drop=True)
print(f"✅ Loaded catalog: {len(df)} entries")

# ------------------------------------------------------------
# Build embeddings
# ------------------------------------------------------------
model = SentenceTransformer(MODEL_NAME)
texts = (df["TestName"].astype(str) + ". " + df["description"].astype(str)).tolist()
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

print(f"✅ Created embeddings: shape={embeddings.shape}")

# ------------------------------------------------------------
# Save pickle
# ------------------------------------------------------------
data = {
    "catalog_df": df,
    "catalog_embeddings": embeddings
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"✅ Saved embeddings to {OUTPUT_FILE}")
