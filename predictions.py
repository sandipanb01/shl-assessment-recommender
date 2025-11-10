!pip install pandas sentence-transformers requests beautifulsoup4 openpyxl tqdm scikit-learn
!pip install requests-html lxml_html_clean
# ============================================================
# generate_predictions.py — Generate Recommendations for SHL
# ============================================================
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# -----------------------------
# Configurations
# -----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_FILE = "embeddings.pkl"
TEST_FILE = "Gen_AI Dataset.xlsx"
OUTPUT_FILE = "predictions.csv"

# -----------------------------
# Load model + embeddings
# -----------------------------
print("Loading model and embeddings...")
model = SentenceTransformer(MODEL_NAME)

with open(EMBED_FILE, "rb") as f:
    data = pickle.load(f)
catalog_df = data["catalog_df"]
catalog_embeddings = np.array(data["catalog_embeddings"], dtype=np.float32)

# Normalize
catalog_norm = catalog_embeddings / np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)

# -----------------------------
# Load test data
# -----------------------------
test_df = pd.read_excel(TEST_FILE)
test_df.columns = test_df.columns.str.lower() # Convert column names to lowercase
assert "query" in test_df.columns, "Missing 'query' column in test file"

results = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    query = str(row["query"])

    # Encode query
    q_emb = model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
    sims = np.dot(q_emb, catalog_norm.T)

    # Get top 10
    top_k_idx = np.argsort(-sims[0])[:10]
    recs = catalog_df.iloc[top_k_idx][["TestName", "URL"]].copy()
    recs["query"] = query
    recs["rank"] = range(1, len(recs) + 1)
    results.append(recs)

# Combine all predictions
pred_df = pd.concat(results, ignore_index=True)
pred_df = pred_df[["query", "rank", "TestName", "URL"]]
pred_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Saved predictions to {OUTPUT_FILE}")
