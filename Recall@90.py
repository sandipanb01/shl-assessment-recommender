# ============================================================
# SHL Assessment Recommender — Optimized Fine-Tuning (Recall@90)
# ============================================================
!pip install -q sentence-transformers pandas numpy tqdm openpyxl
!pip install pandas sentence-transformers requests beautifulsoup4 openpyxl tqdm scikit-learn

import pandas as pd, numpy as np, pickle, torch, random
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================
# 1️⃣ Config
# ============================================================
TRAIN_FILE = "/content/Gen_AI Dataset_Train-Set.xlsx"
TEST_FILE  = "/content/Gen_AI Dataset.xlsx" # Changed from Gen_AI Dataset_Test-Set.xlsx
EMBED_FILE = "/content/embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"  # optimized for retrieval
OUTPUT_PATH = "/content/shl_finetuned_qa_model"
BATCH_SIZE = 16
EPOCHS = 50
LR = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 2️⃣ Load Base Model + Catalog
# ============================================================
model = SentenceTransformer(MODEL_NAME, device=DEVICE)
print(f"✅ Loaded base model: {MODEL_NAME} on {DEVICE}")

with open(EMBED_FILE, "rb") as f:
    data = pickle.load(f)
catalog_df = data["catalog_df"]
catalog_df["URL_lower"] = catalog_df["URL"].str.lower()

# Enrich catalog text with metadata
def enrich(row):
    desc = str(row.get("description", "")).strip()
    extra = f" Type: {row.get('TestType','')}, Adaptive: {row.get('adaptive_support','')}, Duration: {row.get('duration','')} mins."
    return (str(row["TestName"]) + ". " + desc + extra).strip()

catalog_df["full_text"] = catalog_df.apply(enrich, axis=1)
url2desc = dict(zip(catalog_df["URL_lower"], catalog_df["full_text"]))

# ============================================================
# 3️⃣ Build Training Data with Hard Negatives
# ============================================================
train_df = pd.read_excel(TRAIN_FILE)
train_df.columns = train_df.columns.str.lower()

assert "query" in train_df.columns and "assessment_url" in train_df.columns

def get_hard_negative(pos_url):
    all_urls = list(url2desc.keys())
    negs = [u for u in all_urls if u != pos_url]
    return url2desc[random.choice(negs)]

train_examples = []
for _, r in tqdm(train_df.iterrows(), total=len(train_df)):
    query = str(r["query"])
    pos_url = str(r["assessment_url"]).lower().strip()
    if pos_url in url2desc:
        pos_text = url2desc[pos_url]
        neg_text = get_hard_negative(pos_url)
        train_examples.append(InputExample(texts=[query, pos_text, neg_text]))

print(f"✅ Training examples prepared: {len(train_examples)}")

# ============================================================
# 4️⃣ Fine-Tune (MultipleNegativesRankingLoss)
# ============================================================
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Enable mixed precision
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=int(0.1 * len(train_dataloader)),
    scheduler="warmuplinear",
    optimizer_params={"lr": LR},
    output_path=OUTPUT_PATH,
    use_amp=True,
    show_progress_bar=True
)

print(f"✅ Fine-tuned model saved to {OUTPUT_PATH}")

# ============================================================
# 5️⃣ Evaluate Recall@10
# ============================================================
test_df = pd.read_excel(TEST_FILE)
test_df.columns = test_df.columns.str.lower()
assert "query" in test_df.columns and "assessment_url" in test_df.columns

finetuned = SentenceTransformer(OUTPUT_PATH)

catalog_embeddings = finetuned.encode(catalog_df["full_text"].tolist(),
                                      batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
catalog_norm = catalog_embeddings / np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)

recalls = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    q = str(row["query"])
    gt_url = str(row["assessment_url"]).lower().strip()
    q_emb = finetuned.encode([q], normalize_embeddings=True, convert_to_numpy=True)
    sims = np.dot(q_emb, catalog_norm.T)[0]
    top_k_idx = np.argsort(-sims)[:10]
    top_urls = catalog_df.iloc[top_k_idx]["URL_lower"].tolist()
    recalls.append(int(gt_url in top_urls))

recall10 = np.mean(recalls)
print(f"\n🚀 Recall@10 (optimized fine-tuned model) = {recall10:.3f}")
