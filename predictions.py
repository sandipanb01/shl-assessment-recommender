!pip install -q sentence-transformers pandas numpy openpyxl tqdm bs4 uvicorn

import os, re, pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from bs4 import BeautifulSoup # Added Beautiful Soup import

# -----------------------
# Config (edit as needed)
# -----------------------
EMBED_FILE = "/content/embeddings.pkl"                        # uploaded embeddings.pkl
TEST_FILE = "/content/Gen_AI Dataset_Test-Set.xlsx"           # uploaded test file (unlabeled)
OUTPUT_FILENAME = "/content/sandipan_bhattacherjee.csv"           # <-- change to your firstname_lastname.csv
MODEL_NAME = "all-MiniLM-L6-v2"                               # fast; change if needed
TOP_K = 10                                                    # desired number of recs per query (5-10)
MIN_K = 5
MAX_K = 10

assert MIN_K <= TOP_K <= MAX_K, f"TOP_K must be between {MIN_K} and {MAX_K}"

# -----------------------
# Load SentenceTransformer
# -----------------------
print("Loading model...")
model = SentenceTransformer(MODEL_NAME)

# -----------------------
# Load embeddings.pkl robustly
# -----------------------
def load_pickle_catalog(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path + " not found")
    with open(path, "rb") as f:
        data = pickle.load(f)

    catalog_df = None
    catalog_embeddings = None

    # Common expected structure: dict with keys 'catalog_df' and 'catalog_embeddings'
    if isinstance(data, dict):
        if "catalog_df" in data and "catalog_embeddings" in data:
            catalog_df = pd.DataFrame(data["catalog_df"]).reset_index(drop=True)
            catalog_embeddings = np.asarray(data["catalog_embeddings"], dtype=np.float32)
        else:
            # try to find plausible df & array in values
            for v in data.values():
                if catalog_df is None:
                    try:
                        tmp = pd.DataFrame(v)
                        # require at least a URL or TestName-like column
                        cols_lower = [c.lower() for c in tmp.columns]
                        if any(x in cols_lower for x in ("url", "testname", "test_name", "testname")):
                            catalog_df = tmp.reset_index(drop=True)
                    except Exception:
                        pass
                if catalog_embeddings is None:
                    try:
                        arr = np.asarray(v)
                        if arr.ndim == 2:
                            catalog_embeddings = arr.astype(np.float32)
                    except Exception:
                        pass
    elif isinstance(data, (list, tuple)):
        # If a list of dicts, try to convert to df and find embeddings separately
        try:
            cand = pd.DataFrame(data)
            if any(c.lower() == "url" for c in cand.columns):
                catalog_df = cand.reset_index(drop=True)
        except Exception:
            pass

    if catalog_df is None or catalog_embeddings is None:
        raise ValueError("Could not extract catalog_df and catalog_embeddings from embeddings.pkl")

    # normalize URL column name to 'URL' and ensure TestName exists where possible
    for c in catalog_df.columns:
        if c.lower() == "url":
            catalog_df = catalog_df.rename(columns={c: "URL"})
            break
    # Precompute normalized embeddings for cosine sim
    norms = np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    catalog_norm = (catalog_embeddings / norms).astype(np.float32)

    return catalog_df, catalog_embeddings, catalog_norm

print("Loading embeddings.pkl...")
catalog_df, catalog_embeddings, catalog_norm = load_pickle_catalog(EMBED_FILE)
print(f"Catalog rows: {len(catalog_df)}, embeddings dim: {catalog_embeddings.shape[1]}")

# -----------------------
# Filter out "Pre-packaged Job Solutions" (best-effort)
# -----------------------
def filter_prepackaged(df):
    # common markers: 'pre-packaged', 'pre packaged', 'job solution', 'pre-pack', 'packaged job'
    mask = pd.Series([True] * len(df))
    name_cols = [c for c in df.columns if "test" in c.lower() or "name" in c.lower()]
    # create a concatenated name field for matching
    def _name_text(r):
        for c in name_cols:
            v = r.get(c, "")
            if isinstance(v, str) and v.strip():
                return v.lower()
        return ""
    texts = df.apply(_name_text, axis=1)
    exclude_keywords = ["pre-packaged", "pre packaged", "job solution", "pre-pack", "packaged job", "prepackaged"]
    for kw in exclude_keywords:
        mask &= ~texts.str.contains(kw, na=False)
    return df[mask.values].reset_index(drop=True)

catalog_df = filter_prepackaged(catalog_df)
print(f"After filtering pre-packaged job solutions: {len(catalog_df)} rows remain")

# -----------------------
# Type detection (if needed) - minimal mapping using TestName/TestType fields
# -----------------------
def detect_type_from_row(name: str, raw_type: str = "") -> str:
    n = (name or "").lower()
    rt = (raw_type or "").lower()
    if any(k in n for k in ["opq", "personality", "behaviour", "behavior", "sjt", "situational"]):
        return "P"
    if any(k in n for k in ["verify", "numerical", "verbal", "reasoning", "ability", "aptitude"]):
        return "A"
    if any(k in n for k in ["python","java","coding","developer","technical","sql","data"]):
        return "K"
    if "job focused" in n or "manager" in n or "leadership" in n:
        return "C"
    return "K"

catalog_df["TNorm"] = catalog_df.apply(lambda r: detect_type_from_row(str(r.get("TestName","")), str(r.get("TestType",""))), axis=1)

# -----------------------
# Utility: recommend per query using embeddings similarity + simple balancing
# -----------------------
def recommend_for_query(qtext: str, K=TOP_K):
    # Handle URL input by scraping text (optional) - skip scrapping here for speed
    q = qtext.strip()
    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    sims = (q_emb @ catalog_norm.T)[0]
    tmp = catalog_df.copy().reset_index(drop=True)
    tmp["score"] = sims

    # naive intent detection for tech vs behaviour
    ql = q.lower()
    wants_tech = any(x in ql for x in ["python","java","sql","coding","developer","technical","data","ml","ai"])
    wants_beh = any(x in ql for x in ["personality","behaviour","behavior","leadership","collaboration","team","communication","stakeholder"])

    # produce separate sorted lists
    tech_df = tmp[tmp["TNorm"].isin(["K","A"])].sort_values("score", ascending=False)
    beh_df = tmp[tmp["TNorm"].isin(["P","C","B"])].sort_values("score", ascending=False)

    K = max(MIN_K, min(MAX_K, int(K)))
    if wants_tech and wants_beh:
        tech_k = max(1, int(round(K * 0.6)))
        beh_k = K - tech_k
    elif wants_tech:
        tech_k = K; beh_k = 0
    elif wants_beh:
        tech_k = 0; beh_k = K
    else:
        tech_k = K // 2; beh_k = K - tech_k

    picks = []
    seen = set()
    # pick tech
    for _, r in tech_df.iterrows():
        if len([p for p in picks if p.get("TNorm") in ("K","A")]) >= tech_k:
            break
        url = str(r.get("URL","")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        picks.append(r.to_dict())
    # pick beh
    for _, r in beh_df.iterrows():
        if len([p for p in picks if p.get("TNorm") in ("P","C","B")]) >= beh_k:
            break
        url = str(r.get("URL","")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        picks.append(r.to_dict())
    # fill remaining
    if len(picks) < K:
        for _, r in tmp.sort_values("score", ascending=False).iterrows():
            if len(picks) >= K:
                break
            url = str(r.get("URL","")).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            picks.append(r.to_dict())

    # format minimal output list of URLs
    urls = [p.get("URL","").strip() for p in picks[:K] if p.get("URL","").strip()]
    # ensure at least 1 URL returned
    if not urls:
        # fallback top-1 from catalog by sim
        idx = int(np.argmax((q_emb @ catalog_norm.T)[0]))
        urls = [str(catalog_df.iloc[idx]["URL"]).strip()]
    return urls

# -----------------------
# Load test queries and generate recommendations
# -----------------------
print("Loading test queries...")
if not os.path.exists(TEST_FILE):
    raise FileNotFoundError("Test file not found: " + TEST_FILE)
test_df = pd.read_excel(TEST_FILE, engine="openpyxl")
test_df.columns = [c.strip().lower() for c in test_df.columns]
if "query" not in test_df.columns:
    raise ValueError("Test file must contain column 'query'")
queries = test_df["query"].astype(str).tolist()
print(f"{len(queries)} queries loaded")

rows_out = []
for q in tqdm(queries, desc="Generating"):
    urls = recommend_for_query(q, K=TOP_K)
    # append one row per URL (Query repeats)
    for u in urls:
        rows_out.append({"Query": q.strip(), "Assessment_url": u})

# Final DF and dedupe exact duplicates (but preserve order)
final_df = pd.DataFrame(rows_out, columns=["Query", "Assessment_url"])
final_df = final_df[final_df["Query"].ne("") & final_df["Assessment_url"].ne("")]
final_df = final_df.drop_duplicates(subset=["Query", "Assessment_url"], keep="first").reset_index(drop=True)

# Save with EXACT header "Query,Assessment_url" and no index
final_df.to_csv(OUTPUT_FILENAME, index=False, encoding="utf-8") # Removed line_terminator="\n"

print("WROTE:", OUTPUT_FILENAME)
print("Unique queries:", final_df["Query"].nunique(), "Total rows:", len(final_df))
print("\nPreview (first 20 rows):")
print(final_df.head(20).to_string(index=False))
