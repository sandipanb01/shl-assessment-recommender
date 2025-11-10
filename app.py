# app.py — SHL Assessment Recommender (Final Submission-Ready Version)
import os, pickle, re, requests, math, sys, numpy as np
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import uvicorn

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
EMBED_FILE = "embeddings.pkl"   # created by build_embeddings.py or provided
MODEL_NAME = "all-MiniLM-L6-v2"
HF_SPACE_URL = "https://huggingface.co/spaces/sandipanb01/shl-assessment-recommender"
MIN_K = 1
MAX_K = 10

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
model = SentenceTransformer(MODEL_NAME)

# ------------------------------------------------------------
# -------- robust embeddings loader (integrated version) --------
# ------------------------------------------------------------
catalog_df = None
catalog_embeddings = None
_norm_catalog_embeddings = None  # normalized numpy for fast cos-sim if needed

if os.path.exists(EMBED_FILE):
    try:
        with open(EMBED_FILE, "rb") as f:
            data = pickle.load(f)
        # try common key names
        if isinstance(data, dict):
            # prefer 'catalog_df' and 'catalog_embeddings' keys
            if 'catalog_df' in data and 'catalog_embeddings' in data:
                catalog_df = data['catalog_df']
                catalog_embeddings = data['catalog_embeddings']
            else:
                # attempt to find dataframe/array inside dict
                for k, v in data.items():
                    if catalog_df is None and (k.lower().find("catalog") >= 0 or isinstance(v, (list, pd.DataFrame))):
                        try:
                            tmp = pd.DataFrame(v) if not isinstance(v, pd.DataFrame) else v
                            if 'URL' in tmp.columns or 'TestName' in tmp.columns:
                                catalog_df = tmp
                        except Exception:
                            pass
                    if catalog_embeddings is None and (isinstance(v, (np.ndarray, list)) or str(type(v)).lower().find("tensor") >= 0):
                        try:
                            arr = np.asarray(v)
                            if arr.ndim == 2:
                                catalog_embeddings = arr
                        except Exception:
                            pass

        # Final checks & conversions
        if catalog_df is None or catalog_embeddings is None:
            raise ValueError("embeddings.pkl missing 'catalog_df' or 'catalog_embeddings' keys (or unreadable)")

        # Clean TestName values in catalog_df to remove trailing site suffixes like " | SHL"
        try:
            if isinstance(catalog_df, pd.DataFrame) and "TestName" in catalog_df.columns:
                catalog_df["TestName"] = catalog_df["TestName"].astype(str).str.strip()
                catalog_df["TestName"] = catalog_df["TestName"].str.replace(r"\s*\|\s*SHL\s*$", "", regex=True, case=False)
                # also trim any trailing " | <something>" that looks like a site suffix
                catalog_df["TestName"] = catalog_df["TestName"].str.replace(r"\s*\|\s*[A-Za-z0-9\-\s]{1,20}\s*$", "", regex=True)
        except Exception:
            pass

        # If embeddings are torch tensor, convert to numpy
        try:
            import torch
            if 'torch' in str(type(catalog_embeddings)).lower() or isinstance(catalog_embeddings, torch.Tensor):
                catalog_embeddings = catalog_embeddings.detach().cpu().numpy()
        except Exception:
            pass

        # Ensure numpy dtype float32
        if isinstance(catalog_embeddings, list):
            catalog_embeddings = np.asarray(catalog_embeddings, dtype=np.float32)
        elif isinstance(catalog_embeddings, np.ndarray):
            catalog_embeddings = catalog_embeddings.astype(np.float32, copy=False)
        else:
            catalog_embeddings = np.asarray(catalog_embeddings, dtype=np.float32)

        # Precompute L2-normalized copy for cosine similarity (numpy)
        norms = np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _norm_catalog_embeddings = (catalog_embeddings / norms).astype(np.float32)

        print(f"✅ Loaded embeddings.pkl: rows={catalog_embeddings.shape[0]}, dim={catalog_embeddings.shape[1]}")
    except Exception as e:
        print("⚠️ Failed to load embeddings.pkl:", e, file=sys.stderr)
        catalog_df = None
        catalog_embeddings = None
        _norm_catalog_embeddings = None
else:
    print("⚠️ embeddings.pkl not found in repo root. Using fallback minimal catalog.")

# Safe fallback minimal catalog (so endpoints always work)
if catalog_df is None:
    catalog_df = pd.DataFrame([{
        "TestName": "Occupational Personality Questionnaire (OPQ)",
        "URL": "https://www.shl.com/solutions/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
        "description": "Occupational Personality Questionnaire (OPQ32r) — behavioural assessment.",
        "duration": 25, "adaptive_support": "No", "remote_support": "Yes", "TestType": "P"
    }])
    catalog_embeddings = model.encode((catalog_df["TestName"] + ". " + catalog_df["description"]).tolist(),
                                      convert_to_tensor=True)
    # convert to numpy for consistent behavior
    try:
        catalog_embeddings = np.asarray(catalog_embeddings)
        if hasattr(catalog_embeddings, "dtype") and str(catalog_embeddings.dtype).startswith("float"):
            catalog_embeddings = catalog_embeddings.astype(np.float32, copy=False)
    except Exception:
        pass

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def clean_description(text: str) -> str:
    if not text:
        return "No description available."
    s = str(text)
    s = re.sub(r"\s+", " ", s).strip()
    cut_markers = ["Accelerate Your Talent Strategy", "Back to Product Catalog", "Support chat", "Book a Demo"]
    for m in cut_markers:
        idx = s.find(m)
        if idx != -1:
            s = s[:idx].strip()
    return s[:800].rstrip()

def extract_text_from_url(url: str, timeout=6) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","header","footer","nav","svg","aside","noscript"]):
            tag.extract()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p","h1","h2","h3","li"])]
        if not paragraphs:
            divs = [d.get_text(" ", strip=True) for d in soup.find_all("div")]
            paragraphs = [d for d in divs if len(d.split()) > 20]
        text = " ".join(paragraphs)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:6000]
    except Exception:
        return ""

def extract_skills(text: str):
    t = (text or "").lower()
    # Expanded keyword coverage for AI/ML, teamwork, leadership etc.
    TECH_KEYWORDS = {
        "python","java","c++","sql","javascript","aws","docker","kubernetes",
        "ml","machine learning","artificial intelligence","ai","data","pandas",
        "spark","tensorflow","pytorch","neural","deep learning"
    }
    SOFT_KEYWORDS = {
        "communication","collaboration","teamwork","stakeholder","client",
        "lead","leadership","motivation","team","adapt","problem","present",
        "interpersonal","influence","empathy","behavior","behaviour"
    }

    tech = [k for k in TECH_KEYWORDS if k in t]
    soft = [k for k in SOFT_KEYWORDS if k in t]

    return list(dict.fromkeys(tech)), list(dict.fromkeys(soft))

# ------------------------------------------------------------
# Improved classification logic with coverage for Ability, Behavioural, etc.
# ------------------------------------------------------------
BEHAV_SLUGS = [
    "opq", "sjt", "situational", "judgement", "motivation",
    "leadership", "personality", "behavioral", "behaviour", "enterprise"
]
ABILITY_SLUGS = [
    "cognitive", "ability", "aptitude", "reasoning", "numerical",
    "deductive", "verbal", "inductive", "logical", "spatial",
    "verify", "interact", "interactive"
]

# ✅ Added hybrid mapping for SHL Job-Focused & Professional assessments
# ✅ Improved hybrid + explicit type mapping for SHL tests
SPECIAL_HYBRID_NAMES = {
    # --- Hybrid / Competency + Behavioural ---
    "technology professional 8.0 job focused assessment": "C",
    "technology professional 8.8 job focused assessment": "C",
    "technology professional job focused assessment": "C",
    "technology professional": "C",
    "manager 8.0 job focused assessment": "C",
    "manager 8.0+ jfa": "C",
    "leadership 8.0 jfa": "C",
    "leadership job focused assessment": "C",
    "professional 8.0": "C",
    "professional 8.8": "C",
    "enterprise job focused": "C",

    # --- Pure Behavioural / Personality ---
    "opq": "P",             # Occupational Personality Questionnaire
    "occupational personality questionnaire": "P",
    "motivation": "P",
    "sjt": "P",             # Situational Judgement Test
    "situational judgement": "P",
    "behavioral": "P",
    "behaviour": "P",

    # --- Cognitive / Ability / Aptitude ---
    "verify": "A",          # SHL Verify tests
    "inductive reasoning": "A",
    "deductive reasoning": "A",
    "numerical reasoning": "A",
    "verbal reasoning": "A",
    "ability": "A",
    "aptitude": "A",

    # --- Knowledge / Technical tests ---
    "coding": "K",
    "developer": "K",
    "programming": "K",
    "technical test": "K",
    "data": "K",
    "sql": "K",
    "python": "K",
    "java": "K"
}

def detect_type_from_row(name: str, raw_type: str = "") -> str:
    n = (name or "").lower().strip()
    rt = str(raw_type or "").lower()

    # ✅ Check both name and raw_type for known SHL mappings
    for key, code in SPECIAL_HYBRID_NAMES.items():
        if key in n or key in rt:
            return code

    # Explicit hints from type field
    if "p" in rt and not ("present" in rt):
        return "P"
    if "a" in rt and not any(b in rt for b in ["adaptive", "admin"]):
        return "A"

    # --- Competency & hybrid (Job Focused / Professional / Leadership) ---
    if any(k in n for k in [
        "job focused", "job-focused", "professional", "enterprise",
        "leadership", "competency", "competencies", "managerial"
    ]):
        return "C"

    # --- Behavioural / Personality / Situational ---
    if any(k in n for k in [
        "opq", "sjt", "situational", "judgement", "motivation",
        "behavioral", "behaviour", "personality"
    ]):
        return "P"

    # --- Cognitive / Ability / Aptitude ---
    if any(k in n for k in [
        "cognitive", "ability", "aptitude", "reasoning", "numerical",
        "verbal", "logical", "deductive", "inductive", "spatial",
        "interactive", "verify"
    ]):
        return "A"

    # --- Technical / Knowledge-based tests ---
    if any(k in n for k in [
        "python", "java", "sql", "coding", "developer", "technical",
        "data", "programming", "software", "machine learning", "cloud"
    ]):
        return "K"

    # Default fallback
    return "K"

# ------------------------------------------------------------
# Recommendation Core
# ------------------------------------------------------------
def recommend_for_query(query: str, K: int = 10) -> Dict[str, Any]:
    qtext = query
    if isinstance(query, str) and query.strip().lower().startswith("http"):
        scraped = extract_text_from_url(query)
        qtext = scraped if scraped else query

    tech_sk, soft_sk = extract_skills(qtext)
    wants_tech = len(tech_sk) > 0
    wants_beh = len(soft_sk) > 0

    # ✅ Boost hybrid assessments if both tech + behavior skills detected
    hybrid_boost = wants_tech and wants_beh

    # ✅ Detect hybrid intent (when user wants both cognitive & personality)
    # ✅ Improved hybrid intent detection — broader and more realistic
    hybrid_intent = False
    q_low = qtext.lower()
    if (
        any(x in q_low for x in ["cognitive", "aptitude", "ability", "reasoning", "numerical", "verbal"]) or
        any(x in q_low for x in ["ai", "ml", "data", "analyst", "engineer"])
    ) and (
        any(x in q_low for x in ["personality", "behavior", "behaviour", "competency", "leadership", "motivation", "teamwork", "collaboration"])
    ):
        hybrid_intent = True

    # --- robust similarity computation (integrated) ---
    q_emb = model.encode([qtext], convert_to_tensor=True)

    sims = None
    try:
        # if catalog_embeddings is a tensor-like (sentence-transformers tensor) use util.cos_sim directly
        if hasattr(catalog_embeddings, "shape") and not isinstance(catalog_embeddings, np.ndarray):
            sims = util.cos_sim(q_emb, catalog_embeddings)[0].cpu().numpy()
        else:
            q_np = q_emb.cpu().numpy().astype(np.float32) if hasattr(q_emb, "cpu") else np.asarray(q_emb, dtype=np.float32)
            q_np = q_np / (np.linalg.norm(q_np, axis=1, keepdims=True) + 1e-12)
            if _norm_catalog_embeddings is None:
                ce = np.asarray(catalog_embeddings, dtype=np.float32)
                ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-12)
                sims = (q_np @ ce.T)[0]
            else:
                sims = (q_np @ _norm_catalog_embeddings.T)[0]
    except Exception as e:
        try:
            sims = util.cos_sim(model.encode([qtext], convert_to_tensor=True), catalog_embeddings)[0].cpu().numpy()
        except Exception as e2:
            raise RuntimeError(f"Failed to compute similarity: {e} / {e2}")

    # --- ranking and type handling ---
    df = catalog_df.copy().reset_index(drop=True)
    df["score"] = sims

    # Apply hybrid boosts (job-focused / JFA) when tech+beh intent detected
    if hybrid_boost and "job focused" in " ".join(df["TestName"].str.lower()):
        df.loc[df["TestName"].str.lower().str.contains("job focused"), "score"] *= 1.3
        df.loc[df["TestName"].str.lower().str.contains("jfa"), "score"] *= 1.3

    if "TestType" not in df.columns and "test_type" in df.columns:
        df["TestType"] = df["test_type"]
    if "TestType" not in df.columns:
        df["TestType"] = ""

    df["TNorm"] = df.apply(lambda r: detect_type_from_row(r.get("TestName",""), r.get("TestType","")), axis=1)

    # If hybrid intent is detected, boost 'C' type assessments and re-sort
    if hybrid_intent:
        # Stronger hybrid boost to ensure C-type assessments dominate top results
        df.loc[df["TNorm"] == "C", "score"] *= 1.5
        # Also promote top OPQ/SJT items slightly since they pair with cognitive
        df.loc[df["TNorm"].isin(["P", "B"]), "score"] *= 1.2
        df = df.sort_values("score", ascending=False)
        tech_df = df[df["TNorm"].isin(["K","A","S"])].sort_values("score", ascending=False)
        beh_df = df[df["TNorm"].isin(["P","B","C"])].sort_values("score", ascending=False)
    else:
        tech_df = df[df["TNorm"].isin(["K","A","S"])].sort_values("score", ascending=False)
        beh_df = df[df["TNorm"].isin(["P","B","C"])].sort_values("score", ascending=False)


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

    picks, seen_urls = [], set()
    for _, row in tech_df.iterrows():
        if len([p for p in picks if p.get("TNorm") in ["K","A","S"]]) >= tech_k:
            break
        url = str(row.get("URL",""))
        if url in seen_urls:
            continue
        picks.append(row.to_dict()); seen_urls.add(url)
    for _, row in beh_df.iterrows():
        if len([p for p in picks if p.get("TNorm") in ["P","B","C"]]) >= beh_k:
            break
        url = str(row.get("URL",""))
        if url in seen_urls:
            continue
        picks.append(row.to_dict()); seen_urls.add(url)
    if len(picks) < K:
        for _, row in df.sort_values("score", ascending=False).iterrows():
            if len(picks) >= K:
                break
            url = str(row.get("URL",""))
            if url in seen_urls:
                continue
            picks.append(row.to_dict()); seen_urls.add(url)

    TYPE_MAPPING = {
        "K": ["Knowledge & Skills"],
        "P": ["Personality & Behaviour"],
        "A": ["Ability & Aptitude"],
        "B": ["Biodata & Situational Judgement"],
        "C": ["Competencies", "Personality & Behaviour"],  # ✅ SHL hybrid alignment
        "S": ["Simulations"]
    }

    def format_description(text):
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(parts[:2])

    out = []
    for p in picks[:K]:
        tcode = str(p.get("TNorm","K"))
        test_type_list = TYPE_MAPPING.get(tcode, ["Knowledge & Skills"])

        # Clean name (remove "| SHL" and common site suffixes)
        raw_name = str(p.get("TestName","") or "")
        name = re.sub(r"\s*\|\s*SHL\s*$", "", raw_name, flags=re.IGNORECASE).strip()
        name = re.sub(r"\s*\|\s*[A-Za-z0-9\-\s]{1,20}\s*$", "", name).strip()

        out.append({
            "url": str(p.get("URL","")),
            "name": name,
            "adaptive_support": str(p.get("adaptive_support","No")),
            "description": format_description(p.get("description", "")),
            "duration": int(p.get("duration", 20) or 20),
            "remote_support": str(p.get("remote_support","Yes")),
            "test_type": test_type_list
        })

    explanation = {
        "query_text": qtext,
        "tech_skills": tech_sk,
        "soft_skills": soft_sk,
        "k_total": K,
        "k_tech": tech_k,
        "k_behavior": beh_k
    }

    return {"recommended_assessments": out, "explanation": explanation}

# ------------------------------------------------------------
# FastAPI endpoints
# ------------------------------------------------------------
app = FastAPI(title="SHL Assessment Recommender (Submission-ready)")

class RecommendRequest(BaseModel):
    query: str
    k: int = 10

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(req: RecommendRequest):
    if not isinstance(req.query, str) or not req.query.strip():
        raise HTTPException(status_code=400, detail="query must be a non-empty string")
    res = recommend_for_query(req.query, K=req.k)
    return {
        "query": req.query,
        "recommended_assessments": res["recommended_assessments"]
    }

# ------------------------------------------------------------
# Run with uvicorn when executed directly (useful locally)
# Hugging Face Docker runtime will import `app` and serve it automatically.
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, log_level="info")
