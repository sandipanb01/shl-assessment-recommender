# 🧠 SHL Assessment Recommender

**Live App:** [https://sandipanb01-shl-assessment-recommender.hf.space](https://sandipanb01-shl-assessment-recommender.hf.space)  
**API Endpoint:** [https://sandipanb01-shl-assessment-recommender.hf.space/recommend](https://sandipanb01-shl-assessment-recommender.hf.space/recommend)

---

## 🚀 Overview

This project implements an **AI-based SHL Assessment Recommender** system that automatically recommends the most relevant SHL assessments for a given **job description**, **query**, or **JD URL**.  
It uses **Sentence Transformers** (`all-MiniLM-L6-v2`) to generate embeddings and **cosine similarity** to find the closest matching SHL catalog entries.

---

## ⚙️ Tech Stack

- 🧩 **FastAPI** – for backend API  
- 🎨 **Gradio** – for interactive web interface  
- 🤗 **Sentence Transformers (MiniLM-L6-v2)** – for text embeddings  
- 🧠 **Python** (NumPy, pandas, BeautifulSoup, requests)

---

## 🧩 Key Features

✅ Query using free text or job description URLs  
✅ Returns structured JSON with SHL test details:  
  - `name`  
  - `description`  
  - `duration`  
  - `adaptive_support`  
  - `remote_support`  
  - `test_type`  

✅ Covers multiple SHL test families:  
  - Knowledge & Skills  
  - Personality & Behaviour  
  - Ability & Aptitude  
  - Competencies (Hybrid)  

✅ Fully deployable on **Hugging Face Spaces**

---

## 🧠 API Endpoint

### POST `/recommend`

**Example Request:**
```bash
curl -X POST "https://sandipanb01-shl-assessment-recommender.hf.space/recommend" \
     -H "Content-Type: application/json" \
     -d '{"query": "Technology Professional 8.8 Job Focused Assessment", "k": 10}'
#Response example-----
{
  "query": "Technology Professional 8.8 Job Focused Assessment",
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/solutions/products/product-catalog/view/technical-sales-associate-solution/",
      "name": "Technical Sales Associate Solution",
      "adaptive_support": "No",
      "description": "The Technical Sales Associate solution is for entry-level retail positions...",
      "duration": 41,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills"]
    }
  ]
}
