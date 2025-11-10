# 🧠 SHL Assessment Recommender — FastAPI App

This project is a semantic AI recommender system that suggests the most relevant **SHL assessments** based on a job description, query, or URL.  
It runs as both a **FastAPI endpoint** (for API evaluation) and a **Hugging Face Space** (for user interaction).

---

## ⚙️ API Endpoint

### **Base URL**
https://sandipanb01-shl-assessment-recommender.hf.space/

### **Full Endpoint URL**

> ⚠️ Note: This is a **POST-only** endpoint.  
> Opening it directly in a browser will show `"Method Not Allowed"`, which is the expected behavior.  
> You must send a JSON body (see below) using a tool like `curl`, `Postman`, or Python `requests`.

---

## 🧾 Example Request (POST)

```bash
curl -X POST "https://sandipanb01-shl-assessment-recommender.hf.space/recommend" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "I am hiring an AI engineer and want to screen using both cognitive and personality tests",
           "k": 5
         }'
🧠 Example Response (JSON)
{
  "query": "I am hiring an AI engineer and want to screen using both cognitive and personality tests",
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/solutions/products/product-catalog/view/technology-professional-8-e-job-focused-assessment/",
      "name": "Technology Professional 8.8 Job Focused Assessment",
      "adaptive_support": "No",
      "description": "Assesses key behavioural and cognitive competencies required for success in fast-paced technical roles.",
      "duration": 15,
      "remote_support": "Yes",
      "test_type": ["Competencies", "Personality & Behaviour"]
    },
    {
      "url": "https://www.shl.com/solutions/products/product-catalog/view/occupational-personality-questionnaire-opq32r/",
      "name": "Occupational Personality Questionnaire (OPQ32r)",
      "adaptive_support": "No",
      "description": "Measures key personality traits that impact workplace behaviour and cultural fit.",
      "duration": 25,
      "remote_support": "Yes",
      "test_type": ["Personality & Behaviour"]
    }
  ]
}
