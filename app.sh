#!/bin/bash
# Launch both FastAPI (for SHL endpoints) and Gradio (for UI)

echo "Starting SHL Assessment Recommender API..."

# Start FastAPI in background on port 8000
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Start Gradio UI on port 7860 (main Hugging Face interface)
python app.py
