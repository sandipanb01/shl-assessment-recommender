# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 7860

# Run FastAPI directly (not Gradio)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
