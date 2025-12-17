FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fastapi_server/ ./fastapi_server/

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "fastapi_server.main:app", "--host", "0.0.0.0", "--port", "7860"]
