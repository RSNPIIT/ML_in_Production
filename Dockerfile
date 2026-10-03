# Base Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and models
COPY app ./app
COPY models ./models
COPY train.py .

# Expose port (Render sets $PORT dynamically, default 10000)
EXPOSE 10000

# Run Uvicorn server bound to 0.0.0.0 and $PORT
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
