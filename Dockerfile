# Use Python 3.10 as base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY app/ ./app/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Set environment variables for Gunicorn
ENV GUNICORN_CMD_ARGS="--workers=1 --threads=2 --timeout=300 --bind=0.0.0.0:8000"

# Run the application with Gunicorn
CMD ["gunicorn", "app.app:app"] 