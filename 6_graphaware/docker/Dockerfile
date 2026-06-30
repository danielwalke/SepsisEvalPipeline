# Use the official Python 3.11 slim image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system build dependencies required for certain ML packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ensure pip is up to date first
RUN pip install --no-cache-dir --upgrade pip

# Pre-install PyTorch with CUDA 12.1 support
# This ensures that torch-geometric and others link against the correct PyTorch backend
# Pre-install PyTorch with CUDA 13.0 support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Copy the requirements and install the rest of your stack
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies all code into /app
COPY . .

# Run your ML pipeline
CMD ["python", "main.py"]