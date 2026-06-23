# Use the official PyTorch image with CUDA and cuDNN pre-installed
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# explicitly pull the requirements from the docker/ folder
COPY docker/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copies all code from 5_gnn_training/ into /app
COPY . .

# Run your pipeline
CMD ["python", "main.py"]