FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY 7_inference/requirements.txt /app/7_inference/requirements.txt
RUN pip install --no-cache-dir -r /app/7_inference/requirements.txt
RUN pip install --no-cache-dir mcp>=1.0.0

COPY . /app

CMD ["python", "mcp_server/server.py"]
