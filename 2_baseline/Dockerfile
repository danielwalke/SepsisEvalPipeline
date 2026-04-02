FROM python:3.14-slim

WORKDIR /app

# 1. Context is parent, so we reach INTO the subfolder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy the folder contents, then the external file
COPY . .

# 3. Standard setup
RUN mkdir -p /app/input /app/output

CMD ["python", "main.py"]