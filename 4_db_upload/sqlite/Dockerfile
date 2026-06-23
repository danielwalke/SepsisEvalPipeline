# Use a lightweight Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure Python output is logged straight to the terminal (useful for Docker logs)
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy your Python scripts (main.py, SQLiteQueries.py, etc.) into the container
COPY . .

# Run the upload script
CMD ["python", "main.py"]