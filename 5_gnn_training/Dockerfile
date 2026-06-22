# Use an official PyTorch runtime with CUDA support as a parent image
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Ensure the checkpoint directory exists (prevents path errors if not mounted)
RUN mkdir -p /app/checkpoints

# Run main.py when the container launches
CMD ["python", "main.py"]