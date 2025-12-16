# Base image with a specific R version for reproducibility
FROM rocker/r-ver:4.4.0

# Install system dependencies (data.table requires zlib)
RUN apt-get update && apt-get install -y \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install data.table
# install2.r is a helper script included in rocker images
RUN install2.r --error \
    data.table

# Create directories for the script, input, and output
WORKDIR /app
RUN mkdir -p /app/input /app/output /app/extdata /app/features

# Copy your R script into the container
# Replace 'process_files.R' with your actual script name
COPY process_files.R /app/process_files.R
COPY extdata /app/extdata

# Set the command to run your script when the container starts
CMD ["Rscript", "/app/process_files.R"]