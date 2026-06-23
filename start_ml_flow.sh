#!/bin/bash

# Create local data directory
mkdir -p $(pwd)/mlflow_data
chmod 777 $(pwd)/mlflow_data

# Create the shared network (ignores error if it already exists)
docker network create mlflow-network || true

# Start the MLflow container with the proxy configuration
docker run -d -p 5000:5000 \
  --name mlflow_container \
  --network mlflow-network \
  --network-alias mlflow-server \
  -v /home/daniel.walke/git/SepsisEvalPipeline/mlflow_data:/mlflow \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server \
  --host 0.0.0.0 \
  --allowed-hosts "*" \
  --backend-store-uri sqlite:////mlflow/mlflow.db \
  --serve-artifacts \
  --default-artifact-root mlflow-artifacts:/