mkdir -p $(pwd)/mlflow_data/artifacts
chmod 777 $(pwd)/mlflow_data

docker run -d -p 5000:5000 \
  --name mlflow_container \
  -v $(pwd)/mlflow_data:/mlflow \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server \
  --host 0.0.0.0 \
  --allowed-hosts "*" \
  --backend-store-uri sqlite:////mlflow/mlflow.db \
  --default-artifact-root /mlflow/artifacts