docker run -d -p 5000:5000 \
  --name mlflow_container \
  -v $(pwd)/mlflow_data:/mlflow \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server \
  --host 0.0.0.0 \
  --allowed-hosts "host.docker.internal:5000,host.docker.internal,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000" \
  --backend-store-uri sqlite:////mlflow/mlflow.db \
  --default-artifact-root /mlflow/artifacts