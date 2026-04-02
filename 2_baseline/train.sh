docker run -d -p 5000:5000 \
  --name mlflow_container \
  -v "${PWD}/mlflow_data:/mlflow" \
  ghcr.io/mlflow/mlflow:latest \
  mlflow server --host 0.0.0.0 \
  --backend-store-uri /mlflow \
  --default-artifact-root /mlflow/artifacts \
  --allowed-hosts "host.docker.internal:5000,host.docker.internal,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000"