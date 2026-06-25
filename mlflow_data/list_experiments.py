from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

client = MlflowClient(tracking_uri="http://localhost:5000")

# 1. See everything, including deleted, regardless of name collisions
for exp in client.search_experiments(view_type=ViewType.ALL):
    print(exp.experiment_id, exp.name, exp.lifecycle_stage)