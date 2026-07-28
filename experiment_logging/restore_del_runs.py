import mlflow
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://localhost:5000")

experiment = mlflow.get_experiment_by_name("evaluations_CBC")
experiment_id = experiment.experiment_id

deleted_runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    run_view_type=ViewType.DELETED_ONLY
)

print(deleted_runs[["run_id", "tags.mlflow.runName", "end_time"]])