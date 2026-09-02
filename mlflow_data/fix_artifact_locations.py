import os
import sqlite3

# Experiments created while mlflow-server ran with --default-artifact-root got a bare
# filesystem path as their artifact_location, which makes clients try direct filesystem
# access to /mlflow instead of going through the --serve-artifacts proxy. Rewriting it to
# the mlflow-artifacts:/<id> URI fixes existing experiments; idempotent and a no-op on a
# fresh mlflow.db (or one already migrated).
DB_PATH = "/mlflow/mlflow.db"

if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE experiments SET artifact_location = 'mlflow-artifacts:/' || experiment_id "
        "WHERE artifact_location LIKE '/mlflow/artifacts/%'"
    )
    conn.commit()
    conn.close()
