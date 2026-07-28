"""
GraphFlow / SepsisEvalPipeline Model Context Protocol (MCP) Server.
Provides standardized RPC tools to:
  1. Execute pipeline steps (2_baseline, 3_graph_construction, 4_db_upload, 5_gnn_training, 6_graphaware, all_steps).
  2. Query MLflow experiment runs, metrics (AUROC, Sensitivity, Specificity), and logged parameters.
  3. Fetch pre-calculated optimal G-Mean ROC cutoffs per panel.
  4. Run GraphFlow 1-hop spatial neighborhood inference programmatically.
  5. Compute 2N aggregated local SHAP explanations for specific patient cases.
  6. Check and interface with the Streamlit GraphFlow inference dashboard.
"""

import os
import json
import sqlite3
import subprocess
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP("GraphFlow-SepsisEvalPipeline-MCP")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_DB_PATH = os.path.join(BASE_DIR, "mlflow_data", "mlflow.db")
CUTOFFS_PATH = os.path.join(BASE_DIR, "7_inference", "optimal_cutoffs.json")

PIPELINE_STEPS = {
    "2_baseline": {"script": "2_baseline/main.py", "desc": "Train baseline ML models (Logistic Regression, Random Forest, XGBoost)"},
    "3_graph_construction": {"script": "3_graph_construction/main.py", "desc": "Construct temporal patient graphs from preprocessed lab data"},
    "4_db_upload": {"script": "4_db_upload/sqlite/upload_to_sqlite.py", "desc": "Upload graph nodes, edges, and features to SQLite database"},
    "5_gnn_training": {"script": "5_gnn_training/main.py", "desc": "Train PyTorch Geometric GNN models"},
    "6_graphaware": {"script": "6_graphaware/main.py", "desc": "Train GraphFlow 1-hop spatial neighborhood feature XGBoost models & log metrics to MLflow"},
}


@mcp.tool()
def list_pipeline_steps() -> Dict[str, Any]:
    """Lists all available pipeline steps (Steps 2 to 7), their descriptions, and execution scripts."""
    return {
        "pipeline_steps": PIPELINE_STEPS,
        "step_1_note": "Step 1 (Data Preprocessing) is performed independently per user dataset under 1_preprocess/",
        "inference_dashboard_step": "Step 7 (GraphFlow Streamlit Dashboard) runs via 7_inference/app.py or Docker container on port 8501."
    }


@mcp.tool()
def run_pipeline_step(step_name: str, selected_panel: Optional[str] = None) -> Dict[str, Any]:
    """
    Executes a specific pipeline step or all steps starting from step 2.

    Args:
        step_name: One of '2_baseline', '3_graph_construction', '4_db_upload', '5_gnn_training', '6_graphaware', or 'all_steps'.
        selected_panel: Optional panel filter (e.g., 'MIMIC_CBC', 'MIMIC_CBC_BMP', 'SBC_CBC').
    """
    if step_name == "all_steps":
        steps_to_run = list(PIPELINE_STEPS.keys())
    elif step_name in PIPELINE_STEPS:
        steps_to_run = [step_name]
    else:
        return {"status": "error", "message": f"Invalid step_name '{step_name}'. Valid options: {list(PIPELINE_STEPS.keys()) + ['all_steps']}"}

    results = []
    python_bin = os.path.join(BASE_DIR, ".venv", "bin", "python")
    if not os.path.exists(python_bin):
        python_bin = "python3"

    for step in steps_to_run:
        step_info = PIPELINE_STEPS[step]
        script_path = os.path.join(BASE_DIR, step_info["script"])
        if not os.path.exists(script_path):
            results.append({"step": step, "status": "error", "message": f"Script not found: {script_path}"})
            continue

        cmd = [python_bin, script_path]
        if selected_panel:
            cmd.extend(["--panel", selected_panel])

        try:
            res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, timeout=1200)
            if res.returncode == 0:
                results.append({
                    "step": step,
                    "status": "success",
                    "output_snippet": res.stdout[-500:] if res.stdout else "Success"
                })
            else:
                results.append({
                    "step": step,
                    "status": "failed",
                    "returncode": res.returncode,
                    "error_snippet": res.stderr[-500:] if res.stderr else res.stdout[-500:]
                })
                break
        except subprocess.TimeoutExpired:
            results.append({"step": step, "status": "timeout", "message": "Execution timed out after 1200s"})
            break
        except Exception as e:
            results.append({"step": step, "status": "error", "message": str(e)})
            break

    return {"status": "completed", "executed_steps": results}


@mcp.tool()
def get_mlflow_experiment_results(experiment_name: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    """
    Queries MLflow backend SQLite database and returns experiment runs with metrics and hyperparameters.

    Args:
        experiment_name: Optional filter by experiment name (e.g., 'evaluations_CBC_BMP' or 'Default').
        limit: Max number of recent runs to return (default 20).
    """
    if not os.path.exists(MLFLOW_DB_PATH):
        return {"status": "error", "message": f"MLflow database not found at {MLFLOW_DB_PATH}"}

    try:
        conn = sqlite3.connect(MLFLOW_DB_PATH)
        cursor = conn.cursor()

        query = """
        SELECT r.run_uuid, e.name as experiment_name, r.status, r.start_time, r.end_time
        FROM runs r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        """
        params = []
        if experiment_name:
            query += " WHERE e.name LIKE ?"
            params.append(f"%{experiment_name}%")
        query += " ORDER BY r.start_time DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        runs_raw = cursor.fetchall()

        runs = []
        for r_uuid, exp_n, status, start_t, end_t in runs_raw:
            # Fetch latest metrics
            cursor.execute("SELECT key, value FROM latest_metrics WHERE run_uuid = ?", (r_uuid,))
            metrics = {row[0]: round(row[1], 4) for row in cursor.fetchall()}

            # Fetch params
            cursor.execute("SELECT key, value FROM params WHERE run_uuid = ?", (r_uuid,))
            run_params = {row[0]: row[1] for row in cursor.fetchall()}

            # Fetch tags
            cursor.execute("SELECT key, value FROM tags WHERE run_uuid = ?", (r_uuid,))
            tags = {row[0]: row[1] for row in cursor.fetchall()}

            runs.append({
                "run_id": r_uuid,
                "experiment_name": exp_n,
                "model": run_params.get("model", tags.get("model", "Unknown")),
                "status": status,
                "metrics": metrics,
                "parameters": run_params,
                "tags": tags
            })

        conn.close()
        return {"status": "success", "total_runs": len(runs), "runs": runs}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def get_optimal_cutoffs() -> Dict[str, Any]:
    """
    Returns pre-calculated Geometric Mean (G-Mean) ROC classification cutoffs for all laboratory panels.
    """
    if not os.path.exists(CUTOFFS_PATH):
        return {"status": "error", "message": f"Optimal cutoffs JSON file not found at {CUTOFFS_PATH}"}

    with open(CUTOFFS_PATH, "r") as f:
        cutoffs = json.load(f)
    return {"status": "success", "cutoffs": cutoffs}


@mcp.tool()
def run_graphflow_inference(
    selected_panel: str = "MIMIC_CBC",
    dataset_key: Optional[str] = None,
    max_rows: int = 500,
    risk_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Runs GraphFlow 1-hop spatial neighborhood feature aggregation and XGBoost inference programmatically.

    Args:
        selected_panel: Panel name (e.g. 'MIMIC_CBC', 'MIMIC_CBC_BMP', 'SBC_CBC').
        dataset_key: Optional specific sample test dataset filename or relative path.
        max_rows: Number of dataset rows to evaluate (default 500).
        risk_threshold: Optional Sepsis risk cutoff override (defaults to Geometric Mean ROC cutoff).
    """
def _load_inference_app():
    import importlib.util
    app_path = os.path.join(BASE_DIR, "7_inference", "app.py")
    spec = importlib.util.spec_from_file_location("inference_app", app_path)
    app_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_mod)
    return app_mod


@mcp.tool()
def run_graphflow_inference(
    selected_panel: str = "MIMIC_CBC",
    dataset_key: Optional[str] = None,
    max_rows: int = 500,
    risk_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Runs GraphFlow 1-hop spatial neighborhood feature aggregation and XGBoost inference programmatically.

    Args:
        selected_panel: Panel name (e.g. 'MIMIC_CBC', 'MIMIC_CBC_BMP', 'SBC_CBC').
        dataset_key: Optional specific sample test dataset filename or relative path.
        max_rows: Number of dataset rows to evaluate (default 500).
        risk_threshold: Optional Sepsis risk cutoff override (defaults to Geometric Mean ROC cutoff).
    """
    app_mod = _load_inference_app()
    get_sample_datasets = app_mod.get_sample_datasets
    run_graphaware_inference = app_mod.run_graphaware_inference
    get_default_cutoff = app_mod.get_default_cutoff

    sample_datasets = get_sample_datasets(selected_panel)
    if not sample_datasets:
        return {"status": "error", "message": f"No sample test datasets found for panel '{selected_panel}'"}

    target_path = None
    if dataset_key and dataset_key in sample_datasets:
        target_path = sample_datasets[dataset_key]
    else:
        target_key = list(sample_datasets.keys())[0]
        target_path = sample_datasets[target_key]
        dataset_key = target_key

    if not os.path.exists(target_path):
        return {"status": "error", "message": f"Dataset path not found: {target_path}"}

    df = pd.read_csv(target_path)
    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    clean_panel = selected_panel.replace("MIMIC_", "").replace("SBC_", "")
    model_path = os.path.join(BASE_DIR, "6_graphaware", "models", clean_panel, "final_model.xgb")
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "6_graphaware", "models", selected_panel, "final_model.xgb")

    res_df, preds_prob, final_feats, f_cols = run_graphaware_inference(df, model_path, selected_panel)
    if res_df is None or preds_prob is None:
        return {"status": "error", "message": "GraphFlow inference failed to produce prediction probabilities."}

    opt_cutoff = get_default_cutoff(selected_panel, dataset_key)
    active_cutoff = risk_threshold if risk_threshold is not None else opt_cutoff

    c_safe = max(1e-6, min(1.0 - 1e-6, float(active_cutoff)))
    calibrated_risk_pct = np.where(
        preds_prob < c_safe,
        50.0 * (preds_prob / c_safe),
        50.0 + 50.0 * ((preds_prob - c_safe) / (1.0 - c_safe))
    )

    res_df["Sepsis_Prediction_Probability"] = preds_prob.round(6)
    res_df["Sepsis_Risk_%"] = np.clip(calibrated_risk_pct, 0.0, 100.0).round(2)
    res_df["Predicted_Class"] = np.where(preds_prob >= active_cutoff, "Sepsis", "Control")

    sepsis_cnt = int((preds_prob >= active_cutoff).sum())
    control_cnt = int((preds_prob < active_cutoff).sum())

    return {
        "status": "success",
        "panel_name": selected_panel,
        "dataset_key": dataset_key,
        "total_rows_evaluated": len(preds_prob),
        "gmean_roc_cutoff": opt_cutoff,
        "active_risk_threshold": active_cutoff,
        "predicted_sepsis_cases": sepsis_cnt,
        "predicted_control_cases": control_cnt,
        "mean_sepsis_probability": float(np.mean(preds_prob)),
        "top_5_high_risk_predictions": res_df.head(5)[["Id", "Time", "Sepsis_Prediction_Probability", "Sepsis_Risk_%", "Predicted_Class"]].to_dict(orient="records")
    }


@mcp.tool()
def explain_patient_prediction(
    selected_panel: str = "MIMIC_CBC",
    row_index: int = 0,
    dataset_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculates 2N total aggregated local SHAP values (SHAP_orig + SHAP_delta_mean) for a specific patient row.

    Args:
        selected_panel: Panel name (e.g. 'MIMIC_CBC', 'MIMIC_CBC_BMP').
        row_index: 0-indexed row position of the patient observation to explain.
        dataset_key: Optional sample test dataset key.
    """
    app_mod = _load_inference_app()
    get_sample_datasets = app_mod.get_sample_datasets
    run_graphaware_inference = app_mod.run_graphaware_inference
    get_shap_explanations = app_mod.get_shap_explanations
    get_default_cutoff = app_mod.get_default_cutoff

    sample_datasets = get_sample_datasets(selected_panel)
    if not sample_datasets:
        return {"status": "error", "message": f"No sample datasets found for panel '{selected_panel}'"}

    target_key = dataset_key if (dataset_key and dataset_key in sample_datasets) else list(sample_datasets.keys())[0]
    target_path = sample_datasets[target_key]
    df = pd.read_csv(target_path)

    if row_index < 0 or row_index >= len(df):
        return {"status": "error", "message": f"Invalid row_index {row_index}. Valid range: [0, {len(df)-1}]"}

    clean_panel = selected_panel.replace("MIMIC_", "").replace("SBC_", "")
    model_path = os.path.join(BASE_DIR, "6_graphaware", "models", clean_panel, "final_model.xgb")
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "6_graphaware", "models", selected_panel, "final_model.xgb")

    res_df, preds_prob, final_feats, f_cols = run_graphaware_inference(df, model_path, selected_panel)
    shap_vals, base_val = get_shap_explanations(model_path, final_feats)

    opt_cutoff = get_default_cutoff(selected_panel, target_key)
    c_safe = max(1e-6, min(1.0 - 1e-6, float(opt_cutoff)))
    calibrated_risk_pct = np.where(
        preds_prob < c_safe,
        50.0 * (preds_prob / c_safe),
        50.0 + 50.0 * ((preds_prob - c_safe) / (1.0 - c_safe))
    )

    res_df["Sepsis_Prediction_Probability"] = preds_prob.round(6)
    res_df["Sepsis_Risk_%"] = np.clip(calibrated_risk_pct, 0.0, 100.0).round(2)
    res_df["Predicted_Class"] = np.where(preds_prob >= opt_cutoff, "Sepsis", "Control")

    row_res = res_df.iloc[row_index]
    row_final_feats = final_feats[row_index]
    row_shap = shap_vals[row_index]

    num_base_f = len(f_cols)
    orig_vals = row_final_feats[:num_base_f]
    diff_vals = row_final_feats[num_base_f:]
    shap_orig = row_shap[:num_base_f]
    shap_diff = row_shap[num_base_f:]
    shap_total = shap_orig + shap_diff
    clean_names = [c.replace("f__", "") for c in f_cols]

    breakdown = []
    for i in range(num_base_f):
        breakdown.append({
            "feature": clean_names[i],
            "original_value": float(orig_vals[i]),
            "original_shap": float(shap_orig[i]),
            "time_based_delta_mean": float(diff_vals[i]),
            "time_based_shap": float(shap_diff[i]),
            "total_aggregated_shap": float(shap_total[i]),
            "effect": "Increases Sepsis Risk" if shap_total[i] > 0 else "Decreases Sepsis Risk"
        })

    breakdown_sorted = sorted(breakdown, key=lambda x: abs(x["total_aggregated_shap"]), reverse=True)

    return {
        "status": "success",
        "patient_id": str(row_res.get("Id", "N/A")),
        "timestamp": str(row_res.get("Time", "N/A")),
        "predicted_class": str(row_res.get("Predicted_Class", "N/A")),
        "sepsis_prediction_probability": float(row_res.get("Sepsis_Prediction_Probability", 0.0)),
        "calibrated_sepsis_risk_percent": float(row_res.get("Sepsis_Risk_%", 0.0)),
        "top_feature_attributions": breakdown_sorted[:10]
    }


@mcp.tool()
def get_dashboard_status() -> Dict[str, Any]:
    """
    Checks if the GraphFlow Streamlit inference web dashboard is running on port 8501.
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 8501))
    sock.close()
    is_running = (result == 0)

    return {
        "dashboard_name": "GraphFlow Inference Dashboard",
        "port": 8501,
        "is_running": is_running,
        "url": "http://localhost:8501" if is_running else "Not running",
        "launch_command": "streamlit run 7_inference/app.py --server.port=8501"
    }


if __name__ == "__main__":
    mcp.run()
