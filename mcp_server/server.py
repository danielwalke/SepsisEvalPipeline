"""
GraphFlow / SepsisEvalPipeline Model Context Protocol (MCP) Server.
Provides standardized RPC tools to:
  1. Execute pipeline steps (2_baseline, 3_graph_construction, 4_db_upload, 5_gnn_training, 6_graphaware, all_steps).
  2. Query MLflow experiment runs, metrics (AUROC, Sensitivity, Specificity), and logged parameters.
  3. Fetch pre-calculated optimal F2 score (beta=2) cutoffs per panel.
  4. Run GraphFlow 1-hop spatial neighborhood inference programmatically.
  5. Compute 2N aggregated local SHAP explanations for specific patient cases.
  6. Check and interface with the Streamlit GraphFlow inference dashboard.
  7. Programmatically start or restart the Streamlit GraphFlow inference dashboard.
"""

import os
import sys
import json
import socket
import sqlite3
import subprocess
import contextlib
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score, fbeta_score, accuracy_score
)
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
    Returns pre-calculated optimal F2 score (beta=2) classification cutoffs for all laboratory panels.
    """
    if not os.path.exists(CUTOFFS_PATH):
        return {"status": "error", "message": f"Optimal cutoffs JSON file not found at {CUTOFFS_PATH}"}

    with open(CUTOFFS_PATH, "r") as f:
        cutoffs = json.load(f)
    return {"status": "success", "cutoffs": cutoffs}


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
    max_rows: Optional[int] = None,
    risk_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Runs GraphFlow 1-hop spatial neighborhood feature aggregation and XGBoost inference programmatically.

    Args:
        selected_panel: Panel name (e.g. 'MIMIC_CBC', 'MIMIC_CBC_BMP', 'SBC_CBC').
        dataset_key: Optional specific sample test dataset filename or relative path.
        max_rows: Optional max number of dataset rows to evaluate (defaults to None = full dataset).
        risk_threshold: Optional Sepsis risk cutoff override (defaults to optimal F2 score cutoff).
    """
    with contextlib.redirect_stdout(sys.stderr):
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
        if max_rows is not None and len(df) > max_rows:
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

    # Ground-truth evaluation metrics (matching 7_inference/app.py implementation)
    gt_metrics = {}
    gt_col = None
    for candidate in ["y", "label", "Label", "target", "SepsisLabel"]:
        if candidate in res_df.columns:
            gt_col = candidate
            break

    if gt_col is not None:
        y_true_binary = (res_df[gt_col].astype(str).str.contains("Sepsis|1")).astype(int)
        y_pred_binary = (preds_prob >= active_cutoff).astype(int)

        if len(np.unique(y_true_binary)) > 1:
            auroc_score = roc_auc_score(y_true_binary, preds_prob)
            sens = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            f2 = fbeta_score(y_true_binary, y_pred_binary, beta=2, zero_division=0)
            acc = accuracy_score(y_true_binary, y_pred_binary)
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            g_mean = float(np.sqrt(sens * spec))

            gt_metrics = {
                "auroc": round(float(auroc_score), 4),
                "sensitivity": round(float(sens), 4),
                "specificity": round(float(spec), 4),
                "precision_ppv": round(float(prec), 4),
                "npv": round(float(npv), 4),
                "f1_score": round(float(f1), 4),
                "f2_score": round(float(f2), 4),
                "accuracy": round(float(acc), 4),
                "g_mean": round(g_mean, 4),
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
            }
        else:
            gt_metrics = {"warning": "Only one class present in ground truth — cannot compute AUROC or binary metrics."}

    result = {
        "status": "success",
        "panel_name": selected_panel,
        "dataset_key": dataset_key,
        "total_rows_evaluated": len(preds_prob),
        "optimal_f2_cutoff": opt_cutoff,
        "active_risk_threshold": active_cutoff,
        "predicted_sepsis_cases": sepsis_cnt,
        "predicted_control_cases": control_cnt,
        "mean_sepsis_probability": float(np.mean(preds_prob)),
        "top_5_high_risk_predictions": res_df.head(5)[["Id", "Time", "Sepsis_Prediction_Probability", "Sepsis_Risk_%", "Predicted_Class"]].to_dict(orient="records")
    }

    if gt_metrics:
        result["ground_truth_metrics"] = gt_metrics

    return result


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
    with contextlib.redirect_stdout(sys.stderr):
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


@mcp.tool()
def start_dashboard(port: int = 8501, headless: bool = True) -> Dict[str, Any]:
    """
    Launches the Streamlit GraphFlow inference dashboard in the background if it is not already running.

    Args:
        port: Port to run the Streamlit server on (default 8501).
        headless: Run Streamlit in headless mode without attempting to open a browser window (default True).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    check_res = sock.connect_ex(('127.0.0.1', port))
    sock.close()

    if check_res == 0:
        return {
            "status": "already_running",
            "message": f"Dashboard is already active on port {port}",
            "url": f"http://localhost:{port}"
        }

    streamlit_bin = os.path.join(BASE_DIR, ".venv", "bin", "streamlit")
    if not os.path.exists(streamlit_bin):
        streamlit_bin = "streamlit"

    app_script = os.path.join(BASE_DIR, "7_inference", "app.py")
    if not os.path.exists(app_script):
        return {"status": "error", "message": f"Streamlit application script not found at {app_script}"}

    cmd = [
        streamlit_bin,
        "run",
        app_script,
        f"--server.port={port}",
        f"--server.headless={str(headless).lower()}"
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=BASE_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        return {
            "status": "success",
            "message": f"Dashboard process launched on port {port}",
            "pid": proc.pid,
            "url": f"http://localhost:{port}"
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to launch Streamlit dashboard: {str(e)}"}


def _resolve_panel_assets(selected_panel: str = "MIMIC_CBC_BMP"):
    """
    Dynamically resolves asset paths (test dataset CSV, baseline model, 
    GraphAware model, SQLite graph database, and optimal decision cutoffs) 
    for any specified laboratory panel name.

    Supports inputs such as:
      - 'MIMIC_CBC_BMP', 'CBC_BMP'
      - 'MIMIC_CBC', 'CBC'
      - 'MIMIC_BMP', 'BMP'
    """
    clean_panel = selected_panel.replace("MIMIC_", "").replace("SBC_", "")
    panel_full_name = f"MIMIC_{clean_panel}"

    # 1. Dataset CSV path
    csv_path = os.path.join(BASE_DIR, "1_preprocess", "data", "preprocessed_data", clean_panel, "mimic_processed_test.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, "1_preprocess", "data", "preprocessed_data", panel_full_name, "mimic_processed_test.csv")

    # 2. Baseline Model path
    baseline_path = os.path.join(BASE_DIR, "2_baseline", "models", panel_full_name, "XGBClassifier.pkl")
    if not os.path.exists(baseline_path):
        baseline_path = os.path.join(BASE_DIR, "2_baseline", "models", clean_panel, "XGBClassifier.pkl")

    # 3. GraphAware Model path
    ga_model_path = os.path.join(BASE_DIR, "6_graphaware", "models", panel_full_name, "final_model.xgb")
    if not os.path.exists(ga_model_path):
        ga_model_path = os.path.join(BASE_DIR, "6_graphaware", "models", clean_panel, "final_model.xgb")

    # 4. SQLite Graph DB path
    db_path = os.path.join(BASE_DIR, "4_db_upload", "sqlite", "sqlite_data", clean_panel, "mimic_sbc_graph.db")
    if not os.path.exists(db_path):
        db_path = os.path.join(BASE_DIR, "4_db_upload", "sqlite", "sqlite_data", panel_full_name, "mimic_sbc_graph.db")

    # 5. Optimal Cutoffs per panel (Read F2 score cutoffs from panel_optimal_cutoffs.json / 7_inference/optimal_cutoffs.json)
    panel_cutoffs_path = os.path.join(BASE_DIR, "mcp_server", "panel_optimal_cutoffs.json")
    opt_json_path = os.path.join(BASE_DIR, "7_inference", "optimal_cutoffs.json")
    cutoff_base = 0.05715
    cutoff_ga = 0.032564

    # Read panel_optimal_cutoffs.json for baseline and graphaware cutoffs
    if os.path.exists(panel_cutoffs_path):
        with open(panel_cutoffs_path, 'r') as f:
            panel_cut_data = json.load(f)
            panel_entry = panel_cut_data.get(clean_panel, panel_cut_data.get(panel_full_name, {}))
            if panel_entry:
                cutoff_base = float(panel_entry.get("baseline_cutoff", cutoff_base))
                cutoff_ga = float(panel_entry.get("graphaware_cutoff", cutoff_ga))

    # Override graphaware cutoff from 7_inference/optimal_cutoffs.json if available
    if os.path.exists(opt_json_path):
        with open(opt_json_path, 'r') as f:
            cut_data = json.load(f)
            if panel_full_name in cut_data:
                cutoff_ga = float(cut_data[panel_full_name].get("DEFAULT", cutoff_ga))
            elif clean_panel in cut_data:
                cutoff_ga = float(cut_data[clean_panel].get("DEFAULT", cutoff_ga))

    return {
        "panel_name": panel_full_name,
        "clean_panel": clean_panel,
        "csv_path": csv_path,
        "baseline_path": baseline_path,
        "ga_model_path": ga_model_path,
        "db_path": db_path if os.path.exists(db_path) else None,
        "cutoff_baseline": cutoff_base,
        "cutoff_graphaware": cutoff_ga
    }


@mcp.tool()
def find_divergent_patient_trajectories(
    selected_panel: str = "MIMIC_CBC",
    min_events: int = 2,
    max_candidates: int = 5
) -> Dict[str, Any]:
    """
    Scans patient time series trajectories in the specified panel dataset to find cases where 
    Traditional XGBoost and GraphAware XGBoost make divergent predictions.

    Args:
        selected_panel: Laboratory panel name (e.g. 'MIMIC_CBC_BMP', 'MIMIC_CBC', 'MIMIC_BMP').
        min_events: Minimum number of time series observations per patient (default 2).
        max_candidates: Maximum number of divergent patient trajectory cases to return (default 5).
    """
    with contextlib.redirect_stdout(sys.stderr):
        app_mod = _load_inference_app()
        run_graphaware_inference = app_mod.run_graphaware_inference

        assets = _resolve_panel_assets(selected_panel)
        if not os.path.exists(assets["csv_path"]):
            return {"status": "error", "message": f"Dataset CSV not found at {assets['csv_path']}"}

        df_all = pd.read_csv(assets["csv_path"])

        import joblib
        import xgboost as xgb

        if not os.path.exists(assets["ga_model_path"]) or not os.path.exists(assets["baseline_path"]):
            return {"status": "error", "message": f"Model files not found for panel '{assets['panel_name']}'"}

        baseline_model = joblib.load(assets["baseline_path"])
        feature_cols = [c for c in df_all.columns if c.startswith("f__")]

        df_all['y_bin'] = (df_all['y'] == 'Sepsis').astype(int)
        df_all['pred_baseline'] = baseline_model.predict_proba(df_all[feature_cols].to_numpy())[:, 1]
        sepsis_patient_ids = df_all[df_all['y_bin'] == 1]['Id'].unique()

        cutoff_base = assets["cutoff_baseline"]
        cutoff_ga = assets["cutoff_graphaware"]

        df_all['base_pos'] = df_all['pred_baseline'] >= cutoff_base

        divergent_candidates = []

        for pid in sepsis_patient_ids:
            p_df = df_all[df_all['Id'] == pid].sort_values('Time').copy()
            if len(p_df) < min_events:
                continue

            sorted_df, preds_ga, _, _ = run_graphaware_inference(p_df, assets["ga_model_path"], selected_panel)
            sorted_df['pred_graphaware'] = preds_ga
            sorted_df['ga_pos'] = sorted_df['pred_graphaware'] >= cutoff_ga
            sorted_df['base_pos'] = sorted_df['pred_baseline'] >= cutoff_base

            sepsis_events = sorted_df[sorted_df['y_bin'] == 1]
            control_events = sorted_df[sorted_df['y_bin'] == 0]
            if len(sepsis_events) == 0:
                continue

            n_s = len(sepsis_events)
            n_c = len(control_events)
            b_c_fps = control_events['base_pos'].sum() if n_c > 0 else 0
            g_c_fps = control_events['ga_pos'].sum() if n_c > 0 else 0

            b_s_hits = sepsis_events['base_pos'].sum()
            g_s_hits = sepsis_events['ga_pos'].sum()

            ga_c_acc = ((n_c - g_c_fps) / n_c) if n_c > 0 else 1.0
            base_c_acc = ((n_c - b_c_fps) / n_c) if n_c > 0 else 1.0

            ga_perfect_sepsis = (g_s_hits == n_s)
            base_missed_all_sepsis = (b_s_hits == 0)
            baseline_100pct_negative = (sorted_df['base_pos'].sum() == 0)

            # Require GraphAware to perfectly predict sepsis onset (100% sepsis recall) while Baseline XGBoost fails to predict any sepsis events
            if ga_perfect_sepsis and base_missed_all_sepsis:
                events_summary = []
                first_charttime = pd.to_datetime(sorted_df['charttime'].iloc[0])
                for _, r in sorted_df.iterrows():
                    ct = pd.to_datetime(r['charttime'])
                    hr = (ct - first_charttime).total_seconds() / 3600.0
                    events_summary.append({
                        "hours_elapsed": round(hr, 1),
                        "charttime": str(r['charttime']),
                        "true_label": str(r['y']),
                        "wbc": float(r.get('f__WBC', 0.0)),
                        "platelets": float(r.get('f__PLT', 0.0)),
                        "baseline_prob": round(float(r['pred_baseline']), 6),
                        "baseline_status": "POSITIVE" if r['base_pos'] else "NEGATIVE",
                        "graphaware_prob": round(float(r['pred_graphaware']), 6),
                        "graphaware_status": "POSITIVE" if r['ga_pos'] else "NEGATIVE"
                    })

                divergent_candidates.append({
                    "patient_id": str(pid),
                    "subject_id": str(sorted_df['subject_id'].iloc[0]),
                    "hadm_id": str(sorted_df['hadm_id'].iloc[0]),
                    "total_events": len(sorted_df),
                    "baseline_entire_series_100pct_negative": bool(baseline_100pct_negative),
                    "graphaware_control_accuracy": f"{ga_c_acc*100:.1f}% ({n_c - g_c_fps}/{n_c} correct)",
                    "baseline_control_accuracy": f"{base_c_acc*100:.1f}% ({n_c - b_c_fps}/{n_c} correct)",
                    "divergence_type": "GraphAware Perfect Sepsis Onset (Baseline Entirely Negative)" if baseline_100pct_negative else "GraphAware Perfect Sepsis Onset (Baseline Missed All Sepsis)",
                    "baseline_strictly_neg_raw": baseline_100pct_negative,
                    "ga_control_acc_raw": ga_c_acc,
                    "events": events_summary
                })

        # Sort candidates: Prioritize Baseline 100% negative across entire series first, then highest GraphAware Control accuracy
        divergent_candidates.sort(key=lambda x: (x['baseline_strictly_neg_raw'], x['ga_control_acc_raw']), reverse=True)

        for case in divergent_candidates:
            case.pop('baseline_strictly_neg_raw', None)
            case.pop('ga_control_acc_raw', None)

        final_cases = divergent_candidates[:max_candidates]

        return {
            "status": "success",
            "panel_used": assets["panel_name"],
            "selected_panel": selected_panel,
            "baseline_cutoff": cutoff_base,
            "graphaware_cutoff": cutoff_ga,
            "total_divergent_cases_found": len(divergent_candidates),
            "cases": final_cases
        }


@mcp.tool()
def evaluate_patient_traditional_xgboost(
    patient_id: int,
    selected_panel: str = "MIMIC_CBC_BMP"
) -> Dict[str, Any]:
    """
    Evaluates the time series trajectory of a specific patient ID using ONLY the Traditional Baseline XGBoost model 
    for the specified laboratory panel name.

    Args:
        patient_id: Internal Patient ID (e.g. 542802, 115103, 166147).
        selected_panel: Laboratory panel name (e.g. 'MIMIC_CBC_BMP', 'MIMIC_CBC', 'MIMIC_BMP').
    """
    with contextlib.redirect_stdout(sys.stderr):
        assets = _resolve_panel_assets(selected_panel)
        if not os.path.exists(assets["csv_path"]):
            return {"status": "error", "message": f"Dataset CSV not found at {assets['csv_path']}"}

        df_all = pd.read_csv(assets["csv_path"])

        p_df = df_all[df_all['Id'] == patient_id].sort_values('Time').copy()
        if len(p_df) == 0:
            return {"status": "error", "message": f"Patient ID {patient_id} not found in panel '{assets['panel_name']}'."}

        import joblib
        if not os.path.exists(assets["baseline_path"]):
            return {"status": "error", "message": f"Baseline model file not found at {assets['baseline_path']}"}

        baseline_model = joblib.load(assets["baseline_path"])
        feature_cols = [c for c in p_df.columns if c.startswith("f__")]
        preds_base = baseline_model.predict_proba(p_df[feature_cols].to_numpy())[:, 1]
        p_df['pred_baseline'] = preds_base

        cutoff = assets["cutoff_baseline"]
        p_df['base_pos'] = p_df['pred_baseline'] >= cutoff

        events = []
        first_ct = pd.to_datetime(p_df['charttime'].iloc[0])
        for _, r in p_df.iterrows():
            ct = pd.to_datetime(r['charttime'])
            hr = (ct - first_ct).total_seconds() / 3600.0
            events.append({
                "hours_elapsed": round(hr, 1),
                "charttime": str(r['charttime']),
                "true_label": str(r['y']),
                "wbc": float(r.get('f__WBC', 0.0)),
                "platelets": float(r.get('f__PLT', 0.0)),
                "baseline_probability": round(float(r['pred_baseline']), 6),
                "predicted_class": "Sepsis" if r['base_pos'] else "Control",
                "status": "POSITIVE" if r['base_pos'] else "NEGATIVE"
            })

        return {
            "status": "success",
            "model_type": "Traditional Baseline XGBoost",
            "panel_used": assets["panel_name"],
            "selected_panel": selected_panel,
            "patient_id": str(patient_id),
            "subject_id": str(p_df['subject_id'].iloc[0]),
            "hadm_id": str(p_df['hadm_id'].iloc[0]),
            "total_events": len(p_df),
            "model_optimal_cutoff": cutoff,
            "events": events
        }


@mcp.tool()
def evaluate_patient_graphaware_xgboost(
    patient_id: int,
    selected_panel: str = "MIMIC_CBC_BMP"
) -> Dict[str, Any]:
    """
    Evaluates the time series trajectory of a specific patient ID using ONLY the GraphAware XGBoost framework 
    for the specified laboratory panel name.

    Args:
        patient_id: Internal Patient ID (e.g. 542802, 115103, 166147).
        selected_panel: Laboratory panel name (e.g. 'MIMIC_CBC_BMP', 'MIMIC_CBC', 'MIMIC_BMP').
    """
    with contextlib.redirect_stdout(sys.stderr):
        app_mod = _load_inference_app()
        run_graphaware_inference = app_mod.run_graphaware_inference

        assets = _resolve_panel_assets(selected_panel)
        if not os.path.exists(assets["csv_path"]):
            return {"status": "error", "message": f"Dataset CSV not found at {assets['csv_path']}"}

        df_all = pd.read_csv(assets["csv_path"])

        p_df = df_all[df_all['Id'] == patient_id].sort_values('Time').copy()
        if len(p_df) == 0:
            return {"status": "error", "message": f"Patient ID {patient_id} not found in panel '{assets['panel_name']}'."}

        if not os.path.exists(assets["ga_model_path"]):
            return {"status": "error", "message": f"GraphAware model file not found at {assets['ga_model_path']}"}

        sorted_df, preds_ga, _, _ = run_graphaware_inference(p_df, assets["ga_model_path"], selected_panel)
        sorted_df['pred_graphaware'] = preds_ga

        cutoff = assets["cutoff_graphaware"]
        sorted_df['ga_pos'] = sorted_df['pred_graphaware'] >= cutoff

        events = []
        first_ct = pd.to_datetime(sorted_df['charttime'].iloc[0])
        for _, r in sorted_df.iterrows():
            ct = pd.to_datetime(r['charttime'])
            hr = (ct - first_ct).total_seconds() / 3600.0
            events.append({
                "hours_elapsed": round(hr, 1),
                "charttime": str(r['charttime']),
                "true_label": str(r['y']),
                "wbc": float(r.get('f__WBC', 0.0)),
                "platelets": float(r.get('f__PLT', 0.0)),
                "graphaware_probability": round(float(r['pred_graphaware']), 6),
                "predicted_class": "Sepsis" if r['ga_pos'] else "Control",
                "status": "POSITIVE" if r['ga_pos'] else "NEGATIVE"
            })

        return {
            "status": "success",
            "model_type": "GraphAware XGBoost Framework",
            "panel_used": assets["panel_name"],
            "selected_panel": selected_panel,
            "patient_id": str(patient_id),
            "subject_id": str(sorted_df['subject_id'].iloc[0]),
            "hadm_id": str(sorted_df['hadm_id'].iloc[0]),
            "total_events": len(sorted_df),
            "model_optimal_cutoff": cutoff,
            "events": events
        }


@mcp.tool()
def compare_patient_time_series(
    patient_id: int,
    selected_panel: str = "MIMIC_CBC_BMP"
) -> Dict[str, Any]:
    """
    Directly compares the time series trajectory of a specific patient ID between 
    Traditional Baseline XGBoost and GraphAware XGBoost side-by-side for the specified laboratory panel name.

    Args:
        patient_id: Internal Patient ID (e.g. 542802, 115103, 166147).
        selected_panel: Laboratory panel name (e.g. 'MIMIC_CBC_BMP', 'MIMIC_CBC', 'MIMIC_BMP').
    """
    base_res = evaluate_patient_traditional_xgboost(patient_id, selected_panel)
    ga_res = evaluate_patient_graphaware_xgboost(patient_id, selected_panel)

    if base_res.get("status") == "error":
        return base_res
    if ga_res.get("status") == "error":
        return ga_res

    cutoff_base = base_res["model_optimal_cutoff"]
    cutoff_ga = ga_res["model_optimal_cutoff"]
    panel_used = base_res.get("panel_used", selected_panel)

    events_comparison = []
    for b_ev, g_ev in zip(base_res["events"], ga_res["events"]):
        events_comparison.append({
            "hours_elapsed": b_ev["hours_elapsed"],
            "charttime": b_ev["charttime"],
            "true_label": b_ev["true_label"],
            "wbc": b_ev["wbc"],
            "platelets": b_ev["platelets"],
            "baseline_prob": b_ev["baseline_probability"],
            "baseline_status": b_ev["status"],
            "graphaware_prob": g_ev["graphaware_probability"],
            "graphaware_status": g_ev["status"],
            "prediction_match": b_ev["status"] == g_ev["status"]
        })

    return {
        "status": "success",
        "panel_used": panel_used,
        "selected_panel": selected_panel,
        "patient_id": str(patient_id),
        "subject_id": base_res["subject_id"],
        "hadm_id": base_res["hadm_id"],
        "baseline_cutoff": cutoff_base,
        "graphaware_cutoff": cutoff_ga,
        "total_events": len(events_comparison),
        "events_comparison": events_comparison
    }


if __name__ == "__main__":
    mcp.run()