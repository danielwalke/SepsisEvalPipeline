"""
Simple Test Script: Dynamic Panel Sepsis Trajectory Test
=====================================================================================
Evaluates patient time series trajectories for any laboratory panel (e.g. MIMIC_CBC_BMP, MIMIC_CBC) 
using database graph mode, model-specific cutoffs, and dynamic asset resolution.

Usage:
  # Default (CBC_BMP panel, Patient 542802):
  .venv/bin/python 9_use_case/run_simple_test.py

  # Specific panel (CBC panel, Patient 115103):
  .venv/bin/python 9_use_case/run_simple_test.py MIMIC_CBC 115103
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "6_graphaware"))

from GraphAware.EnsembleFramework import Framework
from connectors.SQLiteConnector import SQLiteConnector

def diff_user_fun(kwargs):
    return kwargs['original_features'] - kwargs['mean_neighbors']

def get_panel_assets(selected_panel: str = "MIMIC_CBC_BMP"):
    clean_panel = selected_panel.replace("MIMIC_", "").replace("SBC_", "")
    panel_full_name = f"MIMIC_{clean_panel}"

    csv_path = os.path.join(REPO_ROOT, "1_preprocess", "data", "preprocessed_data", clean_panel, "mimic_processed_test.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(REPO_ROOT, "1_preprocess", "data", "preprocessed_data", panel_full_name, "mimic_processed_test.csv")

    baseline_path = os.path.join(REPO_ROOT, "2_baseline", "models", panel_full_name, "XGBClassifier.pkl")
    if not os.path.exists(baseline_path):
        baseline_path = os.path.join(REPO_ROOT, "2_baseline", "models", clean_panel, "XGBClassifier.pkl")

    ga_model_path = os.path.join(REPO_ROOT, "6_graphaware", "models", panel_full_name, "final_model.xgb")
    if not os.path.exists(ga_model_path):
        ga_model_path = os.path.join(REPO_ROOT, "6_graphaware", "models", clean_panel, "final_model.xgb")

    db_path = os.path.join(REPO_ROOT, "4_db_upload", "sqlite", "sqlite_data", clean_panel, "mimic_sbc_graph.db")
    if not os.path.exists(db_path):
        db_path = os.path.join(REPO_ROOT, "4_db_upload", "sqlite", "sqlite_data", panel_full_name, "mimic_sbc_graph.db")

    cutoff_base_dict = {'CBC_BMP': 0.001613, 'CBC': 0.002117, 'BMP': 0.002000}
    cutoff_ga_dict = {'CBC_BMP': 0.000178, 'CBC': 0.000709, 'BMP': 0.000200}

    return {
        "panel_name": panel_full_name,
        "clean_panel": clean_panel,
        "csv_path": csv_path,
        "baseline_path": baseline_path,
        "ga_model_path": ga_model_path,
        "db_path": db_path if os.path.exists(db_path) else None,
        "cutoff_baseline": cutoff_base_dict.get(clean_panel, 0.001613),
        "cutoff_graphaware": cutoff_ga_dict.get(clean_panel, 0.000178)
    }

def main():
    selected_panel = sys.argv[1] if len(sys.argv) > 1 else "MIMIC_CBC_BMP"
    assets = get_panel_assets(selected_panel)

    default_pids = {"MIMIC_CBC": 115103, "CBC": 115103, "MIMIC_CBC_BMP": 542802, "CBC_BMP": 542802}
    patient_id = int(sys.argv[2]) if len(sys.argv) > 2 else default_pids.get(assets["panel_name"], 542802)

    cutoff_base = assets["cutoff_baseline"]
    cutoff_ga = assets["cutoff_graphaware"]

    print("=" * 85)
    print(f" CLINICAL USE CASE TEST: PATIENT {patient_id} ")
    print(f" Panel Selected: {selected_panel} | Panel Used: {assets['panel_name']}")
    print(f" Model-Specific Cutoffs -> Baseline: {cutoff_base:.6f} | GraphAware: {cutoff_ga:.6f}")
    print("=" * 85)

    if not os.path.exists(assets["csv_path"]):
        print(f"ERROR: Dataset CSV not found at {assets['csv_path']}")
        return

    df_test = pd.read_csv(assets["csv_path"])
    p_df = df_test[df_test['Id'] == patient_id].sort_values('Time').copy()
    if len(p_df) == 0:
        print(f"ERROR: Patient ID {patient_id} not found in panel {assets['panel_name']}.")
        return

    baseline_model = joblib.load(assets["baseline_path"])
    ga_model = xgb.Booster()
    ga_model.load_model(assets["ga_model_path"])

    hops = [0, 1]
    framework = Framework(
        user_functions=[diff_user_fun for _ in hops],
        hops_list=hops,
        clfs=[None for _ in hops],
        gpu_idx=0,
        handle_nan=0.0,
        attention_configs=[None for _ in hops],
        classifier_on_device=False
    )

    feature_cols = [c for c in p_df.columns if c.startswith("f__")]
    preds_baseline = baseline_model.predict_proba(p_df[feature_cols].to_numpy())[:, 1]

    if assets["db_path"] and os.path.exists(assets["db_path"]):
        connector = SQLiteConnector(db_path=assets["db_path"])
        skip = 0
        test_preds_ga = []
        while True:
            X_batch, y_batch = connector.fetch_data_batch('MIMIC_TEST', '', skip, 10000, framework)
            if len(y_batch) == 0:
                break
            preds = ga_model.predict(xgb.DMatrix(X_batch))
            test_preds_ga.extend(preds)
            skip += 10000
        connector.close()
        df_test['pred_graphaware'] = test_preds_ga[:len(df_test)]
        p_df = df_test[df_test['Id'] == patient_id].sort_values('Time').copy()
    else:
        from GraphAware.Inference import run_graphaware_inference
        _, preds_ga, _, _ = run_graphaware_inference(p_df, assets["ga_model_path"], selected_panel)
        p_df['pred_graphaware'] = preds_ga

    p_df['pred_baseline'] = preds_baseline

    print("\n" + "-" * 90)
    print(f"{'Time (h)':>8} | {'True Label':>10} | {'WBC':>5} | {'Base Prob':>11} | {'Base Status':>11} | {'GA Prob':>11} | {'GA Status':>11}")
    print("-" * 90)

    first_charttime = pd.to_datetime(p_df['charttime'].iloc[0])
    for _, r in p_df.iterrows():
        ct = pd.to_datetime(r['charttime'])
        hr = (ct - first_charttime).total_seconds() / 3600.0
        b_p = r['pred_baseline']
        g_p = r['pred_graphaware']
        b_status = "POSITIVE" if b_p >= cutoff_base else "NEGATIVE"
        g_status = "POSITIVE" if g_p >= cutoff_ga else "NEGATIVE"

        print(f"{hr:>8.1f} | {r['y']:>10} | {r.get('f__WBC', 0.0):>5.1f} | {b_p:>11.6f} | {b_status:>11} | {g_p:>11.6f} | {g_status:>11}")

    print("-" * 90)
    print(f"SUMMARY FOR PATIENT {patient_id} ON PANEL {assets['panel_name']}:")
    print(f"  - Baseline XGBoost Cutoff:   {cutoff_base:.6f}")
    print(f"  - GraphAware XGBoost Cutoff: {cutoff_ga:.6f}")
    print("=" * 85)

if __name__ == "__main__":
    main()
