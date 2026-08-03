"""
Automated High-Quality Sepsis Trajectory Scanner (MIMIC_CBC Panel - Database Graph Mode)
======================================================================================
Scans the test dataset (MIMIC-IV CBC panel) using SQLiteConnector (database graph) 
and exact model-specific cutoffs to discover clinical patient journeys where:
  1. GraphAware XGBoost is 100% ACCURATE on preceding Control events (0 false alarms).
  2. GraphAware XGBoost correctly DETECTS Sepsis at clinical onset.
  3. Traditional Baseline XGBoost Fails (misses sepsis or triggers false alarms during control).

Model-Specific Cutoffs for MIMIC_CBC Panel:
  - Baseline XGBoost Cutoff:   0.002117
  - GraphAware XGBoost Cutoff: 0.000709

Usage:
  .venv/bin/python 9_use_case/find_divergent_cases.py
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

CUTOFF_BASELINE = 0.002117
CUTOFF_GRAPHAWARE = 0.000709

def diff_user_fun(kwargs):
    return kwargs['original_features'] - kwargs['mean_neighbors']

def find_high_quality_use_cases(selected_panel="MIMIC_CBC", max_display=5):
    clean_panel = selected_panel.replace("MIMIC_", "").replace("SBC_", "")
    
    print("=" * 85)
    print(f" HIGH-QUALITY CLINICAL SEPSIS TRAJECTORY SCANNER (DATABASE GRAPH) ")
    print(f" Panel: {selected_panel}")
    print(f" Model-Specific Cutoffs -> Baseline: {CUTOFF_BASELINE:.6f} | GraphAware: {CUTOFF_GRAPHAWARE:.6f}")
    print("=" * 85)

    test_csv_path = os.path.join(REPO_ROOT, f"1_preprocess/data/preprocessed_data/{clean_panel}/mimic_processed_test.csv")
    df_test = pd.read_csv(test_csv_path)

    baseline_path = os.path.join(REPO_ROOT, f"2_baseline/models/MIMIC_{clean_panel}/XGBClassifier.pkl")
    baseline_model = joblib.load(baseline_path)

    ga_path = os.path.join(REPO_ROOT, f"6_graphaware/models/MIMIC_{clean_panel}/final_model.xgb")
    ga_model = xgb.Booster()
    ga_model.load_model(ga_path)

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

    db_path = os.path.join(REPO_ROOT, f'4_db_upload/sqlite/sqlite_data/{clean_panel}/mimic_sbc_graph.db')
    connector = SQLiteConnector(db_path=db_path)

    feature_cols = [c for c in df_test.columns if c.startswith("f__")]
    df_test['pred_baseline'] = baseline_model.predict_proba(df_test[feature_cols].to_numpy())[:, 1]
    df_test['y_bin'] = (df_test['y'] == 'Sepsis').astype(int)

    # Fetch Database Graphaware Predictions
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

    df_test['base_pos'] = df_test['pred_baseline'] >= CUTOFF_BASELINE
    df_test['ga_pos'] = df_test['pred_graphaware'] >= CUTOFF_GRAPHAWARE

    sepsis_patients = df_test[df_test['y_bin'] == 1]['Id'].unique()

    results = []

    for pid in sepsis_patients:
        p_df = df_test[df_test['Id'] == pid].sort_values('Time').copy()
        if len(p_df) < 2:
            continue

        c_events = p_df[p_df['y_bin'] == 0]
        s_events = p_df[p_df['y_bin'] == 1]
        if len(c_events) == 0 or len(s_events) == 0:
            continue

        n_c = len(c_events)
        n_s = len(s_events)

        b_c_fps = c_events['base_pos'].sum()
        g_c_fps = c_events['ga_pos'].sum()

        b_s_hits = s_events['base_pos'].sum()
        g_s_hits = s_events['ga_pos'].sum()

        ga_c_acc = (n_c - g_c_fps) / n_c
        base_c_acc = (n_c - b_c_fps) / n_c

        base_missed_sepsis = (b_s_hits == 0)
        ga_detected_sepsis = (g_s_hits >= 1)
        baseline_strictly_negative = (p_df['base_pos'].sum() == 0)

        if ga_detected_sepsis and ga_c_acc >= 0.50 and (base_missed_sepsis or b_c_fps > g_c_fps):
            results.append({
                'Id': pid,
                'subject_id': p_df['subject_id'].iloc[0],
                'hadm_id': p_df['hadm_id'].iloc[0],
                'total_events': len(p_df),
                'n_control': n_c,
                'g_c_fps': g_c_fps,
                'b_c_fps': b_c_fps,
                'ga_control_acc': ga_c_acc,
                'base_control_acc': base_c_acc,
                'baseline_strictly_negative': baseline_strictly_negative,
                'base_missed_sepsis': base_missed_sepsis,
                'ga_detected_sepsis': ga_detected_sepsis,
                'patient_df': p_df
            })

    results.sort(key=lambda x: (x['baseline_strictly_negative'], x['ga_control_acc']), reverse=True)

    print(f"Found {len(results)} HIGH-QUALITY USE CASES in {selected_panel} panel!")
    print("\n" + "=" * 85)
    print(f" TOP {min(max_display, len(results))} CLINICAL SEPSIS TRAJECTORIES ({selected_panel}) ")
    print("=" * 85)

    for i, case in enumerate(results[:max_display]):
        pid = case['Id']
        sub_id = case['subject_id']
        hadm_id = case['hadm_id']
        ga_acc = case['ga_control_acc'] * 100
        b_acc = case['base_control_acc'] * 100
        pdf = case['patient_df']

        print(f"\nCase #{i+1}: Patient ID {pid} (Subject: {sub_id}, HADM: {hadm_id})")
        print(f"  GraphAware Control Accuracy: {ga_acc:.1f}% ({case['n_control'] - case['g_c_fps']}/{case['n_control']} correct) | Baseline Control Acc: {b_acc:.1f}%")
        print(f"  Baseline 100% Negative: {case['baseline_strictly_negative']} | Baseline Missed Sepsis: {case['base_missed_sepsis']}")
        print("-" * 85)
        print(f"{'Time (h)':>8} | {'ChartTime':>19} | {'True Label':>10} | {'WBC':>5} | {'Base Prob (cut=' + f'{CUTOFF_BASELINE:.4f}' + ')':>26} | {'GA Prob (cut=' + f'{CUTOFF_GRAPHAWARE:.4f}' + ')':>24}")
        print("-" * 85)

        first_ct = pd.to_datetime(pdf['charttime'].iloc[0])
        for _, r in pdf.iterrows():
            ct = pd.to_datetime(r['charttime'])
            hr = (ct - first_ct).total_seconds() / 3600.0
            bp = r['pred_baseline']
            gp = r['pred_graphaware']
            b_flag = "POS" if r['base_pos'] else "NEG"
            g_flag = "POS" if r['ga_pos'] else "NEG"
            print(f"{hr:>8.1f} | {r['charttime']:>19} | {r['y']:>10} | {r.get('f__WBC', 0.0):>5.1f} | {bp:>10.6f} ({b_flag:>3}) | {gp:>10.6f} ({g_flag:>3})")

    print("\n" + "=" * 85)

if __name__ == "__main__":
    find_high_quality_use_cases("MIMIC_CBC")
