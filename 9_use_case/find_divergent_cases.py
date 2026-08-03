"""
Automated Divergent Trajectory Search Script
============================================
Scans the entire test dataset (MIMIC-IV CBC_BMP) to automatically identify 
patient journeys where Traditional XGBoost and GraphAware XGBoost make divergent predictions.

Uses model-specific G-Mean ROC optimal decision cutoffs:
  - Baseline XGBoost Cutoff:   0.001613
  - GraphAware XGBoost Cutoff: 0.000178

Usage:
  .venv/bin/python 9_use_case/find_divergent_cases.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "7_inference"))

from app import run_graphaware_inference

# Model-specific optimal decision cutoffs derived from Validation G-Mean ROC optimization
CUTOFF_BASELINE = 0.001613
CUTOFF_GRAPHAWARE = 0.000178

def find_divergent_trajectories(selected_panel="MIMIC_CBC_BMP", max_display=5):
    print("=" * 85)
    print(" AUTOMATED DIVERGENT PATIENT TRAJECTORY SCANNER ")
    print(f" Panel: {selected_panel}")
    print(f" Model-Specific Cutoffs -> Baseline XGBoost: {CUTOFF_BASELINE:.6f} | GraphAware XGBoost: {CUTOFF_GRAPHAWARE:.6f}")
    print("=" * 85)

    test_csv_path = os.path.join(REPO_ROOT, "1_preprocess/data/preprocessed_data/CBC_BMP/mimic_processed_test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

    df_test = pd.read_csv(test_csv_path)

    clean_panel = selected_panel.replace("MIMIC_", "").replace("SBC_", "")
    model_path = os.path.join(REPO_ROOT, "6_graphaware/models", clean_panel, "final_model.xgb")
    if not os.path.exists(model_path):
        model_path = os.path.join(REPO_ROOT, "6_graphaware/models", f"MIMIC_{clean_panel}", "final_model.xgb")

    baseline_path = os.path.join(REPO_ROOT, "2_baseline/models", clean_panel, "XGBClassifier.pkl")
    if not os.path.exists(baseline_path):
        baseline_path = os.path.join(REPO_ROOT, "2_baseline/models", f"MIMIC_{clean_panel}", "XGBClassifier.pkl")

    baseline_model = joblib.load(baseline_path)
    feature_cols = [c for c in df_test.columns if c.startswith("f__")]

    df_test['y_bin'] = (df_test['y'] == 'Sepsis').astype(int)
    sepsis_patients = df_test[df_test['y_bin'] == 1]['Id'].unique()

    print(f"[1/3] Scanning {len(sepsis_patients)} sepsis patients for divergent time series...")

    divergent_results = []

    for pid in sepsis_patients:
        p_df = df_test[df_test['Id'] == pid].sort_values('Time').copy()
        if len(p_df) < 2:
            continue

        control_events = p_df[p_df['y_bin'] == 0]
        sepsis_events = p_df[p_df['y_bin'] == 1]
        if len(control_events) == 0 or len(sepsis_events) == 0:
            continue

        # Run GraphAware inference in app sub-graph mode
        sorted_df, preds_ga, _, _ = run_graphaware_inference(p_df, model_path, selected_panel)
        preds_base = baseline_model.predict_proba(sorted_df[feature_cols].to_numpy())[:, 1]
        sorted_df['pred_baseline'] = preds_base
        sorted_df['pred_graphaware'] = preds_ga

        # Model-specific thresholding
        sorted_df['base_pos'] = sorted_df['pred_baseline'] >= CUTOFF_BASELINE
        sorted_df['ga_pos'] = sorted_df['pred_graphaware'] >= CUTOFF_GRAPHAWARE

        b_sepsis_hits = sorted_df[sorted_df['y_bin'] == 1]['base_pos'].sum()
        g_sepsis_hits = sorted_df[sorted_df['y_bin'] == 1]['ga_pos'].sum()

        b_control_fps = sorted_df[sorted_df['y_bin'] == 0]['base_pos'].sum()
        g_control_fps = sorted_df[sorted_df['y_bin'] == 0]['ga_pos'].sum()

        base_missed_sepsis = (b_sepsis_hits == 0) and (g_sepsis_hits >= 1)
        base_false_alarms = (b_control_fps > 0) and (g_control_fps == 0) and (g_sepsis_hits >= 1)

        if base_missed_sepsis or base_false_alarms:
            divergent_results.append({
                'Id': pid,
                'subject_id': sorted_df['subject_id'].iloc[0],
                'hadm_id': sorted_df['hadm_id'].iloc[0],
                'total_events': len(sorted_df),
                'divergence_category': 'Baseline Sepsis Miss' if base_missed_sepsis else 'Baseline Control False Alarms',
                'base_sepsis_max': sorted_df[sorted_df['y_bin'] == 1]['pred_baseline'].max(),
                'ga_sepsis_max': sorted_df[sorted_df['y_bin'] == 1]['pred_graphaware'].max(),
                'b_control_fps': b_control_fps,
                'g_control_fps': g_control_fps,
                'patient_df': sorted_df
            })

    print(f"[2/3] Found {len(divergent_results)} divergent patient cases!")
    
    # Sort: Prioritize Baseline Sepsis Misses, then highest GA sepsis probability
    divergent_results.sort(key=lambda x: (x['divergence_category'] == 'Baseline Sepsis Miss', x['ga_sepsis_max']), reverse=True)

    print("\n" + "=" * 85)
    print(f" TOP {min(max_display, len(divergent_results))} DIVERGENT PATIENT JOURNEYS ")
    print("=" * 85)

    for i, case in enumerate(divergent_results[:max_display]):
        pid = case['Id']
        sub_id = case['subject_id']
        hadm_id = case['hadm_id']
        category = case['divergence_category']
        pdf = case['patient_df']

        print(f"\nCase #{i+1}: Patient ID {pid} (Subject: {sub_id}, HADM: {hadm_id}) | Category: [{category}]")
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
            print(f"{hr:>8.1f} | {r['charttime']:>19} | {r['y']:>10} | {r['f__WBC']:>5.1f} | {bp:>10.6f} ({b_flag:>3}) | {gp:>10.6f} ({g_flag:>3})")

    print("\n" + "=" * 85)
    print(" DIVERGENT SCAN COMPLETE ")
    print("=" * 85)

if __name__ == "__main__":
    find_divergent_trajectories()
