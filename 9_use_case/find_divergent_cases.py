"""
Automated High-Quality Sepsis Trajectory Scanner (MIMIC_CBC Panel - App.py Inference Engine)
==========================================================================================
Scans the test dataset using 7_inference/app.py single-patient inference and exact optimal cutoffs
from 7_inference/optimal_cutoffs.json to discover clinical patient journeys where:
  1. GraphAware XGBoost detects Sepsis at onset / during sepsis.
  2. Baseline XGBoost fails (misses Sepsis).

Usage:
  .venv/bin/python 9_use_case/find_divergent_cases.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "7_inference"))

from app import run_graphaware_inference, get_default_cutoff

def find_high_quality_use_cases(selected_panel="MIMIC_CBC", max_display=5):
    clean_panel = selected_panel.replace("MIMIC_", "").replace("SBC_", "")
    
    test_csv = os.path.join(REPO_ROOT, f"1_preprocess/data/preprocessed_data/{clean_panel}/mimic_processed_test.csv")
    df_test = pd.read_csv(test_csv)
    
    baseline_path = os.path.join(REPO_ROOT, f"2_baseline/models/{selected_panel}/XGBClassifier.pkl")
    baseline_model = joblib.load(baseline_path)
    feature_cols = [c for c in df_test.columns if c.startswith("f__")]
    
    ga_model_path = os.path.join(REPO_ROOT, f"6_graphaware/models/{selected_panel}/final_model.xgb")
    
    cutoff_baseline_dict = {
        'CBC': 0.002117,
        'CBC_BMP': 0.001613,
        'CBC_HIL': 0.001565,
        'CBC_BMP_HIL': 0.000956
    }
    cutoff_base = cutoff_baseline_dict.get(clean_panel, 0.001613)
    cutoff_ga = get_default_cutoff(selected_panel, f"{clean_panel}/mimic_processed_test.csv")
    
    print(f"Scanning test set for panel {selected_panel}...")
    print(f"  Baseline XGBoost Cutoff:   {cutoff_base:.6f}")
    print(f"  GraphAware XGBoost Cutoff: {cutoff_ga:.6f}")
    
    patient_gt = df_test.groupby('Id')['y'].apply(lambda s: (s == 'Sepsis').any())
    sepsis_pids = patient_gt[patient_gt].index.tolist()
    
    found_count = 0
    for pid in sepsis_pids:
        p_df = df_test[df_test['Id'] == pid].sort_values('Time').copy()
        if len(p_df) < 3 or len(p_df) > 10:
            continue
            
        base_probs = baseline_model.predict_proba(p_df[feature_cols].to_numpy())[:, 1]
        p_df['pred_base'] = base_probs
        
        _, ga_probs, _, _ = run_graphaware_inference(p_df, ga_model_path, clean_panel)
        p_df['pred_ga'] = ga_probs
        
        p_df['base_status'] = np.where(p_df['pred_base'] >= cutoff_base, 'POSITIVE', 'NEGATIVE')
        p_df['ga_status'] = np.where(p_df['pred_ga'] >= cutoff_ga, 'POSITIVE', 'NEGATIVE')
        
        # Check if GraphAware correctly flagged Sepsis while Baseline missed it
        sepsis_rows = p_df[p_df['y'] == 'Sepsis']
        if len(sepsis_rows) == 0:
            continue
            
        ga_sepsis_correct = (sepsis_rows['ga_status'] == 'POSITIVE').any()
        base_sepsis_missed = (sepsis_rows['base_status'] == 'NEGATIVE').any()
        
        if ga_sepsis_correct and base_sepsis_missed:
            found_count += 1
            print("\n" + "=" * 80)
            print(f"Found Divergent Use Case #{found_count}: Patient ID {pid}")
            print("=" * 80)
            print(p_df[['Time', 'y', 'pred_base', 'base_status', 'pred_ga', 'ga_status']].to_string(index=False))
            
            if found_count >= max_display:
                break

if __name__ == "__main__":
    find_high_quality_use_cases("MIMIC_CBC")
