"""
Specific Clinical Use Case Evaluator: Patient 326234 (MIMIC_CBC Panel)
====================================================================
Evaluates Patient 326234 on MIMIC_CBC panel comparing GraphAware XGBoost vs Baseline XGBoost
using the 7_inference/app.py single-patient inference engine.

Usage:
  .venv/bin/python 9_use_case/find_cbc_sepsis_use_case.py
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

from app import run_graphaware_inference, get_default_cutoff

def evaluate_patient_326234():
    test_csv = os.path.join(REPO_ROOT, "1_preprocess/data/preprocessed_data/CBC/mimic_processed_test.csv")
    df_test = pd.read_csv(test_csv)
    
    p_df = df_test[df_test['Id'] == 326234].sort_values('Time').copy()
    
    baseline_path = os.path.join(REPO_ROOT, "2_baseline/models/MIMIC_CBC/XGBClassifier.pkl")
    baseline_model = joblib.load(baseline_path)
    feature_cols = [c for c in p_df.columns if c.startswith("f__")]
    
    base_probs = baseline_model.predict_proba(p_df[feature_cols].to_numpy())[:, 1]
    p_df['pred_base'] = base_probs
    
    ga_model_path = os.path.join(REPO_ROOT, "6_graphaware/models/MIMIC_CBC/final_model.xgb")
    _, ga_probs, _, _ = run_graphaware_inference(p_df, ga_model_path, "CBC")
    p_df['pred_ga'] = ga_probs
    
    cutoff_base = 0.002117
    cutoff_ga = get_default_cutoff("MIMIC_CBC", "CBC/mimic_processed_test.csv")
    
    p_df['base_status'] = np.where(p_df['pred_base'] >= cutoff_base, 'POSITIVE (SEPSIS)', 'NEGATIVE (CONTROL)')
    p_df['ga_status'] = np.where(p_df['pred_ga'] >= cutoff_ga, 'POSITIVE (SEPSIS)', 'NEGATIVE (CONTROL)')
    
    print("=" * 85)
    print(" CLINICAL USE CASE SUMMARY FOR PATIENT 326234 (MIMIC_CBC PANEL) ")
    print("=" * 85)
    print(f"Baseline XGBoost Cutoff:   {cutoff_base:.6f}")
    print(f"GraphAware XGBoost Cutoff: {cutoff_ga:.6f}")
    print("\nTrajectory Results:")
    print(p_df[['Time', 'y', 'pred_base', 'base_status', 'pred_ga', 'ga_status']].to_string(index=False))

if __name__ == "__main__":
    evaluate_patient_326234()
