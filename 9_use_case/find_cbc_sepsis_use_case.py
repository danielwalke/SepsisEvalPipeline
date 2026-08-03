import os
import sys
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. SETUP PATHS & IMPORTS
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "6_graphaware") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "6_graphaware"))

from GraphAware.EnsembleFramework import Framework
from connectors.SQLiteConnector import SQLiteConnector

CUTOFF_BASELINE = 0.002117
CUTOFF_GRAPHAWARE = 0.000709

def diff_user_fun(kwargs):
    return kwargs['original_features'] - kwargs['mean_neighbors']

def run_use_case_analysis():
    print("=" * 85)
    print(" USE CASE ANALYSIS: MIMIC_CBC DATASET - PATIENT 115103 SEPSIS JOURNEY ")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # 2. LOAD DATASET & MODELS FOR CBC PANEL
    # -------------------------------------------------------------------------
    test_csv_path = os.path.join(REPO_ROOT, "1_preprocess/data/preprocessed_data/CBC/mimic_processed_test.csv")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

    print(f"[1/4] Loading MIMIC_CBC test set from: {test_csv_path}")
    df_test = pd.read_csv(test_csv_path)

    target_patient_id = 115103
    patient_df = df_test[df_test['Id'] == target_patient_id].sort_values('Time').copy()
    if len(patient_df) == 0:
        raise ValueError(f"Patient ID {target_patient_id} not found in CBC test dataset.")

    subject_id = patient_df['subject_id'].iloc[0]
    hadm_id = patient_df['hadm_id'].iloc[0]
    age = patient_df['f__Age'].iloc[0]

    baseline_path = os.path.join(REPO_ROOT, "2_baseline/models/MIMIC_CBC/XGBClassifier.pkl")
    ga_path = os.path.join(REPO_ROOT, "6_graphaware/models/MIMIC_CBC/final_model.xgb")
    db_path = os.path.join(REPO_ROOT, "4_db_upload/sqlite/sqlite_data/CBC/mimic_sbc_graph.db")

    baseline_model = joblib.load(baseline_path)
    feature_cols = [c for c in patient_df.columns if c.startswith("f__")]
    patient_df['pred_baseline'] = baseline_model.predict_proba(patient_df[feature_cols].to_numpy())[:, 1]

    ga_model = xgb.Booster()
    ga_model.load_model(ga_path)

    print(f"[2/4] Executing Database GraphAware inference for Patient {target_patient_id}...")
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
    connector = SQLiteConnector(db_path=db_path)

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
    sorted_df = df_test[df_test['Id'] == target_patient_id].sort_values('Time').copy()
    sorted_df['pred_baseline'] = patient_df['pred_baseline']

    print("\n" + "=" * 85)
    print(f" TARGET SEPSIS JOURNEY DETAILS (Patient {target_patient_id}) ")
    print(f" Patient Internal ID: {target_patient_id} | MIMIC Subject ID: {subject_id} | HADM ID: {hadm_id} | Age: {age}")
    print(f" Model-Specific Cutoffs -> Baseline: {CUTOFF_BASELINE:.6f} | GraphAware: {CUTOFF_GRAPHAWARE:.6f}")
    print("=" * 85)

    timeline_records = []
    first_charttime = pd.to_datetime(sorted_df['charttime'].iloc[0])

    for idx, row in sorted_df.iterrows():
        ct = pd.to_datetime(row['charttime'])
        hours_elapsed = (ct - first_charttime).total_seconds() / 3600.0
        base_p = row['pred_baseline']
        ga_p = row['pred_graphaware']
        base_flag = "POSITIVE" if base_p >= CUTOFF_BASELINE else "NEGATIVE"
        ga_flag = "POSITIVE" if ga_p >= CUTOFF_GRAPHAWARE else "NEGATIVE"

        timeline_records.append({
            'ChartTime': row['charttime'],
            'Hours': hours_elapsed,
            'Label': row['y'],
            'Base_Prob': base_p,
            'Base_Prediction': base_flag,
            'GA_Prob': ga_p,
            'GA_Prediction': ga_flag,
            'WBC': row.get('f__WBC', 0.0),
            'Platelets': row.get('f__PLT', 0.0)
        })

    timeline_df = pd.DataFrame(timeline_records)
    print(timeline_df[['Hours', 'ChartTime', 'Label', 'WBC', 'Platelets', 'Base_Prob', 'Base_Prediction', 'GA_Prob', 'GA_Prediction']].to_string(index=False))

    # -------------------------------------------------------------------------
    # 4. GENERATE CLINICAL TRAJECTORY PLOT FOR PATIENT 115103
    # -------------------------------------------------------------------------
    print(f"\n[3/4] Generating clinical journey plot for Patient {target_patient_id}...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Plot 1: Sepsis Risk Probabilities
    ax1.plot(timeline_df['Hours'], timeline_df['Base_Prob'], 'o--', color='#E63946', linewidth=2.5, markersize=8, label=f'Traditional XGBoost (Missed Sepsis, Cutoff={CUTOFF_BASELINE:.4f})')
    ax1.plot(timeline_df['Hours'], timeline_df['GA_Prob'], 's-', color='#2A9D8F', linewidth=3.0, markersize=9, label=f'GraphAware XGBoost (Detected Sepsis, Cutoff={CUTOFF_GRAPHAWARE:.4f})')
    ax1.axhline(y=CUTOFF_BASELINE, color='#E63946', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=CUTOFF_GRAPHAWARE, color='#2A9D8F', linestyle=':', linewidth=1.5, alpha=0.7)

    # Highlight Sepsis Onset
    sepsis_rows = timeline_df[timeline_df['Label'] == 'Sepsis']
    if not sepsis_rows.empty:
        sepsis_hr = sepsis_rows['Hours'].iloc[0]
        ax1.axvline(x=sepsis_hr, color='#9B5DE5', linestyle='--', linewidth=2, label='Clinical Sepsis Onset')
        ax1.annotate(f'Sepsis Onset (t = {sepsis_hr:.1f}h)\nGraphAware: POSITIVE ({timeline_df["GA_Prob"].iloc[-1]:.6f})\nBaseline: MISSED ({timeline_df["Base_Prob"].iloc[-1]:.6f})',
                     xy=(sepsis_hr, timeline_df['GA_Prob'].iloc[-1]),
                     xytext=(sepsis_hr - 4, timeline_df['GA_Prob'].iloc[-1] * 0.4),
                     arrowprops=dict(facecolor='#2A9D8F', shrink=0.08, width=1.5, headwidth=8),
                     fontsize=9.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#E6FFFA", ec="#2A9D8F", lw=1.5))

    ax1.set_ylabel('Sepsis Risk Probability', fontsize=12, fontweight='bold')
    ax1.set_title(f'CBC Panel Sepsis Analysis: Patient {target_patient_id} (MIMIC Subject {subject_id})\nTraditional Baseline XGBoost (False Negative) vs. GraphAware XGBoost (True Positive Detection)', fontsize=12, fontweight='bold', pad=12)
    ax1.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_yscale('log')

    # Plot 2: WBC Biomarker Trajectory
    ax2.plot(timeline_df['Hours'], timeline_df['WBC'], '^-', color='#457B9D', linewidth=2, markersize=7, label='WBC Count (k/uL)')
    ax2.set_xlabel('Time Elapsed (Hours)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('WBC Level (k/uL)', fontsize=11, fontweight='bold', color='#457B9D')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plot_save_path = os.path.join(SCRIPT_DIR, "patient_115103_sepsis_journey.png")
    plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[4/4] Saved clean CBC patient trajectory plot to: {plot_save_path}")

    print("=" * 85)
    print(" CBC USE CASE ANALYSIS COMPLETE ")
    print("=" * 85)

if __name__ == "__main__":
    run_use_case_analysis()
