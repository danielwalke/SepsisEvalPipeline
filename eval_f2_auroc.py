"""
Evaluation script: AUROC + best threshold via F2-score optimisation
for GraphAware XGBoost and Baseline XGBoost across all feature panels.

Predictions are cached to <REPO_ROOT>/pred_cache/<panel>_<model>.npy
so re-running for metric changes does not require re-inference.

Metrics reported at the best F2 threshold:
  - AUROC          (threshold-independent)
  - Best F2 threshold (t*)
  - F2  = (1 + 2²)·PPV·Sens / (2²·PPV + Sens)
  - PPV (Positive Predictive Value = Precision)  = TP / (TP + FP)
  - Sensitivity (Recall / TPR)                   = TP / (TP + FN)
  - Specificity (TNR)                            = TN / (TN + FP)
  - F1  = harmonic mean of PPV & Sensitivity

Usage:
    python eval_f2_auroc.py
"""

import os
import sys
import pickle
import importlib.util
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
from sklearn.metrics import (
    roc_auc_score, fbeta_score, precision_score,
    recall_score, f1_score, roc_curve, confusion_matrix
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
GRAPHAWARE_DIR = os.path.join(REPO_ROOT, "6_graphaware")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if GRAPHAWARE_DIR not in sys.path:
    sys.path.insert(0, GRAPHAWARE_DIR)

# Dynamically import GraphPreprocesser
graph_main_path = os.path.join(REPO_ROOT, "3_graph_construction", "main.py")
spec = importlib.util.spec_from_file_location("graph_main", graph_main_path)
graph_main = importlib.util.module_from_spec(spec)
sys.modules["graph_main"] = graph_main
spec.loader.exec_module(graph_main)
GraphPreprocesser = graph_main.GraphPreprocesser

from GraphAware.EnsembleFramework import Framework

# ---------------------------------------------------------------------------
# Feature sets & panels
# ---------------------------------------------------------------------------
PANEL_BASE_FEATURES = {
    "CBC":         ["f__Age", "f__HGB", "f__MCV", "f__PLT", "f__RBC", "f__Sex", "f__WBC"],
    "CBC_BMP":     ["f__Age", "f__Bicarbonate", "f__Calcium_Total", "f__Chloride",
                    "f__Creatinine", "f__Glucose", "f__HGB", "f__MCV", "f__PLT",
                    "f__Potassium", "f__RBC", "f__Sex", "f__Sodium",
                    "f__Urea Nitrogen", "f__WBC"],
    "CBC_BMP_HIL": ["f__Age", "f__Bicarbonate", "f__Calcium_Total", "f__Chloride",
                    "f__Creatinine", "f__Glucose", "f__H", "f__HGB", "f__I", "f__L",
                    "f__MCV", "f__PLT", "f__Potassium", "f__RBC", "f__Sex",
                    "f__Sodium", "f__Urea Nitrogen", "f__WBC"],
    "CBC_HIL":     ["f__Age", "f__H", "f__HGB", "f__I", "f__L", "f__MCV",
                    "f__PLT", "f__RBC", "f__Sex", "f__WBC"],
}

PANELS = list(PANEL_BASE_FEATURES.keys())

DATA_ROOT             = os.path.join(REPO_ROOT, "1_preprocess", "data", "preprocessed_data")
GRAPHAWARE_MODELS_ROOT = os.path.join(REPO_ROOT, "6_graphaware", "models")
BASELINE_MODELS_ROOT   = os.path.join(REPO_ROOT, "2_baseline", "models")
CACHE_DIR              = os.path.join(REPO_ROOT, "pred_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: encode labels safely
# ---------------------------------------------------------------------------
def encode_labels(arr) -> np.ndarray:
    if arr.dtype.kind in ("U", "O", "S"):
        return (arr == "Sepsis").astype(int)
    return arr.astype(int)


# ---------------------------------------------------------------------------
# Helper: best F2 threshold
# ---------------------------------------------------------------------------
def best_threshold_f2(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0):
    """
    Finds the threshold t* in [min_score, max_score] that maximises F_beta.

    Strategy:
      - Use sklearn roc_curve to get all distinct decision thresholds
        (these correspond to every unique predicted score).
      - Drop the first sentinel value that sklearn appends (max_score + 1),
        which causes all predictions to be 0 and is not a real threshold.
      - Evaluate F_beta at every remaining threshold and return the best one.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)

    # sklearn prepends a sentinel = max(y_prob) + 1 so that (fpr=0, tpr=0).
    # Remove it so we only scan real score values.
    real_mask = thresholds <= y_prob.max()
    thresholds = thresholds[real_mask]

    best_thresh = thresholds[0] if len(thresholds) > 0 else 0.5
    best_f2 = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thresh = t

    return float(best_thresh), float(best_f2)


# ---------------------------------------------------------------------------
# Helper: full metrics dict at a given threshold
# ---------------------------------------------------------------------------
def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f2  = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    return {
        "PPV (Precision)": round(ppv, 4),
        "Sensitivity (Recall)": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "F2": round(f2, 4),
        "F1": round(f1, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


# ---------------------------------------------------------------------------
# Helper: GraphAware inference
# ---------------------------------------------------------------------------
def diff_user_fun(kwargs):
    return kwargs["original_features"] - kwargs["mean_neighbors"]


def run_graphaware_inference(df: pd.DataFrame, model_path: str, f_cols: list, cache_key: str = None):
    """Run GraphAware inference; loads from cache if available."""
    if cache_key:
        cp, cl = os.path.join(CACHE_DIR, f"{cache_key}_probs.npy"), os.path.join(CACHE_DIR, f"{cache_key}_labels.npy")
        if os.path.exists(cp) and os.path.exists(cl):
            print(f"    [cache hit] {cache_key}")
            return None, np.load(cp), np.load(cl)
    model = xgb.Booster()
    model.load_model(model_path)

    has_graph_meta = ("Id" in df.columns and "Time" in df.columns)
    if has_graph_meta:
        gp = GraphPreprocesser(df.copy())
        gp.sort_data()
        sorted_df = gp.data.reset_index(drop=True)
        edge_index, edge_weight = gp.get_edges()
    else:
        sorted_df = df.reset_index(drop=True)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float)

    for col in f_cols:
        if col not in sorted_df.columns:
            sorted_df[col] = 0.0

    X = sorted_df[sorted(f_cols)].to_numpy(dtype=np.float32)
    hops = [0, 1]
    framework = Framework(
        user_functions=[diff_user_fun for _ in hops],
        hops_list=hops,
        clfs=[None for _ in hops],
        gpu_idx=None,
        handle_nan=0.0,
        attention_configs=[None for _ in hops],
        classifier_on_device=False,
    )
    feats_list = framework.get_features(X, edge_index, edge_weight)
    final_feats = np.concatenate(
        [f.cpu().numpy() if hasattr(f, "cpu") else f for f in feats_list], axis=1
    )
    dtest = xgb.DMatrix(final_feats)
    preds_prob = model.predict(dtest)
    if cache_key:
        np.save(os.path.join(CACHE_DIR, f"{cache_key}_probs.npy"), preds_prob)
        labels = encode_labels(sorted_df[sorted_df.columns[sorted_df.columns.str.endswith('y') | (sorted_df.columns == 'Label')][0]].values) if any(c in sorted_df.columns for c in ['y','Label']) else None
        if labels is not None:
            np.save(os.path.join(CACHE_DIR, f"{cache_key}_labels.npy"), labels)
    return sorted_df, preds_prob, None


# ---------------------------------------------------------------------------
# Helper: Baseline XGBoost inference
# ---------------------------------------------------------------------------
def run_baseline_xgb_inference(df: pd.DataFrame, model_path: str, f_cols: list, cache_key: str = None) -> np.ndarray:
    """Run Baseline XGBoost inference; loads from cache if available."""
    if cache_key:
        cp = os.path.join(CACHE_DIR, f"{cache_key}_probs.npy")
        if os.path.exists(cp):
            print(f"    [cache hit] {cache_key}")
            return np.load(cp)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    for col in f_cols:
        if col not in df.columns:
            df = df.copy()
            df[col] = 0.0
    X = df[sorted(f_cols)].to_numpy(dtype=np.float32)
    probs = model.predict_proba(X)[:, 1]
    if cache_key:
        np.save(os.path.join(CACHE_DIR, f"{cache_key}_probs.npy"), probs)
    return probs


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------
SEP = "=" * 64

def print_result(label, auroc, thresh, m):
    print(f"\n  [{label}]")
    print(f"    AUROC              : {auroc:.4f}")
    print(f"    Best F2 threshold  : {thresh:.6f}")
    print(f"    F2  @ threshold    : {m['F2']:.4f}")
    print(f"    PPV (Precision)    : {m['PPV (Precision)']:.4f}")
    print(f"    Sensitivity(Recall): {m['Sensitivity (Recall)']:.4f}")
    print(f"    Specificity        : {m['Specificity']:.4f}")
    print(f"    F1  @ threshold    : {m['F1']:.4f}")
    print(f"    TP={m['TP']:,}  FP={m['FP']:,}  TN={m['TN']:,}  FN={m['FN']:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results = []

    for panel in PANELS:
        test_csv = os.path.join(DATA_ROOT, panel, "mimic_processed_test.csv")
        if not os.path.exists(test_csv):
            print(f"[SKIP] Test CSV not found: {test_csv}")
            continue

        print(f"\n{SEP}\nPanel: {panel}\n{SEP}")
        df_test = pd.read_csv(test_csv)

        label_col = ("y" if "y" in df_test.columns
                     else ("Label" if "Label" in df_test.columns else None))
        if label_col is None:
            print(f"[SKIP] No label column.")
            continue

        y_true_raw = df_test[label_col].values
        y_true = encode_labels(y_true_raw)
        f_cols = PANEL_BASE_FEATURES[panel]

        # ── GraphAware XGBoost ──────────────────────────────────────────
        ga_model_path = os.path.join(GRAPHAWARE_MODELS_ROOT, f"MIMIC_{panel}", "final_model.xgb")
        if os.path.exists(ga_model_path):
            try:
                sorted_df, ga_probs = run_graphaware_inference(df_test.copy(), ga_model_path, f_cols)
                if "Id" in df_test.columns and "Time" in df_test.columns:
                    y_ga = encode_labels(sorted_df[label_col].values)
                else:
                    y_ga = y_true

                ga_auroc = roc_auc_score(y_ga, ga_probs)
                ga_thresh, _ = best_threshold_f2(y_ga, ga_probs)
                ga_m = metrics_at_threshold(y_ga, ga_probs, ga_thresh)
                print_result("GraphAware XGBoost", ga_auroc, ga_thresh, ga_m)

                results.append({
                    "Panel": panel, "Model": "GraphAware XGBoost",
                    "AUROC": round(ga_auroc, 4),
                    "Best_F2_Threshold": round(ga_thresh, 6),
                    **ga_m,
                })
            except Exception as e:
                print(f"  [ERROR] {e}")
        else:
            print(f"  [SKIP] Model not found: {ga_model_path}")

        # ── Baseline XGBoost ────────────────────────────────────────────
        bl_model_path = os.path.join(BASELINE_MODELS_ROOT, f"MIMIC_{panel}", "XGBClassifier.pkl")
        if os.path.exists(bl_model_path):
            try:
                bl_probs = run_baseline_xgb_inference(df_test.copy(), bl_model_path, f_cols)
                bl_auroc = roc_auc_score(y_true, bl_probs)
                bl_thresh, _ = best_threshold_f2(y_true, bl_probs)
                bl_m = metrics_at_threshold(y_true, bl_probs, bl_thresh)
                print_result("Baseline XGBoost", bl_auroc, bl_thresh, bl_m)

                results.append({
                    "Panel": panel, "Model": "Baseline XGBoost",
                    "AUROC": round(bl_auroc, 4),
                    "Best_F2_Threshold": round(bl_thresh, 6),
                    **bl_m,
                })
            except Exception as e:
                print(f"  [ERROR] {e}")
        else:
            print(f"  [SKIP] Model not found: {bl_model_path}")

    # ── Summary comparison table ────────────────────────────────────────
    if not results:
        print("\nNo results computed.")
        return

    df_res = pd.DataFrame(results)
    out_csv = os.path.join(REPO_ROOT, "f2_auroc_results.csv")
    df_res.to_csv(out_csv, index=False)

    print(f"\n\n{SEP}")
    print("COMPARISON SUMMARY — GraphAware XGBoost vs Baseline XGBoost")
    print(f"{SEP}")

    col_order = ["Panel", "Model", "AUROC", "Best_F2_Threshold",
                 "F2", "PPV (Precision)", "Sensitivity (Recall)", "Specificity", "F1"]
    print(df_res[col_order].to_string(index=False))
    print(f"\nFull results (incl. TP/FP/TN/FN) saved to: {out_csv}")


if __name__ == "__main__":
    main()
