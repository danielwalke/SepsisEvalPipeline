import os
import sys

# Auto-reexec under project's .venv if launched via system python (e.g. anaconda)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
venv_dir = os.path.join(repo_root, ".venv")
venv_python = os.path.join(venv_dir, "bin", "python")

if os.path.exists(venv_python) and not sys.prefix.startswith(venv_dir):
    cmd = [venv_python, "-m", "streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
    os.execv(venv_python, cmd)

import glob
import time
import json
import math
import importlib.util
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score,
    f1_score, fbeta_score, accuracy_score, precision_recall_curve, auc, average_precision_score
)
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. SETUP PATHS & IMPORTS
# -----------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GRAPHAWARE_DIR = os.path.join(REPO_ROOT, "6_graphaware")
if GRAPHAWARE_DIR not in sys.path:
    sys.path.insert(0, GRAPHAWARE_DIR)

# Dynamically import GraphPreprocesser from 3_graph_construction/main.py
graph_main_path = os.path.join(REPO_ROOT, "3_graph_construction", "main.py")
spec = importlib.util.spec_from_file_location("graph_main", graph_main_path)
graph_main = importlib.util.module_from_spec(spec)
sys.modules["graph_main"] = graph_main
spec.loader.exec_module(graph_main)
GraphPreprocesser = graph_main.GraphPreprocesser

# Import GraphAware Framework
from GraphAware.EnsembleFramework import Framework

# -----------------------------------------------------------------------------
# 2. STREAMLIT PAGE CONFIG & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sepsis Risk Prediction - GraphFlow Inference",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive dashboard theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .opt-badge {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        color: #047857;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 8px;
    }
    /* Responsive Display Enhancements for Smaller Screen Sizes & Tablets */
    @media (max-width: 992px) {
        .main-header { font-size: 1.5rem !important; }
        .sub-header { font-size: 0.9rem !important; }
        .opt-badge { font-size: 0.8rem !important; padding: 4px 8px !important; }
        
        /* Stack columns vertically below each other on smaller displays */
        div[data-testid="column"], div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
            margin-bottom: 14px !important;
        }

        /* Smaller font sizes for metrics */
        div[data-testid="stMetricValue"] {
            font-size: 1.25rem !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
        }
        
        /* Smaller font & scrollable dataframes */
        .stDataFrame, .stTable {
            font-size: 0.8rem !important;
            overflow-x: auto !important;
            display: block !important;
        }
    }
    @media (max-width: 576px) {
        .main-header { font-size: 1.2rem !important; }
        .sub-header { font-size: 0.8rem !important; }
        div[data-testid="stMetricValue"] {
            font-size: 1.05rem !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS & OPTIMAL CUTOFF LOOKUP
# -----------------------------------------------------------------------------
@st.cache_data
def get_available_trained_models():
    """Scans for trained models in trained_models and 6_graphaware/models."""
    search_paths = [
        os.path.join(REPO_ROOT, "trained_models"),
        os.path.join(REPO_ROOT, "6_graphaware", "models")
    ]
    models_dict = {}
    for base in search_paths:
        if os.path.exists(base):
            for model_file in glob.glob(os.path.join(base, "**", "final_model.xgb"), recursive=True):
                panel_name = os.path.basename(os.path.dirname(model_file))
                if panel_name not in models_dict:
                    models_dict[panel_name] = model_file
    return models_dict

@st.cache_data
def get_sample_datasets(selected_panel=None):
    """Finds sample test/train/val CSVs in 1_preprocess/data/preprocessed_data filtered by panel."""
    sample_dir = os.path.join(REPO_ROOT, "1_preprocess", "data", "preprocessed_data")
    samples = {}
    if not os.path.exists(sample_dir):
        return samples

    all_csvs = glob.glob(os.path.join(sample_dir, "**", "*.csv"), recursive=True)

    if selected_panel:
        clean_panel = selected_panel.replace("MIMIC_", "").replace("SBC_", "")
        panel_dir = os.path.join(sample_dir, clean_panel)
        
        # 1. Direct subfolder match (e.g. 1_preprocess/data/preprocessed_data/CBC_BMP/)
        if os.path.exists(panel_dir):
            for csv_file in sorted(glob.glob(os.path.join(panel_dir, "**", "*.csv"), recursive=True)):
                rel_path = os.path.relpath(csv_file, sample_dir)
                samples[rel_path] = csv_file
                
        # 2. Substring match if direct folder not found
        if not samples:
            for csv_file in sorted(all_csvs):
                rel_path = os.path.relpath(csv_file, sample_dir)
                parts = rel_path.split(os.sep)
                if any(part == clean_panel or clean_panel in part for part in parts):
                    samples[rel_path] = csv_file

    # Fallback to all CSVs if no panel specified or no panel match
    if not samples:
        for csv_file in sorted(all_csvs):
            rel_path = os.path.relpath(csv_file, sample_dir)
            samples[rel_path] = csv_file
            
    return samples

@st.cache_data
def get_optimal_cutoffs_dict():
    """Loads pre-calculated optimal cutoffs JSON dictionary."""
    path = os.path.join(os.path.dirname(__file__), "optimal_cutoffs.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def get_default_cutoff(selected_panel, selected_dataset_key):
    """Retrieves stored optimal cutoff (4 decimal precision) for panel/dataset."""
    cutoffs_dict = get_optimal_cutoffs_dict()
    if selected_panel in cutoffs_dict:
        panel_info = cutoffs_dict[selected_panel]
        if selected_dataset_key and selected_dataset_key in panel_info:
            return float(panel_info[selected_dataset_key])
        elif "DEFAULT" in panel_info:
            return float(panel_info["DEFAULT"])
    return 0.0100

def diff_user_fun(kwargs):
    return kwargs["original_features"] - kwargs["mean_neighbors"]

@st.cache_data(show_spinner=False)
def load_csv_dataset(file_path):
    """Loads CSV dataset with cached memory persistence for instant server-side ops."""
    return pd.read_csv(file_path)

@st.cache_data(show_spinner=False)
def filter_and_sort_server_side(
    df,
    selected_patients=None,
    patient_id_query="",
    age_range=None,
    age_col=None,
    gt_filter=None,
    gt_col=None,
    sort_column=None,
    sort_order="Ascending",
    display_limit=200,
    page_number=1
):
    """
    Computes complete dataset filtering and sorting server-side in Python.
    Preserves original 0-indexed row position in column '_row_pos' and returns only display_limit rows for Streamlit UI preview.
    """
    df_work = df.copy()
    if "_row_pos" not in df_work.columns:
        df_work["_row_pos"] = np.arange(len(df_work))
        
    mask = np.ones(len(df_work), dtype=bool)
    
    if selected_patients and "Id" in df_work.columns:
        mask &= df_work["Id"].isin(selected_patients)
        
    if patient_id_query and patient_id_query.strip() and "Id" in df_work.columns:
        query_str = patient_id_query.strip()
        mask &= df_work["Id"].astype(str).str.contains(query_str, case=False, regex=False)
        
    if age_range and age_col and age_col in df_work.columns:
        mask &= (df_work[age_col] >= age_range[0]) & (df_work[age_col] <= age_range[1])
        
    if gt_filter and gt_col and gt_col in df_work.columns:
        mask &= df_work[gt_col].astype(str).isin(gt_filter)
        
    filtered_df = df_work[mask].copy()
    
    if sort_column and sort_column in filtered_df.columns:
        ascending = (sort_order == "Ascending")
        filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending).reset_index(drop=True)
    else:
        filtered_df = filtered_df.reset_index(drop=True)
        
    total_filtered = len(filtered_df)
    start_idx = max(0, (page_number - 1) * display_limit)
    end_idx = min(total_filtered, start_idx + display_limit)
    
    preview_df = filtered_df.iloc[start_idx:end_idx].copy()
    
    return filtered_df, preview_df, total_filtered, len(df)

PANEL_BASE_FEATURES = {
    "CBC": ["f__Age", "f__HGB", "f__MCV", "f__PLT", "f__RBC", "f__Sex", "f__WBC"],
    "CBC_BMP": ["f__Age", "f__Bicarbonate", "f__Calcium_Total", "f__Chloride", "f__Creatinine", "f__Glucose", "f__HGB", "f__MCV", "f__PLT", "f__Potassium", "f__RBC", "f__Sex", "f__Sodium", "f__Urea Nitrogen", "f__WBC"],
    "CBC_BMP_HIL": ["f__Age", "f__Bicarbonate", "f__Calcium_Total", "f__Chloride", "f__Creatinine", "f__Glucose", "f__H", "f__HGB", "f__I", "f__L", "f__MCV", "f__PLT", "f__Potassium", "f__RBC", "f__Sex", "f__Sodium", "f__Urea Nitrogen", "f__WBC"],
    "CBC_COAG": ["f__Age", "f__HGB", "f__INR(PT)", "f__MCV", "f__PLT", "f__PT", "f__PTT", "f__RBC", "f__Sex", "f__WBC"],
    "CBC_HIL": ["f__Age", "f__H", "f__HGB", "f__I", "f__L", "f__MCV", "f__PLT", "f__RBC", "f__Sex", "f__WBC"],
    "CBC_KIDNEY": ["f__Age", "f__HGB", "f__MCV", "f__PLT", "f__RBC", "f__Sex", "f__WBC"],
    "CBC_LIVER": ["f__Age", "f__Alanine Aminotransferase (ALT)", "f__Albumin", "f__Alkaline Phosphatase", "f__Asparate Aminotransferase (AST)", "f__Bilirubin_Total", "f__HGB", "f__MCV", "f__PLT", "f__RBC", "f__Sex", "f__WBC"],
}

def prepare_panel_features(df_input, panel_name, expected_xgb_feats):
    df = df_input.copy()
    expected_base_feats = expected_xgb_feats // 2
    clean_panel = panel_name.replace("MIMIC_", "").replace("SBC_", "")

    target_cols = PANEL_BASE_FEATURES.get(clean_panel, None)
    
    if target_cols:
        for col in target_cols:
            if col not in df.columns:
                raw_col = col.replace("f__", "")
                if raw_col in df.columns:
                    df[col] = df[raw_col]
                else:
                    df[col] = 0.0
        f_cols = sorted(target_cols)
    else:
        all_f = sorted([c for c in df.columns if c.startswith("f__")])
        if len(all_f) >= expected_base_feats:
            f_cols = all_f[:expected_base_feats]
        else:
            f_cols = list(all_f)
            while len(f_cols) < expected_base_feats:
                dummy_col = f"f__dummy_{len(f_cols)}"
                df[dummy_col] = 0.0
                f_cols.append(dummy_col)
            f_cols = sorted(f_cols)

    return df, f_cols

def run_graphaware_inference(df_input, model_path, panel_name):
    """Executes GraphAware feature aggregation and XGBoost inference."""
    model = xgb.Booster()
    model.load_model(model_path)
    expected_xgb_feats = model.num_features()

    # 1. Select exact expected base features for the model
    df_prep, f_cols = prepare_panel_features(df_input, panel_name, expected_xgb_feats)

    # 2. Graph construction
    has_graph_meta = ("Id" in df_prep.columns and "Time" in df_prep.columns)
    if has_graph_meta:
        gp = GraphPreprocesser(df_prep)
        gp.sort_data()
        sorted_df = gp.data.reset_index(drop=True)
        edge_index, edge_weight = gp.get_edges()
    else:
        sorted_df = df_prep.reset_index(drop=True)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float)

    # 3. Extract matrix X
    X = sorted_df[f_cols].to_numpy(dtype=np.float32)

    # 4. Apply GraphAware Ensemble Framework
    hops = [0, 1]
    framework = Framework(
        user_functions=[diff_user_fun for _ in hops],
        hops_list=hops,
        clfs=[None for _ in hops],
        gpu_idx=None,
        handle_nan=0.0,
        attention_configs=[None for _ in hops]
    )

    feats_list = framework.get_features(X, edge_index, edge_weight)
    final_feats = np.concatenate([f.cpu().numpy() if hasattr(f, 'cpu') else f for f in feats_list], axis=1)

    # 5. Predict probabilities
    dtest = xgb.DMatrix(final_feats)
    preds_prob = model.predict(dtest)
    
    return sorted_df, preds_prob, final_feats, f_cols

@st.cache_data
def get_shap_explanations(model_path, final_feats):
    """Computes TreeExplainer SHAP values for GraphAware XGBoost model."""
    import shap
    model = xgb.Booster()
    model.load_model(model_path)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(final_feats)
    base_val = float(explainer.expected_value) if hasattr(explainer, 'expected_value') and np.isscalar(explainer.expected_value) else 0.0
    return shap_vals, base_val

# -----------------------------------------------------------------------------
# 4. UI HEADER & PIPELINE DIGEST
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🩺 Sepsis Risk Prediction & GraphFlow Inference Module</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive clinical decision support using interpretable graph neural aggregations (GraphFlow) and XGBoost</div>', unsafe_allow_html=True)

with st.expander("📚 Digest of Sepsis Evaluation Pipeline (GraphFlow Architecture)"):
    st.markdown(r"""
    ### Pipeline Architecture Overview
    
    1. **`0_mimic_preprocess` (MIMIC-IV Cohort Preprocessing)**
       - Extracts patient lab measurements and ICU trajectories from MIMIC-IV database.
       - Maps panel items (CBC, BMP, LIVER, COAG, KIDNEY, HIL) to MIMIC item IDs via `panel_name_to_feature_codes.py`.
       - Standardizes laboratory codes and timestamps via R-based containerized scripts.
       
    2. **`1_preprocess` (Data Cleaning & Clinical Labeling)**
       - Filters unique patient episodes and non-ICU ward observations leading up to MICU admission.
       - Defines **Sepsis** label (\(y=1\)) if Sepsis occurs within 6 hours before MICU admission (`SecToIcu` \(\le 21600\)).
       - Defines **Control** label (\(y=0\)) for non-septic ward patients or long lead times (`SecToIcu` \(> 21600\)).
       - Supports imputation strategies (`drop`, `zero_fill`, `mean_fill`, `multi_imputer`).
       - Formats laboratory and demographic features with prefix `f__` (e.g., `f__WBC`, `f__HGB`, `f__Creatinine`, `f__Age`, `f__Sex`).

    3. **`2_baseline` (Standard Tabular Machine Learning)**
       - Benchmarks standard classifiers (Logistic Regression & Random Forest) on isolated tabular features \(f\_*\) without graph structure.
       
    4. **`3_graph_construction` (Patient-Centric Graph Building)**
       - Constructs directed graphs per patient where nodes represent laboratory measurement timestamps sorted by `Id` (Subject ID) and `Time`.
       - Connects past measurements to current measurements with temporal edge weights:
         $$w_{ij} = 1 - \frac{t_j - t_i}{t_{max} - t_{min}}$$
         giving higher edge weight (\(\approx 1.0\)) to recent lab tests.

    5. **`4_db_upload` (Graph Database Ingestion)**
       - Converts node features, directed edges, and temporal weights into SQLite/Neo4j database tables per feature panel name.

    6. **`5_gnn_training` & `6_graphaware` (Graph Models & Interpretable Inference)**
       - Deep GNNs (Graph Attention Networks) trained with graph mini-batching.
       - **GraphFlow Framework**: Computes 1-hop spatial neighborhood feature differences:
         $$\text{Features}_{GraphFlow} = \Big[ \mathbf{X}_{orig},\quad \mathbf{X}_{orig} - \boldsymbol{\mu}_{neighbors} \Big]$$
       - Trains **XGBoost** on GraphFlow representations, optimized via Hyperopt, yielding fast, interpretable, high-AUROC clinical predictions.
    """)

st.divider()

# -----------------------------------------------------------------------------
# 5. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Model & Data Configuration")

# Model Selection
available_models = get_available_trained_models()
if not available_models:
    st.sidebar.error("No trained GraphFlow XGBoost models found in 'trained_models' or '6_graphaware/models'.")
    st.stop()

selected_panel = st.sidebar.selectbox(
    "1. Select Trained Panel Model",
    options=list(available_models.keys()),
    index=0,
    help="Choose the trained XGBoost model corresponding to the clinical panel."
)
selected_model_path = available_models[selected_panel]
st.sidebar.caption(f"📁 Model Path: `{os.path.relpath(selected_model_path, REPO_ROOT)}`")

st.sidebar.divider()

# Data Source Selection
st.sidebar.subheader("2. Data Source")
data_option = st.sidebar.radio(
    "Choose Input Method",
    options=["Upload Test Data File (CSV / Excel)", "Load Built-in Sample Test Dataset"],
    index=1
)

df_loaded = None
selected_sample_key = None

if data_option == "Upload Test Data File (CSV / Excel)":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Test Data (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        help="Upload patient laboratory test data containing 'f__' feature columns or raw lab columns."
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_loaded = pd.read_csv(uploaded_file)
            else:
                df_loaded = pd.read_excel(uploaded_file)
            selected_sample_key = uploaded_file.name
            st.sidebar.success(f"Successfully loaded {len(df_loaded):,} rows from `{uploaded_file.name}`.")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
else:
    sample_datasets = get_sample_datasets(selected_panel)
    if sample_datasets:
        clean_panel = selected_panel.replace("MIMIC_", "").replace("SBC_", "")
        selected_sample_key = st.sidebar.selectbox(
            "Select Sample Test Dataset",
            options=list(sample_datasets.keys()),
            index=0,
            help=f"Filtered to sample test/train/val datasets for panel '{clean_panel}'"
        )
        sample_path = sample_datasets[selected_sample_key]
        if os.path.exists(sample_path):
            df_loaded = pd.read_csv(sample_path)
            st.sidebar.success(f"Loaded dataset `{selected_sample_key}` ({len(df_loaded):,} rows).")

# Row subsampling option (with checkbox to disable limit)
eval_all_rows = st.sidebar.checkbox(
    "Evaluate All Rows (Full Dataset)",
    value=True,
    help="When enabled, processes the entire dataset. Filtering and sorting will apply across all rows, while UI tables preview the top 200 rows."
)

if not eval_all_rows and df_loaded is not None and len(df_loaded) > 500:
    max_rows = st.sidebar.slider(
        "Limit Evaluation Rows (For Fast Exploration)",
        min_value=100,
        max_value=min(len(df_loaded), 50000),
        value=min(len(df_loaded), 5000),
        step=500,
        help="Uncheck 'Evaluate All Rows' to adjust max row count."
    )
    df_loaded = df_loaded.iloc[:max_rows].copy()

# Look up optimal cutoff pre-calculated for panel and dataset
optimal_cutoff_val = get_default_cutoff(selected_panel, selected_sample_key)

st.sidebar.divider()
st.sidebar.subheader("3. Decision Cutoff Threshold")
st.sidebar.markdown(f'<div class="opt-badge">🎯 Optimal F₂ Cutoff (β=2): <b>{optimal_cutoff_val:.4f}</b></div>', unsafe_allow_html=True)

risk_threshold = st.sidebar.number_input(
    "Decision Cutoff Threshold",
    min_value=0.0001,
    max_value=0.9999,
    value=float(optimal_cutoff_val),
    step=0.0001,
    format="%.4f",
    help="Defaults to the optimal validation-calibrated F₂ cutoff (β=2). Adjust this value with 4-decimal precision (e.g., 0.0306) to test custom sensitivity/precision trade-offs."
)

st.sidebar.divider()
run_inference_sidebar = st.sidebar.button(
    "🚀 Run GraphFlow Inference",
    type="primary",
    use_container_width=True,
    key="sidebar_run_inference_btn",
    help="Click here to execute GraphFlow 1-hop spatial neighborhood feature aggregation and XGBoost inference."
)

# -----------------------------------------------------------------------------
# 6. DATA INSPECTION & FILTERING
# -----------------------------------------------------------------------------
if df_loaded is None:
    st.info("👈 Please select a sample dataset or upload a test data file in the sidebar to begin.")
    st.stop()

st.subheader("📋 1. Uploaded Test Data Overview & Filtering")

# Data Filtering Controls
with st.expander("🔍 Server-Side Filter & Sort Controls", expanded=True):
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    selected_patients = []
    patient_id_text = ""
    selected_age_range = None
    age_col = None
    selected_gt_filter = []
    
    gt_col = "y" if "y" in df_loaded.columns else ("Label" if "Label" in df_loaded.columns else None)
    
    with filter_col1:
        if "Id" in df_loaded.columns:
            all_patients = sorted(df_loaded["Id"].unique().tolist())
            selected_patients = st.multiselect("Filter by Patient ID (Id)", options=all_patients, help="Search or select patient IDs from complete dataset")
            patient_id_text = st.text_input("Search Patient ID (text)", value="", help="Type any patient ID to search instantly across full dataset")
                
    with filter_col2:
        if "f__Age" in df_loaded.columns or "Age" in df_loaded.columns:
            age_col = "f__Age" if "f__Age" in df_loaded.columns else "Age"
            min_age, max_age = float(df_loaded[age_col].min()), float(df_loaded[age_col].max())
            if min_age < max_age:
                selected_age_range = st.slider("Filter by Age", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    with filter_col3:
        if gt_col:
            selected_gt_filter = st.multiselect("Filter Ground-Truth Label", options=sorted(df_loaded[gt_col].astype(str).unique()))

    # Build preview column list
    preview_cols = []
    for meta_col in ["Id", "Time"]:
        if meta_col in df_loaded.columns:
            preview_cols.append(meta_col)
    if gt_col and gt_col in df_loaded.columns and gt_col not in preview_cols:
        preview_cols.append(gt_col)
        
    f_cols_preview = [c for c in df_loaded.columns if c.startswith("f_")]
    preview_cols.extend(f_cols_preview)
    
    for raw_meta in ["Age", "Sex"]:
        if raw_meta in df_loaded.columns and raw_meta not in preview_cols and f"f__{raw_meta}" not in preview_cols:
            preview_cols.append(raw_meta)

    with filter_col4:
        sort_column = st.selectbox("Sort Overview by Column", options=preview_cols if preview_cols else list(df_loaded.columns), index=0)
        sort_order = st.radio("Sort Direction", options=["Ascending", "Descending"], index=0, horizontal=True)

    # Server-Side Filter & Sort Engine Call
    filtered_preview_full, preview_overview_df, total_filtered_overview, total_dataset_overview = filter_and_sort_server_side(
        df=df_loaded,
        selected_patients=selected_patients,
        patient_id_query=patient_id_text,
        age_range=selected_age_range,
        age_col=age_col,
        gt_filter=selected_gt_filter,
        gt_col=gt_col,
        sort_column=sort_column,
        sort_order=sort_order,
        display_limit=200,
        page_number=1
    )

    # Render Server-Side Sliced Dataframe (Top 200 rows)
    cols_to_show = [c for c in preview_cols if c in preview_overview_df.columns]
    st.dataframe(preview_overview_df[cols_to_show], use_container_width=True, height=250)
    st.caption(
        f"🖥️ **Server-Side Engine:** Full dataset filtering & sorting computed on Python server across {total_dataset_overview:,} rows. "
        f"Displaying top {len(preview_overview_df):,} preview rows out of {total_filtered_overview:,} matching observations."
    )

# Metric cards (based on server-side filtered preview set)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Filtered Test Observations", f"{total_filtered_overview:,} / {total_dataset_overview:,}")
with col2:
    num_pts = filtered_preview_full["Id"].nunique() if "Id" in filtered_preview_full.columns else "N/A"
    st.metric("Unique Patient IDs", f"{num_pts:,}" if isinstance(num_pts, int) else num_pts)
with col3:
    f_cols_found = [c for c in filtered_preview_full.columns if c.startswith("f__")]
    st.metric("Feature Columns ('f__')", len(f_cols_found))
with col4:
    if gt_col and len(filtered_preview_full) > 0:
        pos_count = (filtered_preview_full[gt_col].astype(str).str.contains("Sepsis|1")).sum()
        st.metric("Ground-Truth Sepsis Cases", f"{pos_count:,} ({pos_count/len(filtered_preview_full):.1%})")
    else:
        st.metric("Ground-Truth Available", "No (Unlabeled)")

st.divider()

# -----------------------------------------------------------------------------
# 7. GRAPHFLOW INFERENCE & PREDICTION PROBABILITIES
# -----------------------------------------------------------------------------
st.subheader("⚡ 2. GraphFlow Sepsis Prediction Probabilities")

run_inference_main = st.button("🚀 Compute GraphFlow Prediction Probabilities", type="primary")

if run_inference_main or run_inference_sidebar:
    with st.spinner(f"Computing 1-hop GraphFlow aggregations & outputting prediction probabilities for '{selected_panel}'..."):
        start_t = time.time()
        res_df, preds_prob, final_feats, f_cols = run_graphaware_inference(df_loaded, selected_model_path, selected_panel)
        elapsed_t = time.time() - start_t
        
    if res_df is not None and preds_prob is not None:
        st.session_state["inference_completed"] = True
        st.session_state["res_df"] = res_df
        st.session_state["preds_prob"] = preds_prob
        st.session_state["final_feats"] = final_feats
        st.session_state["f_cols"] = f_cols
        st.session_state["elapsed_t"] = elapsed_t
        st.success(f"GraphFlow prediction probabilities calculated in **{elapsed_t:.2f} seconds** across {len(preds_prob):,} rows!")

if st.session_state.get("inference_completed", False):
    res_df = st.session_state["res_df"].copy()
    preds_prob = st.session_state["preds_prob"]
    final_feats = st.session_state.get("final_feats", None)
    f_cols = st.session_state.get("f_cols", None)
    
    # Append prediction probabilities and cutoff-calibrated risk labels to dataframe
    # Calibrated Sepsis Risk (%): at raw prob == risk_threshold, risk is exactly 50%.
    # Above cutoff (Sepsis): risk increases continuously from 50% to 100%.
    # Below cutoff (Control): risk decreases continuously from 50% down to 0%.
    c_safe = max(1e-6, min(1.0 - 1e-6, float(risk_threshold)))
    calibrated_risk_pct = np.where(
        preds_prob < c_safe,
        50.0 * (preds_prob / c_safe),
        50.0 + 50.0 * ((preds_prob - c_safe) / (1.0 - c_safe))
    )
    
    res_df["Sepsis_Prediction_Probability"] = preds_prob.round(6)
    res_df["Sepsis_Risk_%"] = np.clip(calibrated_risk_pct, 0.0, 100.0).round(2)
    res_df["Predicted_Class"] = np.where(preds_prob >= risk_threshold, "Sepsis", "Control")
    
    def assign_risk_category(p):
        if p < risk_threshold:
            return f"Low Risk (<{risk_threshold:.4f})"
        else:
            return f"High Risk (≥{risk_threshold:.4f})"
            
    res_df["Risk_Category"] = res_df["Sepsis_Prediction_Probability"].apply(assign_risk_category)
    
    # Server-side filter & sort on prediction probabilities dataframe res_df
    display_cols = ["Sepsis_Prediction_Probability", "Sepsis_Risk_%", "Predicted_Class", "Risk_Category"]
    if "Id" in res_df.columns: display_cols.insert(0, "Id")
    if "Time" in res_df.columns: display_cols.insert(1, "Time")
    if gt_col and gt_col in res_df.columns: display_cols.append(gt_col)
    
    f_labs = [c for c in res_df.columns if c.startswith("f_")]
    display_cols.extend(f_labs)
    
    psort_col1, psort_col2, psort_col3 = st.columns([2, 1, 1])
    with psort_col1:
        pred_sort_column = st.selectbox(
            "Sort Prediction Results by Column",
            options=display_cols,
            index=display_cols.index("Sepsis_Prediction_Probability") if "Sepsis_Prediction_Probability" in display_cols else 0,
            key="pred_sort_col_select"
        )
    with psort_col2:
        pred_sort_order = st.radio(
            "Prediction Sort Order",
            options=["Descending", "Ascending"],
            index=0,
            horizontal=True,
            key="pred_sort_order_radio"
        )
    with psort_col3:
        total_pred_pages = max(1, math.ceil(len(res_df) / 200))
        pred_page = st.number_input(
            "Page (200 rows/page)",
            min_value=1,
            max_value=total_pred_pages,
            value=1,
            step=1,
            key="pred_page_input"
        )

    filtered_res_full, preview_res_df, total_filtered_res, total_res = filter_and_sort_server_side(
        df=res_df,
        selected_patients=selected_patients,
        patient_id_query=patient_id_text,
        age_range=selected_age_range,
        age_col=age_col,
        gt_filter=selected_gt_filter,
        gt_col=gt_col,
        sort_column=pred_sort_column,
        sort_order=pred_sort_order,
        display_limit=200,
        page_number=pred_page
    )

    if total_filtered_res == 0:
        st.warning("⚠️ No observations match the active filter criteria. Please adjust filters above.")
    else:
        # Summary Risk Distribution Cards (evaluated on full server-side filtered set)
        filtered_preds_prob_full = filtered_res_full["Sepsis_Prediction_Probability"].to_numpy()
        rc1, rc2, rc3, rc4 = st.columns(4)
        high_risk_cnt = (filtered_preds_prob_full >= risk_threshold).sum()
        low_risk_cnt = (filtered_preds_prob_full < risk_threshold).sum()
        
        with rc1:
            st.metric("Mean Prediction Probability", f"{filtered_preds_prob_full.mean():.4f} ({filtered_preds_prob_full.mean():.1%})")
        with rc2:
            st.metric(f"High Probability (≥{risk_threshold:.4f})", f"{high_risk_cnt:,} ({high_risk_cnt/len(filtered_preds_prob_full):.1%})")
        with rc3:
            st.metric(f"Low Probability (<{risk_threshold:.4f})", f"{low_risk_cnt:,} ({low_risk_cnt/len(filtered_preds_prob_full):.1%})")
        with rc4:
            st.metric("Active Cutoff Threshold", f"{risk_threshold:.4f}")

        # Display Interactive Prediction Probabilities Table
        st.markdown(f"#### 📊 Patient Prediction Probabilities Table ({total_filtered_res:,} Filtered Observations)")
        st.caption(
            f"❓ **Cutoff-Calibrated Sepsis Risk (%):** At the active cutoff threshold (**{risk_threshold:.4f}**), risk is calibrated to exactly **50.0%**. "
            f"For predictions above the cutoff (Sepsis), risk continuously increases up to **100%**. "
            f"For predictions below the cutoff (Control), risk continuously decreases down to **0%**."
        )

        st.info("👇 **Click directly on any patient row in the table below** to select and calculate its **Local SHAP Values**!")

        pred_cols_to_show = [c for c in display_cols if c in preview_res_df.columns]

        selection_event = st.dataframe(
            preview_res_df[pred_cols_to_show],
            use_container_width=True,
            height=340,
            on_select="rerun",
            selection_mode="single-row",
            key="patient_pred_table_select",
            column_config={
                "Sepsis_Prediction_Probability": st.column_config.NumberColumn(
                    "Prediction Probability P(Sepsis)",
                    format="%.4f",
                    help="Raw probability score output by GraphFlow XGBoost model (range 0.0 to 1.0)."
                ),
                "Sepsis_Risk_%": st.column_config.ProgressColumn(
                    "Sepsis Risk (%)",
                    format="%.2f%%",
                    min_value=0,
                    max_value=100,
                    help=(
                        f"Cutoff-Calibrated Sepsis Risk (%):\n"
                        f"- At the active cutoff threshold (cutoff = {risk_threshold:.4f}), risk is calibrated to exactly 50.0%.\n"
                        f"- Above the cutoff (classified as Sepsis), risk increases continuously from 50% to 100%.\n"
                        f"- Below the cutoff (classified as Control), risk decreases continuously from 50% down to 0%."
                    )
                ),
            }
        )

        start_row_idx = max(1, (pred_page - 1) * 200 + 1)
        end_row_idx = min(pred_page * 200, total_filtered_res)
        st.caption(
            f"🖥️ **Server-Side Engine:** Full filtering & sorting computed on Python backend across {total_res:,} total prediction observations. "
            f"Displaying page {pred_page} (rows {start_row_idx:,}-{end_row_idx:,} of {total_filtered_res:,} matching observations)."
        )

        clicked_row_idx = 0
        if selection_event and hasattr(selection_event, "selection") and selection_event.selection and "rows" in selection_event.selection:
            sel_rows = selection_event.selection.rows
            if len(sel_rows) > 0:
                clicked_row_idx = sel_rows[0]

        # Download Button (Full Filtered set)
        csv_bytes = filtered_res_full.drop(columns=["_row_pos"], errors="ignore").to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download Prediction Probabilities CSV ({len(filtered_res_full):,} rows)",
            data=csv_bytes,
            file_name="graphflow_sepsis_prediction_probabilities_filtered.csv",
            mime="text/csv",
            type="secondary"
        )

        # ---------------------------------------------------------------------
        # 7.1 LOCAL SHAP EXPLANATION & GRAPHFLOW FEATURE ATTRIBUTION
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🔬 Local SHAP Explanation & GraphFlow Feature Attribution")
        st.markdown(
            "Select any patient observation row directly from the table above or the dropdown below to view its **Local SHAP Values**, "
            "aggregated from original clinical features ($\\mathbf{X}_{orig}$) and GraphFlow time-based temporal differences ($\\mathbf{X}_{orig} - \\boldsymbol{\\mu}_{neighbors}$)."
        )

        if final_feats is not None and f_cols is not None:
            # Build dropdown labels for preview_res_df observations
            row_options = []
            for idx_i, row_item in preview_res_df.iterrows():
                label_parts = [f"Row #{idx_i + 1}"]
                if "Id" in row_item: label_parts.append(f"Patient ID: {row_item['Id']}")
                if "Time" in row_item: label_parts.append(f"Time: {row_item['Time']}")
                label_parts.append(f"Risk: {row_item['Sepsis_Risk_%']:.2f}% ({row_item['Predicted_Class']})")
                row_options.append(" | ".join(label_parts))

            selected_row_idx = st.selectbox(
                "Selected Patient Observation Row for SHAP Breakdown:",
                options=list(range(len(preview_res_df))),
                index=clicked_row_idx if clicked_row_idx < len(preview_res_df) else 0,
                format_func=lambda i: row_options[i],
                help="Click any row in the table above or pick from this dropdown to calculate its GraphFlow SHAP feature contributions."
            )

            if selected_row_idx is not None and len(preview_res_df) > 0:
                actual_df_idx = int(preview_res_df.iloc[selected_row_idx]["_row_pos"])
                
                with st.spinner(f"Computing local SHAP attributions for Row #{selected_row_idx + 1} via TreeExplainer..."):
                    shap_vals, base_val = get_shap_explanations(selected_model_path, final_feats)
                
                row_res = res_df.iloc[actual_df_idx]
                row_final_feats = final_feats[actual_df_idx]
                row_shap = shap_vals[actual_df_idx]
                
                num_base_f = len(f_cols)
                orig_vals = row_final_feats[:num_base_f]
                diff_vals = row_final_feats[num_base_f:]
                shap_orig = row_shap[:num_base_f]
                shap_diff = row_shap[num_base_f:]
                shap_total = shap_orig + shap_diff
                clean_names = [c.replace("f__", "") for c in f_cols]
                
                # Highlight Metrics Cards for Selected Observation
                sh_col1, sh_col2, sh_col3, sh_col4, sh_col5 = st.columns(5)
                with sh_col1:
                    st.metric("Selected Patient ID", str(row_res.get("Id", "N/A")))
                with sh_col2:
                    st.metric("Timestamp (Time)", str(row_res.get("Time", "N/A")))
                with sh_col3:
                    st.metric("Predicted Class", str(row_res.get("Predicted_Class", "N/A")))
                with sh_col4:
                    st.metric("Raw P(Sepsis)", f"{row_res.get('Sepsis_Prediction_Probability', 0.0):.4f}")
                with sh_col5:
                    st.metric("Calibrated Risk (%)", f"{row_res.get('Sepsis_Risk_%', 0.0):.2f}%")

                # Build Local SHAP DataFrame
                local_shap_df = pd.DataFrame({
                    "Feature": clean_names,
                    "Original Value": orig_vals.round(4),
                    "Original SHAP": shap_orig.round(5),
                    "Time-Based Δ Mean": diff_vals.round(4),
                    "Time-Based SHAP": shap_diff.round(5),
                    "Total Aggregated SHAP": shap_total.round(5),
                })
                local_shap_df["Abs_Total_SHAP"] = local_shap_df["Total Aggregated SHAP"].abs()
                local_shap_df_sorted = local_shap_df.sort_values(by="Abs_Total_SHAP", ascending=True).reset_index(drop=True)
                
                # Plotly Visualizations: 2 side-by-side columns
                shap_plot_col1, shap_plot_col2 = st.columns([1.1, 1.0])
                
                with shap_plot_col1:
                    # 1. Total Aggregated Local SHAP Bar Chart
                    fig_total_shap = go.Figure()
                    bar_colors = np.where(local_shap_df_sorted["Total Aggregated SHAP"] >= 0, "#EF4444", "#3B82F6")
                    
                    fig_total_shap.add_trace(go.Bar(
                        y=local_shap_df_sorted["Feature"],
                        x=local_shap_df_sorted["Total Aggregated SHAP"],
                        orientation='h',
                        marker=dict(color=bar_colors),
                        hoverinfo="text",
                        hovertext=[
                            f"<b>{r['Feature']}</b><br>"
                            f"Original Value: {r['Original Value']:.4f} (SHAP: {r['Original SHAP']:+.4f})<br>"
                            f"Time-Based Δ Mean: {r['Time-Based Δ Mean']:.4f} (SHAP: {r['Time-Based SHAP']:+.4f})<br>"
                            f"<b>Total Aggregated SHAP: {r['Total Aggregated SHAP']:+.4f}</b>"
                            for _, r in local_shap_df_sorted.iterrows()
                        ]
                    ))
                    st.markdown(
                        f'<div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 8px;">'
                        f'📊 Total Aggregated Local SHAP (Row #{selected_row_idx + 1})'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    fig_total_shap.update_layout(
                        xaxis_title="Local SHAP Impact (+ Increases Risk, - Decreases Risk)",
                        yaxis_title="Clinical Lab Feature",
                        height=max(360, len(f_cols) * 35),
                        margin=dict(l=40, r=40, t=20, b=40)
                    )
                    st.plotly_chart(fig_total_shap, use_container_width=True)

                with shap_plot_col2:
                    # 2. GraphFlow Attribution Breakdown (Original vs Time-Based SHAP)
                    fig_breakdown = go.Figure()
                    fig_breakdown.add_trace(go.Bar(
                        y=local_shap_df_sorted["Feature"],
                        x=local_shap_df_sorted["Original SHAP"],
                        name="Original Feature SHAP",
                        orientation='h',
                        marker=dict(color='#3B82F6')
                    ))
                    fig_breakdown.add_trace(go.Bar(
                        y=local_shap_df_sorted["Feature"],
                        x=local_shap_df_sorted["Time-Based SHAP"],
                        name="GraphFlow Time-Based Δ Mean SHAP",
                        orientation='h',
                        marker=dict(color='#10B981')
                    ))
                    st.markdown(
                        f'<div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 8px;">'
                        f'🧩 GraphFlow Attribution Breakdown (Original vs. Δ Mean)'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    fig_breakdown.update_layout(
                        barmode='relative',
                        xaxis_title="SHAP Contribution",
                        yaxis_title="Clinical Lab Feature",
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.18,
                            xanchor="center",
                            x=0.5,
                            bgcolor="rgba(0,0,0,0)"
                        ),
                        height=max(380, len(f_cols) * 35),
                        margin=dict(l=40, r=40, t=20, b=75)
                    )
                    st.plotly_chart(fig_breakdown, use_container_width=True)

                # 3. Detailed Feature Attribution Table
                st.markdown("#### 📋 Detailed Feature Attribution Breakdown Table")
                table_display = local_shap_df.sort_values(by="Abs_Total_SHAP", ascending=False).drop(columns=["Abs_Total_SHAP"]).reset_index(drop=True)
                table_display["Risk Impact"] = np.where(
                    table_display["Total Aggregated SHAP"] > 0,
                    "↗ Increases Sepsis Risk",
                    "↘ Decreases Sepsis Risk"
                )
                st.dataframe(table_display, use_container_width=True, hide_index=True)

        st.divider()

        # -------------------------------------------------------------------------
        # 8. GROUND-TRUTH EVALUATION & RESPONSIVE CONFUSION MATRIX
        # -------------------------------------------------------------------------
        st.subheader("📈 3. Ground-Truth Performance & AUROC Evaluation (Filtered Set)")
        
        if gt_col and gt_col in filtered_res_full.columns:
            y_true_binary = (filtered_res_full[gt_col].astype(str).str.contains("Sepsis|1")).astype(int)
            filtered_preds_prob = filtered_res_full["Sepsis_Prediction_Probability"].to_numpy()
            
            if len(np.unique(y_true_binary)) > 1:
                auroc_score = roc_auc_score(y_true_binary, filtered_preds_prob)
                fpr, tpr, thresholds = roc_curve(y_true_binary, filtered_preds_prob)
                
                # Binary Metrics at active Cutoff
                y_pred_binary = (filtered_preds_prob >= risk_threshold).astype(int)
                acc = accuracy_score(y_true_binary, y_pred_binary)
                sens = recall_score(y_true_binary, y_pred_binary, zero_division=0)
                prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
                f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
                f2 = fbeta_score(y_true_binary, y_pred_binary, beta=2, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
                g_mean = np.sqrt(sens * spec)

                # Metric Cards
                auc_c1, auc_c2, auc_c3, auc_c4, auc_c5, auc_c6 = st.columns(6)
                with auc_c1:
                    st.metric("🏆 AUROC Score", f"{auroc_score:.4f}")
                with auc_c2:
                    st.metric("Sensitivity (Recall)", f"{sens:.1%}")
                with auc_c3:
                    st.metric("Specificity", f"{spec:.1%}")
                with auc_c4:
                    st.metric("PPV (Precision)", f"{prec:.1%}")
                with auc_c5:
                    st.metric("F₂ Score (β=2)", f"{f2:.4f}")
                with auc_c6:
                    st.metric("Geometric Mean (G-Mean)", f"{g_mean:.4f}")

                # Plot ROC Curve & Enhanced Confusion Matrix side by side
                plot_col1, plot_col2 = st.columns([1.2, 1.0])
                
                with plot_col1:
                    fig_roc = go.Figure()
                    
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'GraphFlow XGBoost (AUC = {auroc_score:.4f})',
                        line=dict(color='#1E3A8A', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(30, 58, 138, 0.1)'
                    ))
                    
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random Baseline (AUC = 0.50)',
                        line=dict(color='#EF4444', width=2, dash='dash')
                    ))
                    
                    st.markdown(
                        f'<div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 8px;">'
                        f'📈 ROC Curve - GraphFlow ({selected_panel}) [{len(filtered_res_full):,} Filtered Obs]'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    fig_roc.update_layout(
                        xaxis_title="False Positive Rate (1 - Specificity)",
                        yaxis_title="True Positive Rate (Sensitivity)",
                        xaxis=dict(range=[-0.01, 1.01]),
                        yaxis=dict(range=[-0.01, 1.01]),
                        legend=dict(x=0.42, y=0.15, bgcolor='rgba(0,0,0,0)'),
                        margin=dict(l=30, r=30, t=15, b=30),
                        height=380
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

                with plot_col2:
                    # Improved Annotated Confusion Matrix
                    total_obs = len(y_true_binary)
                    cm_text = [
                        [f"<b>TN</b>: {tn:,}<br>({tn/total_obs:.1%})", f"<b>FP</b>: {fp:,}<br>({fp/total_obs:.1%})"],
                        [f"<b>FN</b>: {fn:,}<br>({fn/total_obs:.1%})", f"<b>TP</b>: {tp:,}<br>({tp/total_obs:.1%})"]
                    ]
                    
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=[[tn, fp], [fn, tp]],
                        x=['Predicted Control (0)', 'Predicted Sepsis (1)'],
                        y=['Actual Control (0)', 'Actual Sepsis (1)'],
                        text=cm_text,
                        texttemplate="%{text}",
                        textfont={"size": 14},
                        colorscale="Blues",
                        showscale=False
                    ))
                    
                    st.markdown(
                        f'<div style="font-size: 16px; font-weight: 700; color: #0F172A; margin-bottom: 8px;">'
                        f'🎯 Confusion Matrix (Cutoff = {risk_threshold:.4f}) [{total_obs:,} Obs]'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    fig_cm.update_layout(
                        xaxis_title="Predicted Class",
                        yaxis_title="Actual Ground-Truth Label",
                        margin=dict(l=40, r=40, t=20, b=40),
                        height=350
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Clinical Metrics Table Below Confusion Matrix
                    cm_stats_df = pd.DataFrame({
                        "Clinical Metric": ["True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)", "True Positives (TP)", "Negative Predictive Value (NPV)", "Positive Predictive Value (PPV)"],
                        "Value": [f"{tn:,}", f"{fp:,}", f"{fn:,}", f"{tp:,}", f"{npv:.1%}", f"{prec:.1%}"]
                    })
                    st.dataframe(cm_stats_df, use_container_width=True, hide_index=True)

            else:
                st.warning("Ground-truth column contains only a single unique class in the selected dataset slice. AUROC requires both Control and Sepsis cases.")
        else:
            st.info("No ground-truth label column ('y', 'Label', 'Diagnosis') found in uploaded dataset. Ground-truth AUROC evaluation skipped.")

st.sidebar.markdown("---")
st.sidebar.caption("Sepsis Evaluation Pipeline | GraphFlow")
