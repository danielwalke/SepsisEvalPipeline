import mlflow
import pandas as pd
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")

experiment_name = "evaluations_CBC"
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

runs_df = mlflow.search_runs(experiment_ids=[experiment_id])

model_name_mapping = {
    "LogisticRegression_Baseline": "Baseline Logistic Regression",
    "RandomForestClassifier_Baseline": "Baseline Random Forest",
    "XGBClassifier_Baseline": "Baseline XGBoost",
    "GNN": "Graph Neural Network (GATv2)",
    "GraphAwareXGBoost": "GraphAware (XGBoost)"
}

datasets = ["SBC", "MIMIC"]
model_keys = [
    "LogisticRegression_Baseline", 
    "RandomForestClassifier_Baseline", 
    "XGBClassifier_Baseline", 
    "GNN", 
    "GraphAwareXGBoost"
]

results_dict = {ds: {mk: {} for mk in model_keys} for ds in datasets}

def get_val(row, key, default_val=np.nan):
    if key in row and not pd.isna(row[key]):
        return row[key]
    if f"metrics.{key}" in row and not pd.isna(row[f"metrics.{key}"]):
        return row[f"metrics.{key}"]
    return default_val

def get_name(row):
    if "tags.mlflow.runName" in row and pd.notna(row["tags.mlflow.runName"]):
        return str(row["tags.mlflow.runName"])
    if "Run Name" in row and pd.notna(row["Run Name"]):
        return str(row["Run Name"])
    return ""

for index, row in runs_df.iterrows():
    run_name = get_name(row)
    
    training_data = "Unknown"
    model_key = "Unknown"
    
    if run_name.endswith("_MIMIC"):
        training_data = "MIMIC"
        model_key = run_name.replace("_MIMIC", "")
    elif run_name.endswith("_SBC"):
        training_data = "SBC"
        model_key = run_name.replace("_SBC", "")
        
    if training_data in datasets and model_key in model_keys:
        results_dict[training_data][model_key] = {
            "time": get_val(row, "training_time_seconds"),
            "sbc_test": get_val(row, "SBC_TEST__AUROC"),
            "sbc_ext_test": get_val(row, "SBC_EXT_TEST__AUROC"),
            "mimic_test": get_val(row, "MIMIC_TEST__AUROC")
        }

def format_metric(value, is_time=False):
    if pd.isna(value) or value == "-":
        return "-"
    try:
        float_val = float(value)
        if is_time:
            return f"{float_val:.2f}s"
        return f"{float_val:.3f}"
    except ValueError:
        return "-"

latex_code = []
latex_code.append(r"\begin{table}[htbp]")
latex_code.append(r"    \centering")
latex_code.append(r"    \caption{AUROC performance and required training time of models across different test sets.}")
latex_code.append(r"    \label{tab:auroc_results}")
latex_code.append(r"    \small")
latex_code.append(r"    \begin{tabular}{llcccc}")
latex_code.append(r"        \toprule")
latex_code.append(r"        & & & \multicolumn{3}{c}{\textbf{AUROC}} \\")
latex_code.append(r"        \cmidrule(lr){4-6}")
latex_code.append(r"        \textbf{Training Data} & \textbf{Model} & \thead{Training\\Time} & \thead{SBC\\Test} & \thead{SBC\_Ext\\Test} & \thead{MIMIC\\Test} \\")
latex_code.append(r"        \midrule")

for ds_idx, ds in enumerate(datasets):
    for m_idx, mk in enumerate(model_keys):
        model_display_name = model_name_mapping[mk]
        run_data = results_dict[ds][mk]
        
        time_val = format_metric(run_data.get("time", np.nan), is_time=True)
        sbc_val = format_metric(run_data.get("sbc_test", np.nan))
        sbc_ext_val = format_metric(run_data.get("sbc_ext_test", np.nan))
        mimic_val = format_metric(run_data.get("mimic_test", np.nan))
        
        if m_idx == 0:
            dataset_col = f"\\multirow{{5}}{{*}}{{\\textbf{{{ds}}}}}"
            row_str = f"        {dataset_col:<29} & {model_display_name:<30} & {time_val:<8} & {sbc_val:<8} & {sbc_ext_val:<8} & {mimic_val:<8} \\\\"
        else:
            indent_space = " " * 37
            row_str = f"{indent_space}& {model_display_name:<30} & {time_val:<8} & {sbc_val:<8} & {sbc_ext_val:<8} & {mimic_val:<8} \\\\"
            
        latex_code.append(row_str)
        
    if ds_idx == 0:
        latex_code.append(r"        \midrule")

latex_code.append(r"        \bottomrule")
latex_code.append(r"    \end{tabular}")
latex_code.append(r"\end{table}")

print("\n".join(latex_code))