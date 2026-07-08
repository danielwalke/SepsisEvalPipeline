import mlflow
import pandas as pd
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")

def get_latest_runs():
    experiments = mlflow.search_experiments()
    exp_id_to_name = {exp.experiment_id: exp.name for exp in experiments}
    
    df = mlflow.search_runs(search_all_experiments=True)
    df['experiment_name'] = df['experiment_id'].map(exp_id_to_name)
    
    if 'tags.mlflow.runName' in df.columns:
        df = df[df['tags.mlflow.runName'].str.contains('MIMIC', na=False)].copy()
    
    auroc_col = 'metrics.MIMIC_TEST__AUROC'
    if auroc_col not in df.columns:
        df[auroc_col] = np.nan
            
    df = df.sort_values(by='start_time', ascending=False)
    df = df.drop_duplicates(subset=['experiment_name', 'tags.mlflow.runName'], keep='first')
    
    return df, auroc_col

def generate_latex_transposed(df, auroc_col):
    raw_experiments = df['experiment_name'].dropna().unique()
    
    desired_order = [
        "evaluations_CBC",
        "evaluations_CBC_BMP",
        "evaluations_CBC_HIL",
        "evaluations_CBC_BMP_HIL"
    ]
    
    sorted_exp_keys = [exp for exp in desired_order if exp in raw_experiments]
    
    exp_mapping = {
        "evaluations_CBC": "CBC",
        "evaluations_CBC_BMP": "CBC+BMP",
        "evaluations_CBC_HIL": "CBC+HIL",
        "evaluations_CBC_BMP_HIL": "CBC+BMP+HIL"
    }
    
    model_mapping = {
        "LogisticRegression_Baseline_MIMIC": "Baseline Logistic Regression",
        "RandomForestClassifier_Baseline_MIMIC": "Baseline Random Forest",
        "XGBClassifier_Baseline_MIMIC": "Baseline XGBoost",
        "GNN_MIMIC": "Graph Neural Network (GATv2)",
        "GraphAwareXGBoost_MIMIC": "GraphAware (XGBoost)"
    }
    
    num_cols = len(sorted_exp_keys)
    col_format = "l" + "c" * num_cols
    
    caption_text = r"Test AUROC across featuresets on MIMIC IV."
    
    latex_lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \small",
        rf"    \caption{{{caption_text}}}",
        r"    \label{tab:featureset_auroc_mimic}",
        rf"    \begin{{tabular}}{{{col_format}}}",
        r"        \toprule"
    ]
    
    headers = [r"\textbf{Model}"] + [rf"\textbf{{{exp_mapping[exp]}}}" for exp in sorted_exp_keys]
    latex_lines.append("        " + " & ".join(headers) + r" \\")
    latex_lines.append(r"        \midrule")
    
    max_vals = {}
    for exp in sorted_exp_keys:
        exp_data = df[df['experiment_name'] == exp]
        vals = []
        for raw_model in model_mapping.keys():
            model_data = exp_data[exp_data['tags.mlflow.runName'] == raw_model]
            if not model_data.empty:
                val = model_data.iloc[0][auroc_col]
                if pd.notna(val):
                    vals.append(val)
        max_vals[exp] = max(vals) if vals else None
        
    def fmt_a(val, is_max): 
        if pd.isna(val): return "-"
        val_str = f"{val:.3f}"
        if is_max:
            return f"\\textbf{{{val_str}}}"
        return val_str
        
    for raw_model, clean_model in model_mapping.items():
        row_str = f"        {clean_model:<30}"
        
        for exp in sorted_exp_keys:
            exp_data = df[df['experiment_name'] == exp]
            model_data = exp_data[exp_data['tags.mlflow.runName'] == raw_model]
            
            if not model_data.empty:
                val = model_data.iloc[0][auroc_col]
                is_max = False
                if max_vals[exp] is not None and pd.notna(val):
                    if abs(val - max_vals[exp]) < 1e-6:
                        is_max = True
                row_str += f" & {fmt_a(val, is_max):<15}"
            else:
                row_str += " & -              "
                
        row_str += r" \\"
        latex_lines.append(row_str)
        
    latex_lines.extend([
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}"
    ])
    
    print("\n".join(latex_lines))

if __name__ == "__main__":
    df, auroc_col = get_latest_runs()
    generate_latex_transposed(df, auroc_col)