import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(TRACKING_URI)
print(f"Connecting to MLflow server at {TRACKING_URI}...")

try:
    experiments = mlflow.search_experiments()
    experiment_ids = [exp.experiment_id for exp in experiments]
    runs_df = mlflow.search_runs(experiment_ids=experiment_ids)
except Exception as e:
    print(f"Error connecting to MLflow: {e}")
    exit(1)

column_mapping = {
    'tags.mlflow.runName': 'Run Name',
    'metrics.training_time_seconds': 'training_time_seconds',
    'metrics.hyperparameter_tuning_time_seconds': 'hyperparameter_tuning_time_seconds',
    'metrics.MIMIC_TEST__inference_time_seconds': 'MIMIC_TEST__inference_time_seconds'
}
available_cols = [col for col in column_mapping.keys() if col in runs_df.columns]
df = runs_df[available_cols].rename(columns=column_mapping)

df['run_id'] = runs_df['run_id']

if 'Run Name' not in df.columns:
    df['Run Name'] = df['run_id']
else:
    df['Run Name'] = df['Run Name'].fillna(df['run_id'])

desired_order = [
    'Baseline Logistic Regression',
    'Baseline Random Forest',
    'Baseline XGBoost',
    'GNN (GATv2)',
    'GraphAware'
]

def standardize_run_name(name):
    name_lower = str(name).lower()
    if 'logistic' in name_lower or 'lr' in name_lower:
        return desired_order[0]
    if 'randomforest' in name_lower or 'random forest' in name_lower:
        return desired_order[1]
    if ('xgbclassifier' in name_lower) or ('xgb' in name_lower and 'graphaware' not in name_lower):
        return desired_order[2]
    if 'gnn' in name_lower:
        return desired_order[3]
    if 'graphaware' in name_lower:
        return desired_order[4]
    return name

df['Run Name'] = df['Run Name'].apply(standardize_run_name)
df = df[df['Run Name'].isin(desired_order)]

if df.empty:
    print("Error: Could not find any of the target experiments.")
    exit(1)

df['Run Name'] = pd.Categorical(df['Run Name'], categories=desired_order, ordered=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

colors = sns.color_palette('Set2', n_colors=len(desired_order))
color_mapping = dict(zip(desired_order, colors))

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)

metrics_setup = [
    ('training_time_seconds', 'Training Time'),
    ('hyperparameter_tuning_time_seconds', 'Hyperparameter Tuning Time'),
    ('MIMIC_TEST__inference_time_seconds', 'Inference Time')
]

for i, (col, title) in enumerate(metrics_setup):
    if col not in df.columns:
        axes[i].text(0.5, 0.5, f"Missing {col}", ha='center', va='center')
        continue
        
    df_metric = df.dropna(subset=[col])
    
    if df_metric.empty:
        continue

    sns.barplot(
        x=col, 
        y='Run Name', 
        data=df_metric, 
        ax=axes[i], 
        hue='Run Name',      
        palette=color_mapping, 
        legend=False,        
        capsize=0.1,
        err_kws={'linewidth': 1.5}          
    )
    
    axes[i].set_xscale('log')
    
    axes[i].set_title(title, fontsize=14, fontweight='bold', pad=10)
    axes[i].set_xlabel('Time (seconds) - Log Scale', fontsize=11)
    axes[i].set_ylabel('')
    
    axes[i].grid(axis='x', which='both', linestyle='--', alpha=0.5)
    axes[i].grid(axis='y', visible=False)
    
    max_val = df_metric[col].max()
    if pd.notnull(max_val):
        axes[i].set_xlim(right=max_val * 2)

legend_patches = [mpatches.Patch(color=color_mapping[name], label=name) for name in desired_order]
fig.legend(
    handles=legend_patches, 
    loc='lower center', 
    bbox_to_anchor=(0.5, -0.02), 
    ncol=3, 
    fontsize=11,
    frameon=True,
    title="Model Architecture",
    title_fontsize=12
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('mlflow_mimic_aggregated_log_runs.png', dpi=300, bbox_inches='tight')
print("Successfully generated aggregated plot with log scale.")