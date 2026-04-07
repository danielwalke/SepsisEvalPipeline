import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
panel_name = str(config['PANEL']['panel_name'])

mlflow.set_tracking_uri("http://localhost:5000")

runs_df = mlflow.search_runs(search_all_experiments=True)
print(runs_df.head())
print(runs_df.columns)

metric_cols = [col for col in runs_df.columns if col.startswith("metrics.")]
target_tag = "tags.model" 

global_min = runs_df[metric_cols].min().min()
global_max = runs_df[metric_cols].max().max()

y_min = global_min * 0.9 if global_min > 0 else 1e-4
y_max = min(global_max * 1.1, 1.0)

if target_tag in runs_df.columns:
    agg_df = runs_df.groupby(target_tag)[metric_cols].mean().reset_index()
    
    for metric in metric_cols:
        if "VAL" in metric.upper():
            continue

        plt.figure(figsize=(10, 6))
        
        plt.gca().set_axisbelow(True)
        plt.grid(True, which="both", axis="y", linestyle="--", alpha=0.7)
        
        plt.bar(agg_df[target_tag].astype(str), agg_df[metric], color="skyblue")
        
        plt.yscale("log")
        plt.ylim(bottom=y_min, top=y_max)
        
        plt.xlabel(target_tag)
        plt.ylabel(f"Mean {metric} (Log Scale)")
        plt.title(f"Average {metric} Grouped by {target_tag}")
        plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()
        plt.savefig(f"experiment_logging/{metric}_by_tags_model_{panel_name}.png")
        plt.close()