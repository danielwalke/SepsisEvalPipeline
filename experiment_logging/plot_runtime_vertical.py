"""Regenerates runtime_vertical_plot.png (Fig. fig:runtimes in current_manuscript.tex).

For each ML model, plots the median hyperparameter-tuning, training, and inference
time (log scale) across the four MIMIC-IV feature-set panels (CBC, CBC & BMP,
CBC & COAG, CBC & BMP & COAG), with error caps spanning the full min-max range
across those four panels. One most-recent run per (feature set, model) is used.
"""
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://localhost:5000")

FEATURE_SETS = ["CBC", "CBC_BMP", "CBC_COAG", "CBC_BMP_COAG"]
FEATURE_SET_LABELS = {
    "CBC": "CBC",
    "CBC_BMP": "CBC & BMP",
    "CBC_COAG": "CBC & COAG",
    "CBC_BMP_COAG": "CBC & BMP & COAG",
}

MODEL_ORDER = [
    "Baseline Log. Reg.",
    "Baseline Random Forest",
    "Baseline XGBoost",
    "GNN (GATv2)",
    "GraphAware (XGBoost)",
]
RUN_NAME_TO_MODEL = {
    "LogisticRegression_Baseline_MIMIC": "Baseline Log. Reg.",
    "RandomForestClassifier_Baseline_MIMIC": "Baseline Random Forest",
    "XGBClassifier_Baseline_MIMIC": "Baseline XGBoost",
    "GNN_MIMIC": "GNN (GATv2)",
    "GraphAwareXGBoost_MIMIC": "GraphAware (XGBoost)",
}

METRICS = [
    ("hyperparameter_tuning_time_seconds", "Hyperparameter Tuning Time"),
    ("training_time_seconds", "Training Time"),
    ("MIMIC_TEST__inference_time_seconds", "Inference Time"),
]

# Connect to MLflow server with SQLite fallback
try:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.search_experiments()
except Exception:
    mlflow.set_tracking_uri("sqlite:///mlflow_data/mlflow.db")

rows = []
for fs in FEATURE_SETS:
    exp_name = f"evaluations_{fs}"
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp:
        print(f"WARNING: experiment not found: {exp_name}")
        continue
    df = mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["attributes.start_time DESC"])
    if df.empty or "tags.mlflow.runName" not in df.columns:
        print(f"WARNING: no valid runs for {exp_name}")
        continue

    df = df[df["tags.mlflow.runName"].isin(RUN_NAME_TO_MODEL.keys())]

    # For GNN and GraphAware, use the most recent run (reflecting hyperopt fix).
    # For baseline models, use the best-AUROC run (matching Table 2).
    is_graph_model = df["tags.mlflow.runName"].isin(["GNN_MIMIC", "GraphAwareXGBoost_MIMIC"])
    graph_df = df[is_graph_model].sort_values("start_time", ascending=False)
    baseline_df = df[~is_graph_model]
    auroc_col = "metrics.MIMIC_TEST__AUROC"
    if auroc_col in baseline_df.columns:
        baseline_df = baseline_df.sort_values(auroc_col, ascending=False)
    df = pd.concat([graph_df, baseline_df])
    df = df.drop_duplicates(subset=["tags.mlflow.runName"], keep="first")

    for _, row in df.iterrows():
        model = RUN_NAME_TO_MODEL[row["tags.mlflow.runName"]]
        entry = {"feature_set": fs, "model": model}
        for metric_col, _ in METRICS:
            full_col = f"metrics.{metric_col}"
            entry[metric_col] = row[full_col] if full_col in row and pd.notna(row[full_col]) else np.nan
        rows.append(entry)

data = pd.DataFrame(rows)
if data.empty:
    raise SystemExit("No data collected from MLflow - aborting plot generation.")

print("Collected runtime data across panels:")
print(data[["feature_set", "model"] + [m for m, _ in METRICS]].to_string(index=False))

import matplotlib as mpl
import os
import shutil

# Publication styling matching author's established palette and journal aesthetics
COLORS = ["#427dae", "#e67d15", "#279876", "#92569a", "#a0623e"]
HATCHES = ["//", "\\\\\\\\", "xx", "..", "---"]

mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
mpl.rcParams["hatch.linewidth"] = 1.0
mpl.rcParams["hatch.color"] = "black"

fig, axes = plt.subplots(3, 1, figsize=(8.5, 12), sharex=True)

for i, (ax, (metric_col, title)) in enumerate(zip(axes, METRICS)):
    medians, mins, maxs = [], [], []
    for model in MODEL_ORDER:
        vals = data.loc[data["model"] == model, metric_col].dropna().to_numpy()
        med = np.median(vals)
        medians.append(med)
        mins.append(vals.min())
        maxs.append(vals.max())

    medians = np.array(medians)
    mins = np.array(mins)
    maxs = np.array(maxs)

    lo_err = np.maximum(medians - mins, 0)
    hi_err = np.maximum(maxs - medians, 0)

    x = np.arange(len(MODEL_ORDER))
    width = 0.68
    bars = ax.bar(
        x, medians, width=width,
        color=COLORS, edgecolor="black", linewidth=1.0, zorder=3
    )
    for bar, hatch in zip(bars, HATCHES):
        bar.set_hatch(hatch)

    ax.errorbar(
        x, medians, yerr=[lo_err, hi_err],
        fmt="none", ecolor="black", elinewidth=1.3, capsize=0, zorder=4
    )

    ax.set_yscale("log")
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_ylabel("Time (s)", fontsize=13)

    ax.grid(axis="y", which="both", linestyle="--", alpha=0.35, color="gray", zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Annotate median runtime values directly above each bar / error bar
    for xi, med, hi in zip(x, medians, maxs):
        top_val = max(med, hi)
        text_y = top_val * 1.35
        ax.text(
            xi, text_y, f"{med:.2f}s",
            ha="center", va="bottom", fontsize=11, color="black", zorder=5
        )

    # Set y-limits with headroom for annotations
    if i == 0:
        ax.set_ylim(bottom=1.5, top=55000)
    elif i == 1:
        ax.set_ylim(bottom=0.15, top=1500)
    elif i == 2:
        ax.set_ylim(bottom=0.005, top=60)

axes[-1].set_xticks(np.arange(len(MODEL_ORDER)))
axes[-1].set_xticklabels(MODEL_ORDER, rotation=15, ha="right", fontsize=12.5)
axes[-1].tick_params(axis="x", length=4, width=1.0)

for ax in axes[:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

plt.tight_layout(h_pad=2.0)

out_path = "runtime_vertical_plot.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved publication-ready plot to {out_path}")

os.makedirs("figures", exist_ok=True)
figures_out = os.path.join("figures", "runtime_vertical_plot.png")
shutil.copy2(out_path, figures_out)
print(f"Synced plot to {figures_out}")
