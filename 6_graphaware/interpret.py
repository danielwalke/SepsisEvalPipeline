import os
import shutil
import configparser
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import shap
import matplotlib.pyplot as plt
from GraphAware.EnsembleFramework import Framework
from connectors.Neo4jConnector import Neo4jConnector ## TODO Test if this also works with sqlite connector
from connectors.SQLiteConnector import SQLiteConnector
from training_containers.SBCTraining import SBCTraining
from training_containers.MIMICTraining import MIMICTraining
from training_containers.SBCTrainingSQLite import SBCTraining as SBCTrainingSQLite
from training_containers.MIMICTrainingSQLite import MIMICTraining as MIMICTrainingSQLite

def diff_user_fun(kwargs):
    return kwargs["original_features"] - kwargs["mean_neighbors"]

PANEL_DISPLAY = {
    "CBC": "CBC",
    "CBC_BMP": "CBC & BMP",
    "CBC_COAG": "CBC & COAG",
    "CBC_BMP_COAG": "CBC, BMP & COAG",
    "CBC_CRP": "CBC & CRP",
    "CBC_KIDNEY": "CBC & Kidney",
    "CBC_LIVER": "CBC & Liver",
    "HIL": "HIL",
}

COHORT_DISPLAY = {
    "MIMIC_TEST": "MIMIC-IV Test Cohort",
    "MIMIC_VAL": "MIMIC-IV Validation Cohort",
    "MIMIC_TRAIN": "MIMIC-IV Training Cohort",
    "SBC_TEST": "SBC Test Cohort",
    "SBC_VAL": "SBC Validation Cohort",
    "SBC_EXT_TEST": "SBC External Test Cohort",
    "SBC_TRAIN": "SBC Training Cohort",
}

def clean_feature_name(name):
    if name.startswith("Total: "):
        name = name[len("Total: "):]
    if name.startswith("f__"):
        name = name[len("f__"):]
    return name

"""
IMPORT COMMENT: Diff to sex is not zero across the same patient because of the weighted avergaing based on the time difference to previous measurements (if the edge weight is indeed 1 for each edge and all are equally weighted then it would be zero, but the edge weights are not all 1 and they are not all the same, so the weighted average of neighbors can differ from the original features even for static features)
"""
def interpret_and_visualize():
    config = configparser.ConfigParser()
    if os.path.exists('/app/config/config.ini'):
        config_path = '/app/config/config.ini'
        app_root = '/app'
        is_docker = True
    else:
        app_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        config_path = os.path.join(app_root, 'config.ini')
        is_docker = False

    config.read(config_path)
    feature_set_name = config['PANEL']['panel_name']
    include_sbc = config['PANEL'].getboolean('include_sbc', fallback=False)
    db = "sqlite"
    
    BATCH_SIZE = 100000
    hops = [0, 1]

    features_file = os.path.join(app_root, "0_mimic_preprocess", "features", "feature_names.txt")
    with open(features_file, "r") as f:
        base_feature_names = [n.strip() for n in f.read().replace("[", "").replace("]", "").replace("'", "").split(",")]
    
    framework = Framework(user_functions=[diff_user_fun for _ in hops], 
                        hops_list=hops,
                        clfs=[None for _ in hops],
                        gpu_idx=0,
                        handle_nan=0.0,
                        attention_configs=[None for _ in hops], classifier_on_device=False)
                        
    if is_docker:
        db_path = f"/app/db/{feature_set_name}/mimic_sbc_graph.db"
    else:
        db_path = os.path.join(app_root, "4_db_upload", "sqlite", "sqlite_data", feature_set_name, "mimic_sbc_graph.db")

    connector = SQLiteConnector(db_path=db_path) if db == "sqlite" else Neo4jConnector(uri="bolt://localhost:7687", user="neo4j", password="password")
    
    training_container = []
    if connector.has_sbc_nodes() and include_sbc:
        sbc_training = SBCTraining() if db == "neo4j" else SBCTrainingSQLite()
        training_container.append(sbc_training)
    if connector.has_mimic_nodes():
        mimic_training = MIMICTraining() if db == "neo4j" else MIMICTrainingSQLite()
        training_container.append(mimic_training)

    panel_label = PANEL_DISPLAY.get(feature_set_name, feature_set_name.replace("_", " & "))

    for container in training_container:
        node_split_container = container.get_node_split_containers(connector)
        train_label_name = node_split_container.train_split_information.label_name
        exp_name = f"{train_label_name.split('_')[0]}_{feature_set_name}"

        if is_docker:
            model_exp_path = os.path.join("/app", "models", exp_name)
            figure_exp_path = os.path.join("/app", "figures", exp_name)
        else:
            model_exp_path = os.path.join(os.path.dirname(__file__), "models", exp_name)
            figure_exp_path = os.path.join(os.path.dirname(__file__), "figures", exp_name)

        os.makedirs(model_exp_path, exist_ok=True)
        os.makedirs(figure_exp_path, exist_ok=True)
        
        model = xgb.Booster()
        model.load_model(os.path.join(model_exp_path, "final_model.xgb"))
        
        for test_info in node_split_container.test_split_information_list:
            all_test_preds, all_test_labels, X_test_all = [], [], []
            skip = 0
            while True:
                X_test, y_test = connector.fetch_data_batch(test_info.label_name, test_info.condition, skip, BATCH_SIZE, framework, test_info.node_ids)
                if len(y_test) == 0: break
                all_test_preds.extend(model.predict(xgb.DMatrix(X_test)))
                all_test_labels.extend(y_test)
                X_test_all.append(X_test)
                skip += BATCH_SIZE
                
            if len(all_test_labels) > 0:
                print(f"Test ({test_info.name}) AUROC: {roc_auc_score(all_test_labels, all_test_preds)}")
                
            if X_test_all:
                X_test_all_np = np.vstack(X_test_all)

                num_features = X_test_all_np.shape[1]
                half_f = num_features // 2
                
                # Strip 'Total: ' and 'f__' prefixes
                actual_base = [clean_feature_name(n) for n in base_feature_names[:half_f]]
                full_names = actual_base + [f"{n} (Δ Mean)" for n in actual_base]
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_all_np)

                cohort_label = COHORT_DISPLAY.get(test_info.name, f"{test_info.name} Cohort")
                
                # Full SHAP summary plot (Raw vs. Delta)
                plt.figure(figsize=(12, 10))
                shap.summary_plot(shap_values, X_test_all_np, feature_names=full_names, show=False)
                main_ax = plt.gcf().axes[0]
                main_ax.set_title(
                    f"Global SHAP feature contributions: Raw vs. Delta Features ({panel_label} - {cohort_label})",
                    fontweight="bold", fontsize=14, pad=16
                )
                main_ax.set_xlabel("SHAP value (impact on sepsis risk model output)", fontweight="bold", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(figure_exp_path, f"shap_full_{test_info.name}.png"), dpi=300, bbox_inches="tight")
                plt.close()
                
                # Aggregated SHAP summary plot
                agg_shap = shap_values[:, :half_f] + shap_values[:, half_f:]
                agg_names = [clean_feature_name(n) for n in actual_base]
                
                plt.figure(figsize=(12, 10))
                shap.summary_plot(agg_shap, X_test_all_np[:, :half_f], feature_names=agg_names, show=False)
                main_ax = plt.gcf().axes[0]
                main_ax.set_title(
                    f"Aggregated global SHAP feature contributions ({panel_label} - {cohort_label})",
                    fontweight="bold", fontsize=14, pad=16
                )
                main_ax.set_xlabel("SHAP value (impact on sepsis risk model output)", fontweight="bold", fontsize=12)
                plt.tight_layout()
                agg_out_path = os.path.join(figure_exp_path, f"shap_aggregated_{test_info.name}.png")
                plt.savefig(agg_out_path, dpi=300, bbox_inches="tight")
                plt.close()

                # Sync to figures/ directory for manuscript compilation
                repo_figures_dir = os.path.join(app_root, "figures")
                if os.path.exists(repo_figures_dir) and test_info.name == "MIMIC_TEST" and feature_set_name == "CBC_BMP":
                    shutil.copy2(agg_out_path, os.path.join(repo_figures_dir, f"shap_aggregated_{test_info.name}.png"))
                    print(f"Synced {agg_out_path} to {os.path.join(repo_figures_dir, f'shap_aggregated_{test_info.name}.png')}")

    connector.close()

if __name__ == "__main__":
    interpret_and_visualize()