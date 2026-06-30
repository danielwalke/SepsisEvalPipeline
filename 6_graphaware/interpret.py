import os
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


"""
IMPORT COMMENT: Diff to sex is not zero across the same patient because of the weighted avergaing based on the time difference to previous measurements (if the edge weight is indeed 1 for each edge and all are equally weighted then it would be zero, but the edge weights are not all 1 and they are not all the same, so the weighted average of neighbors can differ from the original features even for static features)
"""
def interpret_and_visualize():
    config = configparser.ConfigParser()
    config.read('/app/config/config.ini')
    feature_set_name = config['PANEL']['panel_name']
    include_sbc = config['PANEL'].getboolean('include_sbc', fallback=False)
    db = "sqlite"
    
    BATCH_SIZE = 100000
    hops = [0, 1]
    ##TODO adopt paths for docker
    with open("/app/0_mimic_preprocess/features/feature_names.txt", "r") as f:
        base_feature_names = [n.strip() for n in f.read().replace("[", "").replace("]", "").replace("'", "").split(",")]
    
    framework = Framework(user_functions=[diff_user_fun for _ in hops], 
                        hops_list=hops,
                        clfs=[None for _ in hops],
                        gpu_idx=0,
                        handle_nan=0.0,
                        attention_configs=[None for _ in hops], classifier_on_device=False)
                        
    connector = SQLiteConnector(db_path = "/app/db_data/mimic_sbc_graph.db") if db == "sqlite" else Neo4jConnector(uri="bolt://localhost:7687", user="neo4j", password="password")
    
    training_container = []
    if connector.has_sbc_nodes() and include_sbc:
        sbc_training = SBCTraining() if db == "neo4j" else SBCTrainingSQLite()
        training_container.append(sbc_training)
    if connector.has_mimic_nodes():
        mimic_training = MIMICTraining() if db == "neo4j" else MIMICTrainingSQLite()
        training_container.append(mimic_training)


    for container in training_container:
        node_split_container = container.get_node_split_containers(connector)
        train_label_name = node_split_container.train_split_information.label_name
        exp_name = f"{train_label_name.split('_')[0]}_{feature_set_name}"


        ##TODO adopt paths for docker -> TODO test
        model_exp_path = os.path.join("/app", "models", exp_name)
        figure_exp_path = os.path.join("/app", "figures", exp_name)
        os.makedirs(model_exp_path, exist_ok=True)
        os.makedirs(figure_exp_path, exist_ok=True)
        
        model = xgb.Booster()
        model.load_model(os.path.join(model_exp_path, "final_model.xgb"))
        
        val_info = node_split_container.val_split_information
        
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
                
                actual_base = base_feature_names[:half_f]
                full_names = actual_base + [f"{n} (Δ Mean)" for n in actual_base]
                
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_all_np)
                
                plt.figure(figsize=(12, 10))
                shap.summary_plot(shap_values, X_test_all_np, feature_names=full_names, show=False)
                plt.title(f"Global Importance: Raw vs. Delta Features ({test_info.name})")
                plt.tight_layout()
                plt.savefig(os.path.join(figure_exp_path, f"shap_full_{test_info.name}.png"))
                plt.close()
                
                agg_shap = shap_values[:, :half_f] + shap_values[:, half_f:]
                agg_names = [f"Total: {n}" for n in actual_base]
                
                plt.figure(figsize=(12, 10))
                shap.summary_plot(agg_shap, X_test_all_np[:, :half_f], feature_names=agg_names, show=False)
                plt.title(f"Aggregated Global Importance ({test_info.name})")
                plt.tight_layout()
                plt.savefig(os.path.join(figure_exp_path, f"shap_aggregated_{test_info.name}.png"))
                plt.close()

    connector.close()