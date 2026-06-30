import json
from sklearn.model_selection import train_test_split
from GraphAware.EnsembleFramework import Framework
from connectors.Neo4jConnector import Neo4jConnector
from connectors.SQLiteConnector import SQLiteConnector
from XGBoostManager import XGBoostManager
from training_containers.SBCTraining import SBCTraining
from training_containers.MIMICTraining import MIMICTraining
from training_containers.SBCTrainingSQLite import SBCTraining as SBCTrainingSQLite
from training_containers.MIMICTrainingSQLite import MIMICTraining as MIMICTrainingSQLite
import mlflow
import time
import configparser
import os

def diff_user_fun(kwargs):
    return kwargs["original_features"] - kwargs["mean_neighbors"]

def cross_evaluate(framework, manager, final_model, connector, training_container, use_full_batch, mlflow_args):
    
    for container in training_container:
        print(f"Starting evaluation for {container.name}...")
        node_split_container = container.get_node_split_containers(connector)
        
        mlflow.set_tag("model", "GraphAware XGBoost")
        mlflow.set_tag("approach", "Graph-based")
        mlflow.set_tag("feature_set", mlflow_args["feature_set_name"])
        mlflow.log_params(mlflow_args["final_params"])           
        for test_info in node_split_container.test_split_information_list:
            inference_start_time = time.time()
            if use_full_batch:
                auroc = manager.evaluate_model_full_batch(connector, final_model, test_info.label_name, test_info.condition, framework, test_info.node_ids)
            else:
                auroc = manager.evaluate_model_mini_batch(connector, final_model, test_info.label_name, test_info.condition, framework, test_info.node_ids)
            inference_end_time = time.time()
            print(f"{test_info.name} AUROC: {auroc}")
            mlflow.log_metric(f"{test_info.name}__inference_time_seconds", inference_end_time - inference_start_time)
            mlflow.log_metric(f"{test_info.name}__AUROC", auroc)
        mlflow.log_metric(f"hyperparameter_tuning_time_seconds", mlflow_args["hyperparam_tuning_end_time"] - mlflow_args["hyperparam_tuning_start_time"])
        mlflow.log_metric(f"training_time_seconds", mlflow_args["train_end_time"] - mlflow_args["train_start_time"])


def train_and_evaluate():
    run_hyperparameter_tuning = True
    use_full_batch = False
    config = configparser.ConfigParser()
    config.read('/app/config/config.ini')
    RANDOM_SEED = int(config['RANDOM']['seed'])
    feature_set_name = config['PANEL']['panel_name']
    include_sbc = config['PANEL'].getboolean('include_sbc', fallback=False)
    hops = [0, 1]
    framework = Framework(user_functions=[diff_user_fun for _ in hops], 
                        hops_list=hops,
                        clfs=[None for _ in hops],
                        gpu_idx=0,
                        handle_nan=0.0,
                        attention_configs=[None for _ in hops], classifier_on_device=False)
    db = "sqlite"
    if db == "neo4j":
        connector = Neo4jConnector(uri="bolt://localhost:7687", user="neo4j", password="password")
    elif db == "sqlite":
         db_path = "/app/db_data/mimic_sbc_graph.db"
         connector = SQLiteConnector(db_path=db_path)
    else:
        raise ValueError("Unsupported database type. Please choose 'neo4j' or 'sqlite'.")
    BATCH_SIZE = connector.get_node_count() if use_full_batch else int(eval(config['TRAINING']['batch_size']))
    print(f"Using batch size of {BATCH_SIZE} for training.")

    training_container = []
    if connector.has_sbc_nodes() and include_sbc:
        sbc_training = SBCTraining() if db == "neo4j" else SBCTrainingSQLite()
        training_container.append(sbc_training)
    if connector.has_mimic_nodes():
        mimic_training = MIMICTraining() if db == "neo4j" else MIMICTrainingSQLite()
        training_container.append(mimic_training)

    for container in training_container:
        print(f"Starting training for {container.name}...")
        node_split_container = container.get_node_split_containers(connector)
        train_label_name = node_split_container.train_split_information.label_name
        val_label_name = node_split_container.val_split_information.label_name
        train_seed_ids = node_split_container.train_split_information.node_ids
        val_seed_ids = node_split_container.val_split_information.node_ids
        train_condition = node_split_container.train_split_information.condition
        val_condition = node_split_container.val_split_information.condition
        exp_name = f"{train_label_name.split('_')[0]}_{feature_set_name}"

        model_exp_path = os.path.join("/app", "models", exp_name)
        hyperparams_path = os.path.join("/app", "hyperparameters", exp_name)
        os.makedirs(model_exp_path, exist_ok=True)
        os.makedirs(hyperparams_path, exist_ok=True)

        manager = XGBoostManager(train_label_name=train_label_name, val_label_name=val_label_name, batch_size=BATCH_SIZE)

        hyperparam_tuning_start_time = time.time()
        if run_hyperparameter_tuning:
            mode_str = 'full_batch' if use_full_batch else 'mini_batch'
            final_params = manager.optimize_hyperparams(connector, train_condition, val_condition, train_seed_ids, val_seed_ids, framework, mode=mode_str, max_evals=20)
        else:
            # final_params = {'alpha': 9.538284629683702, 'booster': 'gbtree', 'colsample_bytree': 0.9080724508103653, 'gamma': 0.8284271722786946, 'lambda': 0.00906801548010611, 'learning_rate': 0.15996292193138167, 'max_depth': 3, 'min_child_weight': 3.3449149107880025, 'n_estimators': 150, 'n_jobs': -1, 'objective': 'binary:logistic', 'random_state': RANDOM_SEED, 'scale_pos_weight': 1, 'subsample': 0.8923516991674396}
            with open(os.path.join(hyperparams_path, "best_params.json"), "r") as f:
                final_params = json.load(f)
            # final_params.pop('n_jobs', None)
        num_trees = int(final_params.pop('n_estimators', 150))
        hyperparam_tuning_end_time = time.time()
            
        print(final_params)
        with open(os.path.join(hyperparams_path, "best_params.json"), "w") as f:
            json.dump(final_params, f, indent=4)

        train_start_time = time.time()
        if use_full_batch:
            print("Running in FULL-BATCH training mode...")
            final_model = manager.train_model_full_batch(connector, final_params, train_label_name, train_condition, num_trees, framework, train_seed_ids)
        else:
            print("Running in MINI-BATCH iterator training mode...")
            final_model = manager.train_model_iterator(connector, final_params, train_label_name, train_condition, num_trees, framework, train_seed_ids)
        train_end_time = time.time()
        manager.save_model(final_model, os.path.join(model_exp_path, "final_model.xgb"))

        mlflow.set_tracking_uri("http://mlflow-server:5000")
        mlflow.set_experiment(f"evaluations_{feature_set_name}")
        mlflow_args = {
            "feature_set_name": feature_set_name,
            "final_params": final_params,
            "hyperparam_tuning_start_time": hyperparam_tuning_start_time,
            "hyperparam_tuning_end_time": hyperparam_tuning_end_time,
            "train_start_time": train_start_time,
            "train_end_time": train_end_time
        }
        with mlflow.start_run(run_name=f"GraphAwareXGBoost_{container.name}"):
            cross_evaluate(framework, manager, final_model, connector, training_container, use_full_batch, mlflow_args)
    connector.close()