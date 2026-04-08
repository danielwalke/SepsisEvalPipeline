import json
from sklearn.model_selection import train_test_split
from GraphAware.EnsembleFramework import Framework
from Neo4jConnector import Neo4jConnector
from XGBoostManager import XGBoostManager
from SBCTraining import SBCTraining
from MIMICTraining import MIMICTraining
import mlflow
import time
import configparser

def diff_user_fun(kwargs):
    return kwargs["original_features"] - kwargs["mean_neighbors"]

if __name__ == "__main__":
    run_hyperparameter_tuning = False
    config = configparser.ConfigParser()
    config.read('config.ini')
    RANDOM_SEED = int(config['RANDOM']['seed'])
    feature_set_name = config['PANEL']['panel_name']
    include_sbc = config['PANEL'].getboolean('include_sbc', fallback=False)

    BATCH_SIZE = 100000
    hops = [0, 1]

    framework = Framework(user_functions=[diff_user_fun for _ in hops], 
                        hops_list=hops,
                        clfs=[None for _ in hops],
                        gpu_idx=0,
                        handle_nan=0.0,
                        attention_configs=[None for _ in hops], classifier_on_device=False)

    connector = Neo4jConnector(uri="bolt://localhost:7687", user="neo4j", password="password")

    training_container = []
    if connector.has_sbc_nodes() and include_sbc:

        sbc_training = SBCTraining()
        training_container.append(sbc_training)
    if connector.has_mimic_nodes():
        mimic_training = MIMICTraining()
        training_container.append(mimic_training)

    for container in training_container:
        node_split_container = container.get_node_split_containers(connector)
        train_label_name = node_split_container.train_split_information.label_name
        val_label_name = node_split_container.val_split_information.label_name
        train_seed_ids = node_split_container.train_split_information.node_ids
        val_seed_ids = node_split_container.val_split_information.node_ids
        train_condition = node_split_container.train_split_information.condition
        exp_name = f"{train_label_name.split('_')[0]}_{feature_set_name}"

        manager = XGBoostManager(train_label_name=train_label_name, val_label_name=val_label_name, batch_size=BATCH_SIZE)

        hyperparam_tuning_start_time = time.time()
        if run_hyperparameter_tuning:
            final_params = manager.optimize_hyperparams(connector, train_seed_ids, val_seed_ids, framework)
        else:
            final_params = {'alpha': 9.538284629683702, 'booster': 'gbtree', 'colsample_bytree': 0.9080724508103653, 'gamma': 0.8284271722786946, 'lambda': 0.00906801548010611, 'learning_rate': 0.15996292193138167, 'max_depth': 3, 'min_child_weight': 3.3449149107880025, 'n_estimators': 150, 'n_jobs': -1, 'objective': 'binary:logistic', 'random_state': RANDOM_SEED, 'scale_pos_weight': 1, 'subsample': 0.8923516991674396}
            num_trees = int(final_params.pop('n_estimators', 150))
            final_params.pop('n_jobs', None)
        hyperparam_tuning_end_time = time.time()
            
        print(final_params)
        with open("true_mini_batch_best_params.json", "w") as f:
            json.dump(final_params, f, indent=4)

        train_start_time = time.time()
        final_model = manager.train_model_iterator(connector, final_params, train_label_name, train_condition, num_trees, framework, train_seed_ids)
        train_end_time = time.time()

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(f"evaluations_{feature_set_name}")

        with mlflow.start_run():
            mlflow.set_tag("model", "GraphAware XGBoost")
            mlflow.set_tag("approach", "Graph-based")
            mlflow.set_tag("feature_set", feature_set_name)
            mlflow.log_params(final_params)           
            for test_info in node_split_container.test_split_information_list:
                inference_start_time = time.time()
                auroc = manager.evaluate_model(connector, final_model, test_info.label_name, test_info.condition, framework, test_info.node_ids)
                inference_end_time = time.time()
                print(f"{test_info.name} AUROC: {auroc}")
                mlflow.log_metric(f"{test_info.name}__inference_time_seconds", inference_end_time - inference_start_time)
                mlflow.log_metric(f"{test_info.name}__auroc", auroc)
            mlflow.log_metric(f"{exp_name}__hyperparameter_tuning_time_seconds", hyperparam_tuning_end_time - hyperparam_tuning_start_time)
            mlflow.log_metric(f"{exp_name}__training__time_seconds", train_end_time - train_start_time)
    connector.close()