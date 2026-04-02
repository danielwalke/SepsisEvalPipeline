from neo4j import GraphDatabase
import torch
import numpy as np
from torch_geometric.data import Data
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from GraphAware.EnsembleFramework import Framework
import json
from Neo4jConnector import Neo4jConnector
from Neo4jDataloader import Neo4jDataIter

run_hyperparameter_tuning = False
BATCH_SIZE = 100000

def diff_user_fun(kwargs):
    return kwargs["original_features"] - kwargs["mean_neighbors"]

hops = [0, 1]
attention_config = None
attention_configs = [attention_config for _ in hops]

user_function = diff_user_fun
user_functions = [user_function for _ in hops]
models = [None for _ in hops]

framework = Framework(user_functions=user_functions, 
                    hops_list=hops,
                    clfs=models,
                    gpu_idx=0,
                    handle_nan=0.0,
                    attention_configs=attention_configs, classifier_on_device=False)


def train_model_iterator(connector, params, node_label, condition, num_trees, framework, target_ids=None):
    iterator = Neo4jDataIter(connector, node_label, condition, framework, BATCH_SIZE, target_ids)
    dtrain = xgb.QuantileDMatrix(iterator)
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=num_trees,
        verbose_eval=False
    )
    
    return model

def evaluate_model(connector, model, node_label, condition, framework, target_ids=None):
    skip = 0
    all_preds = []
    all_labels = []
    
    while True:
        X_test, y_test = connector.fetch_data_batch(node_label, condition, skip, BATCH_SIZE, framework, target_ids)
        
        if len(y_test) == 0:
            break
            
        dtest = xgb.DMatrix(X_test)
        preds = model.predict(dtest)
        
        all_preds.extend(preds)
        all_labels.extend(y_test)
        skip += BATCH_SIZE
        
    if len(all_labels) == 0:
        return 0.0
        
    return roc_auc_score(all_labels, all_preds)

def optimize_hyperparams(connector, train_ids, val_ids, framework):
    space = {
        'alpha': hp.uniform('alpha', 5.0, 15.0),
        'booster': 'gbtree',
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
        'gamma': hp.uniform('gamma', 0.4, 1.3),
        'lambda': hp.loguniform('lambda', np.log(0.001), np.log(0.05)),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.05), np.log(0.3)),
        'max_depth': hp.choice('max_depth', [2, 3, 4, 5]),
        'min_child_weight': hp.uniform('min_child_weight', 1.5, 5.5),
        'n_estimators': hp.choice('n_estimators', [50, 100, 125, 150, 175, 200]),
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'random_state': 42,
        'scale_pos_weight': 1,
        'subsample': hp.uniform('subsample', 0.7, 1.0)
    }
    
    def objective(params):
        num_trees = params.get('n_estimators', 100)
        
        xgb_params = {
            "objective": "binary:logistic",
            "scale_pos_weight": 1,
            "max_depth": int(params['max_depth']),
            "learning_rate": float(params['learning_rate']),
            "subsample": float(params['subsample']),
            "colsample_bytree": float(params['colsample_bytree']),
            "min_child_weight": float(params['min_child_weight']),
            "gamma": float(params['gamma']),
            "alpha": float(params['alpha']),
            "reg_lambda": float(params['lambda']),
            "random_state": 42,
            "booster": 'gbtree',
        }
        
        train_cond = "WHERE n.patientId IN $ids"
        val_cond = "WHERE n.patientId IN $ids"
        
        model = train_model_iterator(connector, xgb_params, "SBC_TRAIN", train_cond, int(num_trees), framework, train_ids)
        auroc = evaluate_model(connector, model, "SBC_TRAIN", val_cond, framework, val_ids)
        
        return {'loss': -auroc, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=80, trials=trials, verbose=1)
    return best

connector = Neo4jConnector(uri="bolt://localhost:7687", user="neo4j", password="password")

all_train_ids = connector.get_ids('SBC_TRAIN')
train_seed_ids, val_seed_ids = train_test_split(all_train_ids, test_size=0.2, random_state=42)

if run_hyperparameter_tuning:
    best_params = optimize_hyperparams(connector, train_seed_ids, val_seed_ids, framework)
    print(best_params)
    
    num_trees = int(best_params.get('n_estimators', 150))
    
    final_params = {
        "objective": "binary:logistic",
        "scale_pos_weight": 1,
        "max_depth": int(best_params['max_depth']),
        "learning_rate": float(best_params['learning_rate']),
        "subsample": float(best_params['subsample']),
        "colsample_bytree": float(best_params['colsample_bytree']),
        "min_child_weight": float(best_params['min_child_weight']),
        "gamma": float(best_params['gamma']),
        "alpha": float(best_params['alpha']),
        "reg_lambda": float(best_params['lambda']),
        "random_state": 42,
        "booster": 'gbtree',
    }
    
else:
    final_params = {'alpha': 9.538284629683702, 'booster': 'gbtree', 'colsample_bytree': 0.9080724508103653, 'gamma': 0.8284271722786946, 'lambda': 0.00906801548010611, 'learning_rate': 0.15996292193138167, 'max_depth': 3, 'min_child_weight': 3.3449149107880025, 'n_estimators': 150, 'n_jobs': -1, 'objective': 'binary:logistic', 'random_state': 42, 'scale_pos_weight': 1, 'subsample': 0.8923516991674396}
    num_trees = int(final_params.pop('n_estimators', 150))
    final_params.pop('n_jobs', None)
    
print(final_params)
with open("true_mini_batch_best_params.json", "w") as f:
    json.dump(final_params, f, indent=4)

train_cond = "WHERE n.id IN $ids"
final_model = train_model_iterator(connector, final_params, "SBC_TRAIN", train_cond, num_trees, framework, train_seed_ids)

train_auroc = evaluate_model(connector, final_model, "SBC_TRAIN", train_cond, framework, train_seed_ids)
print(f"Train AUROC: {train_auroc}")

test_auroc = evaluate_model(connector, final_model, "SBC_TEST", "", framework)
print(f"Test AUROC: {test_auroc}")
ext_test_auroc = evaluate_model(connector, final_model, "SBC_EXT_TEST", "", framework)
print(f"Ext Test AUROC: {ext_test_auroc}")

connector.close()