import json
from sklearn.model_selection import train_test_split
from GraphAware.EnsembleFramework import Framework
from Neo4jConnector import Neo4jConnector
from XGBoostManager import XGBoostManager

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

connector = Neo4jConnector(uri="bolt://localhost:7687", user="neo4j", password="password")
manager = XGBoostManager(batch_size=BATCH_SIZE)

all_train_ids = connector.get_ids('SBC_TRAIN')
train_seed_ids, val_seed_ids = train_test_split(all_train_ids, test_size=0.2, random_state=42)

if run_hyperparameter_tuning:
    best_params = manager.optimize_hyperparams(connector, train_seed_ids, val_seed_ids, framework)
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
final_model = manager.train_model_iterator(connector, final_params, "SBC_TRAIN", train_cond, num_trees, framework, train_seed_ids)

train_auroc = manager.evaluate_model(connector, final_model, "SBC_TRAIN", train_cond, framework, train_seed_ids)
print(f"Train AUROC: {train_auroc}")

test_auroc = manager.evaluate_model(connector, final_model, "SBC_TEST", "", framework)
print(f"Test AUROC: {test_auroc}")

ext_test_auroc = manager.evaluate_model(connector, final_model, "SBC_EXT_TEST", "", framework)
print(f"Ext Test AUROC: {ext_test_auroc}")

connector.close()