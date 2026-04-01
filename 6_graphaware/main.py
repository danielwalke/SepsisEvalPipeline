import xgboost as xgb
from neo4j import GraphDatabase
import numpy as np
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from GraphAware.EnsembleFramework import Framework
import json

uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"
driver = GraphDatabase.driver(uri, auth=(user, password))

run_hyperparameter_tuning = True
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

def fetch_data_batch(tx, node_label, condition, skip, limit):
    query = f"MATCH (n:{node_label}) {condition} WITH n ORDER BY n.patientId SKIP $skip LIMIT $limit OPTIONAL MATCH (n)<-[e]-(m) RETURN id(n) AS seed_id, n.label AS seed_label, n.features AS seed_f1, id(m) AS neighbor_id, m.features AS neighbor_f1, e.weight AS edge_weight"
    result = tx.run(query, skip=skip, limit=limit)
    
    node_features = []
    node_labels = []
    node_mask = []
    id_to_index = {}
    edge_sources = []
    edge_targets = []
    edge_weights = []
    if result.peek() is None or len(result.peek()) == 0:
        return np.array([]), np.array([])
    
    for record in result:
        seed_id = record["seed_id"]
        
        if seed_id not in id_to_index:
            id_to_index[seed_id] = len(node_features)
            f1 = record["seed_f1"] or []
            node_features.append(f1)
            
            label = record["seed_label"]
            node_labels.append(label if label is not None else np.nan)
            node_mask.append(True)
            
        neighbor_id = record["neighbor_id"]
        
        if neighbor_id is not None:
            if neighbor_id not in id_to_index:
                id_to_index[neighbor_id] = len(node_features)
                f1_n = record["neighbor_f1"] or []
                node_features.append(f1_n)
                
                node_labels.append(np.nan)
                node_mask.append(False)
                
            edge_sources.append(id_to_index[seed_id])
            edge_targets.append(id_to_index[neighbor_id])
            
            weight = record["edge_weight"]
            edge_weights.append(weight if weight is not None else 1.0)
            
    features_arr = np.array(node_features, dtype=np.float32)
    labels_arr = np.array(node_labels, dtype=np.float32)
    edge_index_arr = np.array([edge_sources, edge_targets], dtype=np.int64)
    edge_weights_arr = np.array(edge_weights, dtype=np.float32)
    mask_arr = np.array(node_mask, dtype=bool)

    final_features = framework.get_features(features_arr, edge_index_arr, edge_weights_arr, mask=mask_arr, is_training=False)
    final_features = np.concatenate([f.cpu() for f in final_features], axis=1)
    
    extracted_labels = labels_arr[mask_arr]
    valid_mask = ~np.isnan(extracted_labels)
    
    return final_features[valid_mask], extracted_labels[valid_mask]

def evaluate_model(session, model, node_label, condition=""):
    skip = 0
    all_preds = []
    all_labels = []
    
    while True:
        X_test, y_test = session.execute_read(fetch_data_batch, node_label, condition, skip, BATCH_SIZE)
        
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

def train_model_batched(session, params, node_label, trees_per_batch=50):
    train_cond = "WHERE toInteger(n.patientId) % 4 < 3"
    skip = 0
    model = None
    
    while True:
        X_train, y_train = session.execute_read(fetch_data_batch, node_label, train_cond, skip, BATCH_SIZE)
        
        if len(y_train) == 0:
            break
            
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=trees_per_batch,
            xgb_model=model,
            verbose_eval=False
        )
        
        skip += BATCH_SIZE
        
    return model

def optimize_hyperparams(session):
    space = {
        'alpha': hp.uniform('alpha', 5.0, 15.0),
        'booster': 'gbtree',
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
        'gamma': hp.uniform('gamma', 0.4, 1.3),
        'lambda': hp.loguniform('lambda', np.log(0.001), np.log(0.05)),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.05), np.log(0.3)),
        'max_depth': hp.choice('max_depth', [2, 3, 4, 5]),
        'min_child_weight': hp.uniform('min_child_weight', 1.5, 5.5),
        'n_estimators': hp.choice('n_estimators', [100, 125, 150, 175, 200]),
        'n_jobs': -1,
        'objective': 'binary:logistic',
        'random_state': 42,
        'scale_pos_weight': 1,
        'subsample': hp.uniform('subsample', 0.7, 1.0)
    }
    
    def objective(params):
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
        
        val_cond = "WHERE toInteger(n.patientId) % 4 = 3"
        
        model = train_model_batched(session, xgb_params, "SBC_TRAIN", trees_per_batch=20)
        auroc = evaluate_model(session, model, "SBC_TRAIN", val_cond)
        
        return {'loss': -auroc, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=80, trials=trials, verbose=1)
    return best

with driver.session() as session:
    if run_hyperparameter_tuning:
        best_params = optimize_hyperparams(session)
        print(best_params)
        
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
        #final_params =  {'alpha': np.float64(9.458143880601247), 'colsample_bytree': np.float64(0.8118251996907339), 'gamma': np.float64(1.013022312981939), 'lambda': np.float64(0.0013768509710992743), 'learning_rate': np.float64(0.09107721668432846), 'max_depth': np.int64(0), 'min_child_weight': np.float64(3.486381643016147), 'n_estimators': np.int64(2), 'subsample': np.float64(0.7972547446622092)}
    print(final_params)
    with open("best_params.json", "w") as f:
        json.dump(final_params, f, indent=4)

    final_model = train_model_batched(session, final_params, "SBC_TRAIN", trees_per_batch=50)
    
    train_cond = "WHERE toInteger(n.patientId) % 4 < 3"
    train_auroc = evaluate_model(session, final_model, "SBC_TRAIN", train_cond)
    print(f"Train AUROC: {train_auroc}")
    
    test_auroc = evaluate_model(session, final_model, "SBC_TEST", "")
    print(f"Test AUROC: {test_auroc}")

driver.close()