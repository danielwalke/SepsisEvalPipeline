import xgboost as xgb
from neo4j import GraphDatabase
import numpy as np
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

uri = "bolt://localhost:7687"
user = "neo4j"
password = "password"
driver = GraphDatabase.driver(uri, auth=(user, password))

run_hyperparameter_tuning = True

def fetch_data_full(tx, node_label, condition=""):
    query = f"MATCH (n:{node_label}) {condition} RETURN n.label AS label, n.features AS features, n.aggregated_features AS agg_features"
    result = tx.run(query)
    
    labels = []
    combined_features = []
    
    for record in result:
        labels.append(record["label"])
        f1 = record["features"] or []
        f2 = record["agg_features"] or []
        combined_features.append(f1 + f2)
        
    return np.array(combined_features, dtype=np.float32), np.array(labels, dtype=np.float32)

def evaluate_model(session, model, node_label, condition=""):
    X_test, y_test = session.execute_read(fetch_data_full, node_label, condition)
    
    if len(y_test) == 0:
        return 0.0
        
    dtest = xgb.DMatrix(X_test)
    preds = model.predict(dtest)
    
    return roc_auc_score(y_test, preds)

def train_model_full(session, params, node_label):
    train_cond = "WHERE toInteger(n.patientId) % 4 < 3"
    val_cond = "WHERE toInteger(n.patientId) % 4 = 3"

    X_train, y_train = session.execute_read(fetch_data_full, node_label, train_cond)
    X_val, y_val = session.execute_read(fetch_data_full, node_label, val_cond)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        params, 
        dtrain, 
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    
    train_preds = model.predict(dtrain)
    train_auroc = roc_auc_score(y_train, train_preds)
    print(f"Train AUROC: {train_auroc}")
    
    return model

def optimize_hyperparams(session):
    space = {
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'learning_rate': hp.uniform('learning_rate', 0.001, 0.3),
        'subsample': hp.uniform('subsample', 0.4, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.4, 1.0),
        'min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
        'gamma': hp.uniform('gamma', 0.0, 5.0),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 10.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 10.0)
    }
    
    def objective(params):
        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": 640,
            "max_depth": int(params['max_depth']),
            "learning_rate": params['learning_rate'],
            "subsample": params['subsample'],
            "colsample_bytree": params['colsample_bytree'],
            "min_child_weight": int(params['min_child_weight']),
            "gamma": params['gamma'],
            "reg_alpha": params['reg_alpha'],
            "reg_lambda": params['reg_lambda']
        }
        
        train_cond = "WHERE toInteger(n.patientId) % 4 < 3"
        val_cond = "WHERE toInteger(n.patientId) % 4 = 3"
        
        X_train, y_train = session.execute_read(fetch_data_full, "SBC_TRAIN", train_cond)
        X_val, y_val = session.execute_read(fetch_data_full, "SBC_TRAIN", val_cond)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            xgb_params, 
            dtrain, 
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        preds = model.predict(dval)
        auroc = roc_auc_score(y_val, preds)
        
        return {'loss': -auroc, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials, verbose=1)
    return best

with driver.session() as session:
    if run_hyperparameter_tuning:
        best_params = optimize_hyperparams(session)
        print(best_params)
        
        final_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": 640,
            "max_depth": int(best_params['max_depth']),
            "learning_rate": best_params['learning_rate'],
            "subsample": best_params['subsample'],
            "colsample_bytree": best_params['colsample_bytree'],
            "min_child_weight": int(best_params['min_child_weight']),
            "gamma": best_params['gamma'],
            "reg_alpha": best_params['reg_alpha'],
            "reg_lambda": best_params['reg_lambda']
        }
    else:
        final_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": 640
        }

    final_model = train_model_full(session, final_params, "SBC_TRAIN")
    test_auroc = evaluate_model(session, final_model, "SBC_TEST", "")
    print(f"Test AUROC: {test_auroc}")

driver.close()