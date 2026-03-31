import pandas as pd
import numpy as np
import ast
from xgboost import XGBClassifier
from GraphAware.EnsembleFramework import Framework
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def get_graph_data(nodes_path, edges_path):
    train_nodes = pd.read_csv(nodes_path)
    train_edges = pd.read_csv(edges_path)

    X_train = np.array(train_nodes["X"].apply(ast.literal_eval).tolist())
    y_train = train_nodes["y"].str.contains("Sepsis").astype(int).values
    edge_index = np.array(train_edges[["source", "target"]].values).T
    edge_weights = train_edges["weight"].values

    return X_train, y_train, edge_index, edge_weights

X_train, y_train, edge_index, edge_weights = get_graph_data("../3_graph_construction/data/sbc_nodes.csv", "../3_graph_construction/data/sbc_edges.csv")
X_test, y_test, edge_index_test, edge_weights_test = get_graph_data("../3_graph_construction/data/sbc_validation_nodes.csv", "../3_graph_construction/data/sbc_validation_edges.csv")

def norm_user_function(kwargs):
    return  kwargs["original_features"] - kwargs["mean_neighbors"]

hops = [0, 1]
attention_config = None
attention_configs = [attention_config for _ in hops]

user_function = norm_user_function
user_functions = [user_function for _ in hops]
models = [None for _ in hops]

framework = Framework(user_functions=user_functions, 
                    hops_list=hops,
                    clfs=models,
                    gpu_idx=0,
                    handle_nan=0.0,
                    attention_configs=attention_configs, classifier_on_device=False)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_data = []

for train_idx, val_idx in skf.split(X_train, y_train):
    train_mask = np.zeros(len(y_train), dtype=bool)
    train_mask[train_idx] = True
    
    val_mask = np.zeros(len(y_train), dtype=bool)
    val_mask[val_idx] = True

    trn_feat_raw = framework.get_features(X_train, edge_index, edge_weights, mask=train_mask, is_training=True)
    trn_feat = np.concatenate([f.cpu() for f in trn_feat_raw], axis=1)
    y_trn = y_train[train_idx]

    val_feat_raw = framework.get_features(X_train, edge_index, edge_weights, mask=val_mask, is_training=False)
    val_feat = np.concatenate([f.cpu() for f in val_feat_raw], axis=1)
    y_val = y_train[val_idx]

    fold_data.append((trn_feat, y_trn, val_feat, y_val))

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
    fold_aurocs = []
    
    for trn_feat, y_trn, val_feat, y_val in fold_data:
        model = XGBClassifier(**params)
        model.fit(trn_feat, y_trn)
        
        pred_proba = model.predict_proba(val_feat)[:, 1]
        auroc = roc_auc_score(y_val, pred_proba)
        fold_aurocs.append(auroc)
        
    mean_auroc = np.mean(fold_aurocs)
    return {'loss': -mean_auroc, 'status': STATUS_OK}

trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)

best_hyperparameters = space_eval(space, best)
print(f"Best Hyperparameters: {best_hyperparameters}")

full_train_mask = np.ones(len(y_train), dtype=bool)
final_train_raw = framework.get_features(X_train, edge_index, edge_weights, mask=full_train_mask, is_training=True)
final_train_features = np.concatenate([f.cpu() for f in final_train_raw], axis=1)

full_test_mask = np.ones(len(y_test), dtype=bool)
final_test_raw = framework.get_features(X_test, edge_index_test, edge_weights_test, mask=full_test_mask, is_training=False)
final_test_features = np.concatenate([f.cpu() for f in final_test_raw], axis=1)

final_model = XGBClassifier(**best_hyperparameters)
final_model.fit(final_train_features, y_train)

final_pred_proba = final_model.predict_proba(final_test_features)[:, 1]
final_auroc = roc_auc_score(y_test, final_pred_proba)

print(f"Final Test AUROC: {final_auroc}")
#Final Test AUROC: 0.923420559112506