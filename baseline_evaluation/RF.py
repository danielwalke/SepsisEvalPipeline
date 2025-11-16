from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from constants.feature_names import FEATURES

def get_best_hyperparameters(X_train, y_train, X_val, y_val):
    def objective(params):
        rf = RandomForestClassifier(**params, random_state=42)
        rf.fit(X_train, y_train)
        val_probas = rf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probas)
        return -val_auc  # Minimize negative AUC

    space = {
        'n_estimators': hyperopt.hp.choice('n_estimators', [50, 100, 200]),
        'max_depth': hyperopt.hp.choice('max_depth', [None, 10, 20, 30]),
        'min_samples_split': hyperopt.hp.choice('min_samples_split', [2, 5, 10]),
        'min_samples_leaf': hyperopt.hp.choice('min_samples_leaf', [1, 2, 4]),
        'bootstrap': hyperopt.hp.choice('bootstrap', [True, False])
    }

    best = hyperopt.fmin(
        fn=objective,
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=20
    )
    return space_eval(space, best)

if __name__ == "__main__":
    mimic_df = pd.read_csv("./data/graph_data/mimic_sorted_processed.csv", header=0)
    labels = mimic_df["Label"].astype('category').cat.codes.astype(float)
    train_ids, test_ids = train_test_split(mimic_df["Id"].unique(), test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)
    
    train_mask = mimic_df["Id"].isin(train_ids)
    val_mask = mimic_df["Id"].isin(val_ids)
    test_mask = mimic_df["Id"].isin(test_ids)
    
    X_train = mimic_df.loc[train_mask, FEATURES]
    y_train = labels[train_mask]
    X_val = mimic_df.loc[val_mask, FEATURES]
    y_val = labels[val_mask]
    X_test = mimic_df.loc[test_mask, FEATURES]
    y_test = labels[test_mask]

    best_hyperparams = get_best_hyperparameters(X_train, y_train, X_val, y_val)
    print("Best Hyperparameters:", best_hyperparams)
    
    rf = RandomForestClassifier(**best_hyperparams, random_state=42)
    rf.fit(X_train, y_train)
    
    val_probas = rf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probas)
    print(f"Validation AUC: {val_auc:.4f}")
    
    test_probas = rf.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probas)
    print(f"Test AUC: {test_auc:.4f}") #{'bootstrap': False, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50} 
    # 0.8026