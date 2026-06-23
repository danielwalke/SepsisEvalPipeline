import time
import xgboost as xgb
import numpy as np
import scipy.sparse as sp
from functools import wraps
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from GraphDataloader import GraphDataLoader

def timeit(func):
    """Decorator to track and log the execution time of a function."""
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"[Timer] Function '{func.__name__}' executed in {total_time:.4f} seconds")
        return result
    return timeit_wrapper

class XGBoostManager:
    def __init__(self, train_label_name, val_label_name, batch_size=100000):
        self.batch_size = batch_size
        self.train_label_name = train_label_name
        self.val_label_name = val_label_name

    @timeit
    def _fetch_all_data(self, connector, node_label, condition, framework, target_ids=None):
        """Helper method to accumulate all data from the connector into a single block."""
        skip = 0
        all_X = []
        all_labels = []
        
        while True:
            X_batch, y_batch = connector.fetch_data_batch(
                node_label, condition, skip, self.batch_size, framework, target_ids
            )
            
            if len(y_batch) == 0:
                break
                
            all_X.append(X_batch)
            all_labels.append(y_batch)
            skip += self.batch_size
            
        if len(all_labels) == 0:
            return None, None
            
        # Efficiently concatenate based on data type
        y_full = np.concatenate(all_labels) if isinstance(all_labels[0], np.ndarray) else np.array(all_labels)
        
        if sp.issparse(all_X[0]):
            X_full = sp.vstack(all_X)
        elif isinstance(all_X[0], np.ndarray):
            X_full = np.vstack(all_X)
        else:
            X_full = np.array(all_X)
            
        return X_full, y_full

    @timeit
    def train_model_iterator(self, connector, params, node_label, condition, num_trees, framework, target_ids=None):
        iterator = GraphDataLoader(connector, node_label, condition, framework, self.batch_size, target_ids)
        dtrain = xgb.QuantileDMatrix(iterator)
        
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=num_trees,
            verbose_eval=False
        )
        
        return model
    
    @timeit
    def train_model_full_batch(self, connector, params, node_label, condition, num_trees, framework, target_ids=None):
        """Full-batch approach (Faster convergence/exact gradients, higher memory usage)."""
        X_train, y_train = self._fetch_all_data(connector, node_label, condition, framework, target_ids)
        
        if X_train is None:
            raise ValueError("No training data fetched from the connector.")
            
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=num_trees,
            verbose_eval=False
        )
        return model

    @timeit
    def evaluate_model_mini_batch(self, connector, model, node_label, condition, framework, target_ids=None):
        """Mini-batch inference approach."""
        skip = 0
        all_preds = []
        all_labels = []
        
        while True:
            X_test, y_test = connector.fetch_data_batch(node_label, condition, skip, self.batch_size, framework, target_ids)
            
            if len(y_test) == 0:
                break
                
            dtest = xgb.DMatrix(X_test)
            preds = model.predict(dtest)
            
            all_preds.extend(preds)
            all_labels.extend(y_test)
            skip += self.batch_size
            
        if len(all_labels) == 0:
            return 0.0
            
        return roc_auc_score(all_labels, all_preds)

    @timeit
    def evaluate_model_full_batch(self, connector, model, node_label, condition, framework, target_ids=None):
        """Full-batch inference approach."""
        X_test, y_test = self._fetch_all_data(connector, node_label, condition, framework, target_ids)
        
        if X_test is None:
            return 0.0
            
        dtest = xgb.DMatrix(X_test)
        preds = model.predict(dtest)
        
        return roc_auc_score(y_test, preds)

    @timeit
    def optimize_hyperparams(self, connector, train_cond, val_cond, train_ids, val_ids, framework, max_evals=80, mode='mini_batch'):
        space = {
            'alpha': hp.uniform('alpha', 7.5, 11.5),
            'booster': 'gbtree',
            'colsample_bytree': hp.uniform('colsample_bytree', 0.8, 1.0),
            'gamma': hp.uniform('gamma', 0.6, 1.0),
            'lambda': hp.loguniform('lambda', np.log(0.005), np.log(0.02)),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.1), np.log(0.25)),
            'max_depth': hp.choice('max_depth', [2, 3, 4]),
            'min_child_weight': hp.uniform('min_child_weight', 2.0, 4.5),
            'n_estimators': hp.choice('n_estimators', [125, 150, 175]),
            'n_jobs': 3,
            'objective': 'binary:logistic',
            'random_state': 42,
            'scale_pos_weight': 1,
            'subsample': hp.uniform('subsample', 0.75, 0.95)
        }
        
        def objective(params):
            num_trees = params.get('n_estimators', 150)
            
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
            if mode == 'full_batch':
                model = self.train_model_full_batch(connector, xgb_params, self.train_label_name, train_cond, int(num_trees), framework, train_ids)
                auroc = self.evaluate_model_full_batch(connector, model, self.val_label_name, val_cond, framework, val_ids)
            else:
                model = self.train_model_iterator(connector, xgb_params, self.train_label_name, train_cond, int(num_trees), framework, train_ids)
                auroc = self.evaluate_model_mini_batch(connector, model, self.val_label_name, val_cond, framework, val_ids)
            return {'loss': -auroc, 'status': STATUS_OK}

        trials = Trials()
        best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=1)
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
        return final_params

    @timeit
    def save_model(self, model, filepath):
        model.save_model(filepath)

    @timeit
    def load_model(self, filepath):
        model = xgb.Booster()
        model.load_model(filepath)
        return model