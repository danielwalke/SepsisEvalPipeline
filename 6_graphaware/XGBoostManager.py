import xgboost as xgb
import numpy as np
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from Neo4jDataloader import Neo4jDataIter

class XGBoostManager:
    def __init__(self, train_label_name, val_label_name, batch_size=100000):
        self.batch_size = batch_size
        self.train_label_name = train_label_name
        self.val_label_name = val_label_name

    def train_model_iterator(self, connector, params, node_label, condition, num_trees, framework, target_ids=None):
        iterator = Neo4jDataIter(connector, node_label, condition, framework, self.batch_size, target_ids)
        dtrain = xgb.QuantileDMatrix(iterator)
        
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=num_trees,
            verbose_eval=False
        )
        
        return model

    def evaluate_model(self, connector, model, node_label, condition, framework, target_ids=None):
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

    def optimize_hyperparams(self, connector, train_ids, val_ids, framework):
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
            
            model = self.train_model_iterator(connector, xgb_params, self.train_label_name, train_cond, int(num_trees), framework, train_ids)
            auroc = self.evaluate_model(connector, model, self.val_label_name, val_cond, framework, val_ids)
            
            return {'loss': -auroc, 'status': STATUS_OK}

        trials = Trials()
        best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=80, trials=trials, verbose=1)
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

    def save_model(self, model, filepath):
        model.save_model(filepath)

    def load_model(self, filepath):
        model = xgb.Booster()
        model.load_model(filepath)
        return model