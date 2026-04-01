import logging
from hyperopt import tpe, Trials, fmin, STATUS_OK, space_eval
from sklearn.metrics import roc_auc_score
from Data import Data
from SbcData import SbcData
import configparser
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ResultsLogger import append_result

class BaseModel:
    def __init__(self, input_dir, ModelClass, metric=roc_auc_score, maximize_metric=True, metric_pred_proba=True):
        
        
        self.data = Data(input_dir) if "mimic_processed_train.csv" in input_dir else SbcData(input_dir)
        self.ModelClass = ModelClass
        self.metric = metric
        self.maximize_metric = maximize_metric
        self.metric_pred_proba = metric_pred_proba
        self.config = configparser.ConfigParser()
        files_read = self.config.read('/app/config/config.ini')
        self.seed = int(self.config['RANDOM'].get('seed', '42'))
        
        logging.basicConfig(filename='/app/output/training_logs.log',
                            filemode='a',
                            level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def tune_params(self, param_space, max_evals=50):
        self.logger.info(f"Model Class: {self.ModelClass.__name__}")
        self.logger.info(f"Hyperparameter Space: {param_space}")

        def objective(params):
            normalize = params.pop("normalize", False)
            
            if normalize:
                scaler = MinMaxScaler()
                train_X = scaler.fit_transform(self.data.train_X.copy())
                val_X = scaler.transform(self.data.val_X.copy())
            else:
                train_X = self.data.train_X
                val_X = self.data.val_X
            model = self.ModelClass(**params, n_jobs=10)
            model.fit(train_X, self.data.train_y)
            if self.metric_pred_proba:
                preds = model.predict_proba(val_X)[:, 1]
            else:
                preds = model.predict(val_X)
            score = self.metric(self.data.val_y, preds)
            if not self.maximize_metric:
                score = -score
            return {'loss': -score, 'status': STATUS_OK}
        trials = Trials()
        best = fmin(fn=objective,
                    space=param_space,
                    algo=tpe.suggest,
                    max_evals=max_evals,
                    rstate=np.random.default_rng(self.seed),
                    trials=trials)
        best_params = space_eval(param_space, best)
        best_val_score = -trials.best_trial['result']['loss']
        
        self.logger.info(f"Best Hyperparameters: {best_params}")
        self.logger.info(f"Best Validation Score: {best_val_score}")
        
        return best_params
    
    def get_score(self, best_params, seed = None):
        if seed is None:
            seed = self.seed
        normalize = best_params.pop("normalize", False)
            
        if normalize:
            scaler = MinMaxScaler()
            train_X = scaler.fit_transform(self.data.train_X.copy())
            test_X = scaler.transform(self.data.test_X.copy())
        else:
            train_X = self.data.train_X
            test_X = self.data.test_X
        model = self.ModelClass(**best_params)
        model.fit(train_X, self.data.train_y)
        if self.metric_pred_proba:
            preds = model.predict_proba(test_X)[:, 1]
        else:
            preds = model.predict(test_X)
        score = self.metric(self.data.test_y, preds)
        
        self.logger.info(f"Final Test Score: {score}")
        if self.data.name == "SBC":
            ext_test_X = self.data.ext_test_X
            ext_test_y = self.data.ext_test_y
            if normalize:
                ext_test_X = scaler.transform(ext_test_X.copy())
            ext_preds = model.predict_proba(ext_test_X)[:, 1] if self.metric_pred_proba else model.predict(ext_test_X)
            ext_score = self.metric(ext_test_y, ext_preds)
            self.logger.info(f"Final External Test Score: {ext_score}")
        append_result(
            dataset=self.data.name,
            features=self.data.train_X.columns.tolist(),
            model=self.ModelClass.__name__,
            approach="baseline",
            test_score=score,
            ext_test_score=ext_score if self.data.name == "SBC" else None,
            hyperparameters=best_params
        )
        return score