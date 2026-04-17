import logging
from hyperopt import tpe, Trials, fmin, STATUS_OK, space_eval
from sklearn.metrics import roc_auc_score
import configparser
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import mlflow
import os
import pickle
import time

class BaseModel:
    def __init__(self, data, ModelClass, metric=roc_auc_score, maximize_metric=True, metric_pred_proba=True):
        self.data = data
        self.ModelClass = ModelClass
        self.metric = metric
        self.maximize_metric = maximize_metric
        self.metric_pred_proba = metric_pred_proba
        self.config = configparser.ConfigParser()
        files_read = self.config.read('/app/config/config.ini')
        self.seed = int(self.config['RANDOM'].get('seed', '42'))
        
        
        self.feature_set_name = self.config['PANEL']['panel_name']
        
        logging.basicConfig(filename='/app/output/training_logs.log',
                            filemode='a',
                            level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        mlflow.set_tracking_uri("http://host.docker.internal:5000")
        mlflow.set_experiment(f"evaluations_{self.feature_set_name}")
        mlflow.start_run(run_name=f"{self.ModelClass.__name__}_Baseline_{data.name}")
        mlflow.set_tag("feature_set", self.feature_set_name)
        mlflow.set_tag("model", self.ModelClass.__name__)
        mlflow.set_tag("approach", "Baseline")
    
    def tune_params(self, param_space, max_evals=20):
        exp_name = f"{self.data.name}_{self.feature_set_name}"
        hyperparam_start_time = time.time()
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
        hyperparam_end_time = time.time()
        self.logger.info(f"Best Hyperparameters: {best_params}")
        self.logger.info(f"Best Validation Score: {best_val_score}")
        self.logger.info(f"Hyperparameter Tuning Time: {hyperparam_end_time - hyperparam_start_time}")
        mlflow.log_metric(f"hyperparameter_tuning_time_seconds", hyperparam_end_time - hyperparam_start_time)
        return best_params

    def save_model(self, trained_model, best_params):
        exp_name = f"{self.data.name}_{self.feature_set_name}"
        os.makedirs(f"/app/models/{exp_name}", exist_ok=True)
        model_path = f"/app/models/{exp_name}/{self.ModelClass.__name__}.pkl"
        with open(model_path, 'wb') as f:            
            pickle.dump(trained_model, f)
        mlflow.log_artifact(model_path, artifact_path="models")
        mlflow.log_params(best_params)     
        print(f"Model saved to {model_path} and logged to MLflow.")

    def normalize_data(self, train_X, val_X, test_X):
        scaler = MinMaxScaler()
        train_X_scaled = scaler.fit_transform(train_X.copy())
        val_X_scaled = scaler.transform(val_X.copy())
        test_X_scaled = scaler.transform(test_X.copy())
        return train_X_scaled, val_X_scaled, test_X_scaled

    
    def get_score(self, best_params, seed = None):
        exp_name = f"{self.data.name}_{self.feature_set_name}"

        if seed is None:
            seed = self.seed
        
        ## Scaling
        normalize = best_params.pop("normalize", False)            
        if normalize:
            train_X, val_X, test_X = self.normalize_data(self.data.train_X, self.data.val_X, self.data.test_X)
        else:
            train_X, val_X, test_X = self.data.train_X, self.data.val_X, self.data.test_X

        ## Retrain with best params
        train_start_time = time.time()
        model = self.ModelClass(**best_params)
        model.fit(train_X, self.data.train_y)
        train_end_time = time.time()
        self.save_model(model, best_params)
        
        mlflow.log_metric(f"training_time_seconds", train_end_time - train_start_time)
        
        ## Inference and Scoring
        inference_test_start_time = time.time()
        if self.metric_pred_proba:
            test_preds = model.predict_proba(test_X)[:, 1]
        else:
            test_preds = model.predict(test_X)
        score = self.metric(self.data.test_y, test_preds)
        inference_test_end_time = time.time()
        mlflow.log_metric(f"{self.data.name}_TEST__inference_time_seconds", inference_test_end_time - inference_test_start_time)

        inference_val_start_time = time.time()
        if self.metric_pred_proba:
            val_preds = model.predict_proba(val_X)[:, 1]
        else:
            val_preds = model.predict(val_X)
        val_score = self.metric(self.data.val_y, val_preds)
        inference_val_end_time = time.time()
        mlflow.log_metric(f"{self.data.name}_VAL__inference_time_seconds", inference_val_end_time - inference_val_start_time)


        self.logger.info(f"Final Test Score: {score}")
        self.logger.info(f"Final Validation Score: {val_score}")
        if self.data.name == "SBC":
            ext_test_X = self.data.ext_test_X
            ext_test_y = self.data.ext_test_y
            if normalize:
                ext_test_X = scaler.transform(ext_test_X.copy())
            inference_ext_test_start_time = time.time()
            ext_preds = model.predict_proba(ext_test_X)[:, 1] if self.metric_pred_proba else model.predict(ext_test_X)
            inference_ext_test_end_time = time.time()
            ext_score = self.metric(ext_test_y, ext_preds)
            mlflow.log_metric(f"{self.data.name}_EXT_TEST__inference_time_seconds", inference_ext_test_end_time - inference_ext_test_start_time)
            self.logger.info(f"Final External Test Score: {ext_score}")
              
        if self.data.name == "SBC":
            mlflow.log_metric(f"{self.data.name}_TEST__AUROC", score)
            mlflow.log_metric(f"{self.data.name}_EXT_TEST__AUROC", ext_score)
            mlflow.log_metric(f"{self.data.name}_VAL__AUROC", val_score)
        else:
            mlflow.log_metric(f"{self.data.name}_TEST__AUROC", score)
            mlflow.log_metric(f"{self.data.name}_VAL__AUROC", val_score)
        mlflow.end_run()
        return score