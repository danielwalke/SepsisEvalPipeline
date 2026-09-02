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
        if os.path.exists("config.ini"):
            self.config.read("config.ini")
        elif os.path.exists("/app/config/config.ini"):
            self.config.read("/app/config/config.ini")
        
        self.seed = int(self.config['RANDOM'].get('seed', '42')) if 'RANDOM' in self.config else 42
        
        self.feature_set_name = getattr(data, "feature_set_name", self.config['PANEL'].get('panel_name', 'CBC') if 'PANEL' in self.config else 'CBC')
        
        log_path = "2_baseline/training_logs.log" if os.path.exists("2_baseline") else "/app/output/training_logs.log"
        logging.basicConfig(filename=log_path,
                            filemode='a',
                            level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        if mlflow.active_run():
            mlflow.end_run()
            
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            # Check if mlflow-server hostname resolves, else use localhost
            try:
                import socket
                socket.gethostbyname("mlflow-server")
                tracking_uri = "http://mlflow-server:5000"
            except Exception:
                tracking_uri = "http://localhost:5000"
        mlflow.set_tracking_uri(tracking_uri)
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
            scaler = None
            
            if normalize:
                scaler = MinMaxScaler()
                train_X = scaler.fit_transform(self.data.train_X.copy())
                val_X = scaler.transform(self.data.val_X.copy())
            else:
                train_X = self.data.train_X
                val_X = self.data.val_X
            try:
                model = self.ModelClass(**params, n_jobs=10)
            except TypeError:
                model = self.ModelClass(**params)
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
        os.makedirs(f"2_baseline/models/{exp_name}", exist_ok=True)
        model_path = f"2_baseline/models/{exp_name}/{self.ModelClass.__name__}.pkl"
        try:
            with open(model_path, 'wb') as f:            
                pickle.dump(trained_model, f)
        except PermissionError:
            fallback_dir = f"/tmp/models/{exp_name}"
            os.makedirs(fallback_dir, exist_ok=True)
            model_path = f"{fallback_dir}/{self.ModelClass.__name__}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(trained_model, f)
            print(f"Saved to fallback path {model_path} due to permission limits.")
        try:
            mlflow.log_artifact(model_path, artifact_path="models")
        except Exception as e:
            print(f"Artifact logging warning: {e}")
        # Convert params values to string if dict
        str_params = {k: str(v) if isinstance(v, dict) else v for k, v in best_params.items()}
        mlflow.log_params(str_params)     
        print(f"Model saved to {model_path} and logged to MLflow.")

    def normalize_data(self, train_X, val_X, test_X):
        scaler = MinMaxScaler()
        train_X_scaled = scaler.fit_transform(train_X.copy())
        return scaler, train_X_scaled

    
    def log_scores(self, best_params, *test_datasets, seed = None):
        try:
            exp_name = f"{self.data.name}_{self.feature_set_name}"

            if seed is None:
                seed = self.seed
            
            ## Scaling
            normalize = best_params.pop("normalize", False)            
            ## Retrain with best params
            train_start_time = time.time()
            try:
                model = self.ModelClass(**best_params, n_jobs=10)
            except TypeError:
                model = self.ModelClass(**best_params)
            if normalize:
                from sklearn.pipeline import Pipeline
                scaler = MinMaxScaler()
                full_model = Pipeline([('scaler', scaler), ('model', model)])
                full_model.fit(self.data.train_X, self.data.train_y)
            else:
                full_model = model
                full_model.fit(self.data.train_X, self.data.train_y)
            train_end_time = time.time()
            self.save_model(full_model, best_params)
            model = full_model
            
            mlflow.log_metric(f"training_time_seconds", train_end_time - train_start_time)
            
            ## Inference and Scoring
            for test_data in test_datasets:
                if test_data is None: continue
                for test_data_set in test_data.test_data_containers:
                    test_name, test_X, test_y = test_data_set
                    test_X = test_X.copy()
                    print(f"Evaluating on test dataset: {test_name}")

                    inference_test_start_time = time.time()
                    if self.metric_pred_proba:
                        test_preds = model.predict_proba(test_X)[:, 1]
                    else:
                        test_preds = model.predict(test_X)
                    score = self.metric(test_y, test_preds)
                    inference_test_end_time = time.time()
                    mlflow.log_metric(f"{test_name}__inference_time_seconds", inference_test_end_time - inference_test_start_time)
                    mlflow.log_metric(f"{test_name}__AUROC", score)
                    print(f"Test Score on {test_name}: {score} with inference time {inference_test_end_time - inference_test_start_time} seconds")

            return score
        finally:
            if mlflow.active_run():
                mlflow.end_run()