from sklearn.ensemble import RandomForestClassifier
from BaseModel import BaseModel
import hyperopt as hp
import numpy as np

class RandomForestModel(BaseModel):
    def __init__(self, input_dir, metric, maximize_metric, metric_pred_proba):
        super().__init__(input_dir, RandomForestClassifier, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
    
    def evaluate(self):
        search_space = {
            "n_estimators": hp.hp.choice("n_estimators", [100, 200, 300, 400, 500]),
            "max_depth": hp.hp.choice("max_depth", [None, 10, 20, 30, 40, 50]),
            "min_samples_split": hp.hp.choice("min_samples_split", [2, 5, 10]),
            "min_samples_leaf": hp.hp.choice("min_samples_leaf", [1, 2, 4]),
            "bootstrap": hp.hp.choice("bootstrap", [True, False]),  
            "random_state": self.seed
        }
        best_params = self.tune_params(search_space, max_evals=50)
        test_score = self.get_score(best_params)
        return best_params, test_score