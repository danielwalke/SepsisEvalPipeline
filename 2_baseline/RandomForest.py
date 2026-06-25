from sklearn.ensemble import RandomForestClassifier
from BaseModel import BaseModel
import hyperopt as hp
import numpy as np

class RandomForestModel(BaseModel):
    def __init__(self, data, metric, maximize_metric, metric_pred_proba):
        super().__init__(data, RandomForestClassifier, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
    
    def evaluate(self, *test_data):
        search_space = {
            "n_estimators": hp.hp.choice("n_estimators", [100, 200, 300, 400, 500]),
            "max_depth": hp.hp.choice("max_depth", [None, 10, 20, 30, 40, 50]),
            "max_leaf_nodes": hp.hp.choice("max_leaf_nodes", [None, 50, 79, 100]),
            "min_samples_split": hp.hp.uniform("min_samples_split", 0.001, 0.01),
            "min_samples_leaf": hp.hp.uniform("min_samples_leaf", 0.0001, 0.01),
            "bootstrap": hp.hp.choice("bootstrap", [True, False]),  
            "class_weight": hp.hp.choice("class_weight", [
                None, 
                "balanced", 
                {0: 0.001, 1: 1}, 
                {0: 0.0015, 1: 1}, 
                {0: 0.002, 1: 1}, 
                {0: 0.0025, 1: 1}, 
                {0: 0.005, 1: 1}, 
                {0: 0.01, 1: 1}
            ])
        }
        best_params = self.tune_params(search_space)
        self.log_scores(best_params, *test_data)
        return best_params