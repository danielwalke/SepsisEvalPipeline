from xgboost import XGBClassifier
from BaseModel import BaseModel
import hyperopt as hp
import numpy as np

class XGBoostModel(BaseModel):
    def __init__(self, data, metric, maximize_metric, metric_pred_proba):
        # Pass XGBClassifier to the BaseModel just like RandomForestClassifier
        super().__init__(data, XGBClassifier, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
    
    def evaluate(self, *test_data):
        # XGBoost specific search space
        search_space = {
            "n_estimators": hp.hp.choice("n_estimators", [100, 200, 300, 400, 500]),
            "max_depth": hp.hp.choice("max_depth", [3, 5, 7, 9, 12, 15]), # Generally kept lower in XGBoost than RF
            "learning_rate": hp.hp.uniform("learning_rate", 0.01, 0.3),   # Step size shrinkage
            "subsample": hp.hp.uniform("subsample", 0.5, 1.0),            # Equivalent to bootstrap/bagging fraction
            "colsample_bytree": hp.hp.uniform("colsample_bytree", 0.5, 1.0), # Fraction of features per tree
            "min_child_weight": hp.hp.choice("min_child_weight", [1, 3, 5, 7]), # Similar to min_samples_leaf
            "gamma": hp.hp.uniform("gamma", 0.0, 0.5),                    # Minimum loss reduction for a split
        }
        
        best_params = self.tune_params(search_space)
        self.log_scores(best_params, *test_data)
        
        return best_params