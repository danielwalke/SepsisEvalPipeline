from sklearn.linear_model import LogisticRegression
from BaseModel import BaseModel
import numpy as np
import hyperopt as hp

class LogisticRegressionModel(BaseModel):
    
    def __init__(self, input_dir, metric, maximize_metric, metric_pred_proba):
        super().__init__(input_dir, LogisticRegression, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
    
    def evaluate(self):
        search_space = {
            "C": hp.hp.loguniform("C", np.log(0.001), np.log(1000)),
            "penalty": hp.hp.choice("penalty", ["l2"]),
            "solver": hp.hp.choice("solver", ["lbfgs", "saga"]),
            "max_iter": hp.hp.choice("max_iter", [100, 200, 300, 400, 500]),
            "random_state": self.seed
        }
        best_params = self.tune_params(search_space, max_evals=50)
        test_score = self.get_score(best_params)
        return best_params, test_score