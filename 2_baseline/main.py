from LogisticRegression import LogisticRegressionModel
from RandomForest import RandomForestModel
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    input_dir = "/app/input"
    metric = roc_auc_score
    maximize_metric = True
    metric_pred_proba = True
    # rf_model = RandomForestModel(input_dir=input_dir, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
    # rf_model.evaluate()
    lr_model = LogisticRegressionModel(input_dir=input_dir, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
    lr_model.evaluate()
    
    