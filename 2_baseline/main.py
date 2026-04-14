from LogisticRegression import LogisticRegressionModel
from RandomForest import RandomForestModel
from sklearn.metrics import roc_auc_score
import os
from Data import Data
from SbcData import SbcData
import configparser

config = configparser.ConfigParser()
config.read('/app/config/config.ini')

if __name__ == "__main__":
    
    input_dir = "/app/input"
    metric = roc_auc_score
    maximize_metric = True
    metric_pred_proba = True
    include_sbc = config['PANEL'].getboolean('include_sbc', fallback=False)
    if "sbc_processed.csv" in os.listdir(input_dir) and include_sbc:
        print("Found SBC data, training on SBC dataset")
        sbc_data = SbcData(input_dir)
        rf_model = RandomForestModel(data=sbc_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
        rf_model.evaluate()
        lr_model = LogisticRegressionModel(data=sbc_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
        lr_model.evaluate()
    print("Training on MIMIC dataset")
    mimic_data = Data(input_dir)
    rf_model = RandomForestModel(data=mimic_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
    rf_model.evaluate()
    lr_model = LogisticRegressionModel(data=mimic_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
    lr_model.evaluate()
    
    