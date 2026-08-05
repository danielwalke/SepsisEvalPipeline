from LogisticRegression import LogisticRegressionModel
from RandomForest import RandomForestModel
from XGBoost import XGBoostModel
from sklearn.metrics import roc_auc_score
import os
from Data import Data
from SbcData import SbcData
import configparser

config = configparser.ConfigParser()
config_paths = ['/app/config/config.ini', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.ini')), 'config.ini']
config.read(config_paths)

def resolve_path(app_path, local_fallback):
    if os.path.exists('/app') and os.access('/app', os.W_OK):
        return app_path
    return os.path.abspath(local_fallback)

if __name__ == "__main__":
    include_sbc = config['PANEL'].getboolean('include_sbc', fallback=False)
    panel_name = config['PANEL']["panel_name"]

    input_dir = resolve_path(f"/app/input/{panel_name}", f"1_preprocess/data/preprocessed_data/{panel_name}")
    os.makedirs(input_dir, exist_ok=True)
    metric = roc_auc_score
    maximize_metric = True
    metric_pred_proba = True
    
    mimic_data = Data(input_dir)
    sbc_data = SbcData(input_dir) if "sbc_processed.csv" in os.listdir(input_dir) and include_sbc else None
    if sbc_data is not None:
        print("Found SBC data, training on SBC dataset")       
        lr_model = LogisticRegressionModel(data=sbc_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
        lr_model.evaluate(sbc_data, mimic_data)
        rf_model = RandomForestModel(data=sbc_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
        rf_model.evaluate(sbc_data, mimic_data)
        xgb_model = XGBoostModel(data=sbc_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
        xgb_model.evaluate(sbc_data, mimic_data)
    print("Training on MIMIC dataset")    
    
    lr_model = LogisticRegressionModel(data=mimic_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
    lr_model.evaluate(sbc_data, mimic_data)
    rf_model = RandomForestModel(data=mimic_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
    rf_model.evaluate(sbc_data, mimic_data)
    xgb_model = XGBoostModel(data=mimic_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
    xgb_model.evaluate(sbc_data, mimic_data)
    