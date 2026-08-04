import os
import sys

sys.path.insert(0, os.path.abspath('2_baseline'))

from LogisticRegression import LogisticRegressionModel
from RandomForest import RandomForestModel
from XGBoost import XGBoostModel
from sklearn.metrics import roc_auc_score
from Data import Data
from SbcData import SbcData

os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'

PANELS = ["CBC", "CBC_BMP", "CBC_HIL", "CBC_BMP_HIL"]
DATA_ROOT = "1_preprocess/data/preprocessed_data"

metric = roc_auc_score
maximize_metric = True
metric_pred_proba = True

for panel in PANELS:
    input_dir = os.path.join(DATA_ROOT, panel)
    if not os.path.exists(input_dir):
        print(f"Skipping {panel}: directory {input_dir} does not exist.")
        continue
    
    print(f"\n=======================================================")
    print(f"       PROCESSING PANEL: {panel}")
    print(f"=======================================================")
    
    mimic_data = Data(input_dir)
    mimic_data.feature_set_name = panel
    
    sbc_data = SbcData(input_dir) if ("sbc_processed.csv" in os.listdir(input_dir) and os.path.exists(os.path.join(input_dir, "sbc_processed_validation.csv"))) else None
    
    # 1. Logistic Regression
    print(f"\n--- [1/3] LogisticRegression for {panel} ---")
    try:
        lr = LogisticRegressionModel(data=mimic_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
        lr.evaluate(mimic_data)
        print(f"SUCCESS: LogisticRegression for {panel}")
    except Exception as e:
        print(f"ERROR running LogisticRegression on {panel}: {e}")

    # 2. Random Forest
    print(f"\n--- [2/3] RandomForest for {panel} ---")
    try:
        rf = RandomForestModel(data=mimic_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
        rf.evaluate(mimic_data)
        print(f"SUCCESS: RandomForest for {panel}")
    except Exception as e:
        print(f"ERROR running RandomForest on {panel}: {e}")

    # 3. XGBoost
    print(f"\n--- [3/3] XGBoost for {panel} ---")
    try:
        xgb_m = XGBoostModel(data=mimic_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
        xgb_m.evaluate(mimic_data)
        print(f"SUCCESS: XGBoost for {panel}")
    except Exception as e:
        print(f"ERROR running XGBoost on {panel}: {e}")

print("\n=======================================================")
print(" ALL BASELINE EXPERIMENTS COMPLETED & LOGGED TO MLFLOW!")
print("=======================================================")
