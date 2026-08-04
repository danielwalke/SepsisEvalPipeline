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

input_dir = '1_preprocess/data/preprocessed_data/CBC'
metric = roc_auc_score
maximize_metric = True
metric_pred_proba = True

mimic_data = Data(input_dir)
mimic_data.feature_set_name = 'CBC'

sbc_data = SbcData(input_dir)
sbc_data.feature_set_name = 'CBC'

print("\n=======================================================")
print("  TRAINING BASELINE MODELS ON SBC DATASET (CBC PANEL)")
print("=======================================================")

# 1. Logistic Regression on SBC
print("\n--- [1/3] LogisticRegression (SBC) ---")
lr_sbc = LogisticRegressionModel(data=sbc_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
lr_sbc.evaluate(sbc_data, mimic_data)

# 2. Random Forest on SBC
print("\n--- [2/3] RandomForest (SBC) ---")
rf_sbc = RandomForestModel(data=sbc_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
rf_sbc.evaluate(sbc_data, mimic_data)

# 3. XGBoost on SBC
print("\n--- [3/3] XGBoost (SBC) ---")
xgb_sbc = XGBoostModel(data=sbc_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
xgb_sbc.evaluate(sbc_data, mimic_data)

print("\n=======================================================")
print("  TRAINING BASELINE MODELS ON MIMIC DATASET (CBC PANEL)")
print("=======================================================")

# 4. Logistic Regression on MIMIC (evaluating on both SBC & MIMIC)
print("\n--- [1/3] LogisticRegression (MIMIC -> SBC+MIMIC) ---")
lr_mimic = LogisticRegressionModel(data=mimic_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
lr_mimic.evaluate(sbc_data, mimic_data)

# 5. Random Forest on MIMIC (evaluating on both SBC & MIMIC)
print("\n--- [2/3] RandomForest (MIMIC -> SBC+MIMIC) ---")
rf_mimic = RandomForestModel(data=mimic_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
rf_mimic.evaluate(sbc_data, mimic_data)

# 6. XGBoost on MIMIC (evaluating on both SBC & MIMIC)
print("\n--- [3/3] XGBoost (MIMIC -> SBC+MIMIC) ---")
xgb_mimic = XGBoostModel(data=mimic_data, metric=metric, maximize_metric=maximize_metric, metric_pred_proba=metric_pred_proba)
xgb_mimic.evaluate(sbc_data, mimic_data)

print("\n=======================================================")
print(" CBC BASELINE EXPERIMENTS (SBC & MIMIC) COMPLETED!")
print("=======================================================")
