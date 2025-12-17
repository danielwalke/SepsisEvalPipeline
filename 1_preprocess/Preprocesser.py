from constants.feature_names import LABEL_COLUMN_NAME, SEX_COLUMN_NAME, AGE_COLUMN_NAME
import pandas as pd
import os
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import configparser



nan_handlers = [
    "drop", #default
    "zero_fill",
    "mean_fill",
    "multi_imputer"
]

class Preprocesser:
    def __init__(self, data, path):
        self.raw_data = data
        self.pre_processed_data = None
        self.features = [AGE_COLUMN_NAME, SEX_COLUMN_NAME]
        self.path = path
        self.append_user_features()
        self.config = configparser.ConfigParser()
        files_read = self.config.read('./config.ini')
        self.nan_handler_type = self.config['PREPROCESSING'].get('imputation', 'drop')
        assert self.nan_handler_type in nan_handlers, f"nan_handler_type must be one of {nan_handlers}"
        
        
        
    def append_user_features(self):
        feature_codes = pd.read_csv(os.path.join(self.path, "features", "feature_codes.csv"), header=0)
        d_labitems = pd.read_csv(os.path.join(self.path, "extdata", "d_labitems.csv"), header=0)
        feature_mappings = pd.merge(feature_codes, d_labitems, on="itemid", how="left")
        self.features.extend(feature_mappings["label"].values.tolist())
    
    def nan_handler(self, data):
        if self.nan_handler_type == "drop":
            data = data.dropna(subset=self.features)
            
        elif self.nan_handler_type == "zero_fill":
            data.loc[:, self.features] = data[self.features].fillna(0)
            
        elif self.nan_handler_type == "mean_fill":
            for feature in self.features:
                if pd.api.types.is_numeric_dtype(data[feature]):
                    mean_value = data[feature].mean()
                    data.loc[:, feature] = data[feature].fillna(mean_value)
                else:
                    if not data[feature].mode().empty:
                        mode_value = data[feature].mode()[0]
                        data.loc[:, feature] = data[feature].fillna(mode_value)
                        
        elif self.nan_handler_type == "multi_imputer":
            numeric_features = [f for f in self.features if pd.api.types.is_numeric_dtype(data[f])]
            categorical_features = [f for f in self.features if not pd.api.types.is_numeric_dtype(data[f])]

            if numeric_features:
                imputer = IterativeImputer(max_iter=10, random_state=self.config['RANDOM'].getint('seed', 42))
                imputed_matrix = imputer.fit_transform(data[numeric_features])
                
                data[numeric_features] = pd.DataFrame(
                    imputed_matrix, 
                    columns=numeric_features, 
                    index=data.index
                )

            for feature in categorical_features:
                if not data[feature].mode().empty:
                    mode_value = data[feature].mode()[0]
                    data[feature] = data[feature].fillna(mode_value)

        return data
    
    
    def preprocess_data(self):
        data = self.raw_data.copy()
        unique_data = data.drop_duplicates(subset=["Id", "Center", "Time"], keep=False)
        non_icu_unique_data = unique_data.query("~(Sender.str.contains('ICU')) & ~(~SecToIcu.isnull() & SecToIcu < 0)",
                                                engine='python')
        first_non_icu_unique_data = non_icu_unique_data.query("Episode == 1 ", engine='python')
        complete_first_non_icu_unique_data = self.nan_handler(first_non_icu_unique_data)
        sirs_complete_first_non_icu_unique_data = complete_first_non_icu_unique_data.query("Diagnosis != 'SIRS'",
                                                                                           engine='python')
        sirs_complete_first_non_icu_unique_data = \
            sirs_complete_first_non_icu_unique_data.query("(Diagnosis == 'Control') | ((Diagnosis == 'Sepsis') & ("
                                                          "~TargetIcu.isnull() & "
                                                          "TargetIcu.str.contains('MICU')))",
                                                                                           engine='python')
        self.pre_processed_data = sirs_complete_first_non_icu_unique_data.copy()
        self.pre_processed_data['Label'] = self.pre_processed_data['Diagnosis']

        control_filter = (self.pre_processed_data["Diagnosis"] == 'Control') | \
                         ((self.pre_processed_data["SecToIcu"] > 3600 * 6) & (
                                     ~self.pre_processed_data["TargetIcu"].isnull() & self.pre_processed_data["TargetIcu"]
                                     .str.contains('MICU', na=False)))
        sepsis_filter = (self.pre_processed_data["Diagnosis"] == 'Sepsis') & \
                        (self.pre_processed_data["SecToIcu"] <= 3600 * 6) & \
                        (self.pre_processed_data["TargetIcu"].str.contains('MICU', na=False))
        self.pre_processed_data.loc[control_filter, "Label"] = "Control"
        self.pre_processed_data.loc[sepsis_filter, "Label"] = "Sepsis"

        self.pre_processed_data[SEX_COLUMN_NAME] = self.pre_processed_data[SEX_COLUMN_NAME].astype("category")
        self.pre_processed_data[SEX_COLUMN_NAME] = self.pre_processed_data[SEX_COLUMN_NAME].cat.codes

        self.control_data = self.pre_processed_data.loc[control_filter]
        self.sepsis_data = self.pre_processed_data.loc[sepsis_filter]
        self.resample_data() 

    def get_control_data(self):
        return self.control_data

    def get_sepsis_data(self):
        return self.sepsis_data

    def resample_data(self):
        self.pre_processed_data = self.pre_processed_data.sample(frac=1).reset_index()
        
    def get_data(self):
        return self.pre_processed_data
    
    def get_X(self):
        data = self.get_data()
        data[SEX_COLUMN_NAME] = data[SEX_COLUMN_NAME].astype("category")
        data[SEX_COLUMN_NAME] = data[SEX_COLUMN_NAME].cat.codes
        return data.loc[:, self.features].values
    
    def get_y(self):
        data = self.get_data()
        data[LABEL_COLUMN_NAME] = data[LABEL_COLUMN_NAME].astype('category')
        return (data.loc[:, LABEL_COLUMN_NAME].cat.codes).values