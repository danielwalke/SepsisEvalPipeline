from constants.feature_names import FEATURES_IN_TABLE, FEATURES, LABEL_COLUMN_NAME, SEX_COLUMN_NAME, SEX_CATEGORY_COLUMN_NAME

class Preprocesser:
    def __init__(self, data):
        self.raw_data = data
        self.pre_processed_data = None
    
    def preprocess_data(self):
        data = self.raw_data.copy()
        unique_data = data.drop_duplicates(subset=["Id", "Center", "Time"], keep=False)
        non_icu_unique_data = unique_data.query("~(Sender.str.contains('ICU')) & ~(~SecToIcu.isnull() & SecToIcu < 0)",
                                                engine='python')
        first_non_icu_unique_data = non_icu_unique_data.query("Episode == 1 ", engine='python')
        complete_first_non_icu_unique_data = first_non_icu_unique_data.query("~(" + " | ".join([i + ".isnull()" for i in FEATURES_IN_TABLE]) +")", engine='python') ## filters all rows with an empty feature value
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
        self.pre_processed_data[SEX_CATEGORY_COLUMN_NAME] = self.pre_processed_data[SEX_COLUMN_NAME].cat.codes

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
        data[SEX_CATEGORY_COLUMN_NAME] = data[SEX_COLUMN_NAME].cat.codes
        return data.loc[:, FEATURES].values
    
    def get_y(self):
        data = self.get_data()
        data[LABEL_COLUMN_NAME] = data[LABEL_COLUMN_NAME].astype('category')
        return (data.loc[:, LABEL_COLUMN_NAME].cat.codes).values