import os
from Preprocesser import Preprocesser
from utils.CountFunctions import count_cbc_cases, count_cbc
import pandas as pd
import configparser

class PreprocessWrapper:
    def __init__(self, feature_input_dir_path, extdata_input_dir_path, sbc_data=None, mimic_data=None, print_logs=False):
        self.config = configparser.ConfigParser()
        self.config.read('config/config.ini')
        self.include_sbc = self.config['PANEL'].getboolean('include_sbc', fallback=False)
        self.feature_input_dir_path = feature_input_dir_path
        self.extdata_input_dir_path = extdata_input_dir_path

        if mimic_data is not None:
            mimic_validation_data = mimic_data.query(
                "Center == 'MIMIC-IV' & Set == 'Validation'"
            )
            self.mimic = Preprocesser(feature_input_dir_path, extdata_input_dir_path, mimic_validation_data)
            self.mimic.preprocess_data()
            if print_logs:
                print(20 * "$")
                print("MIMIC: ")
                print(
                    f"Controls: {self.mimic.get_control_data().shape[0]},"
                    f" Sepsis: {self.mimic.get_sepsis_data().shape[0]}"
                )
                print(
                    f"Assessable data are {count_cbc_cases(self.mimic.get_data())} cases "
                    f"and {count_cbc(self.mimic.get_data())} CBCs"
                )
                print(
                    f"Control data are {count_cbc_cases(self.mimic.get_control_data())} cases "
                    f"and {count_cbc(self.mimic.get_control_data())} CBCs"
                )
                print(
                    f"Sepsis data are {count_cbc_cases(self.mimic.get_sepsis_data())} cases "
                    f"and {count_cbc(self.mimic.get_sepsis_data())} CBCs"
                )
        if self.include_sbc and sbc_data is not None:
            sbc_training_data = sbc_data.query(
                "Center == 'Leipzig' & Set == 'Training'"
            )
            self.sbc = Preprocesser(feature_input_dir_path, extdata_input_dir_path, sbc_training_data)
            self.sbc.preprocess_data()
            sbc_validation_data = sbc_data.query(
                "Center == 'Leipzig' & Set == 'Validation'"
            )
            self.sbc_validation = Preprocesser(feature_input_dir_path, extdata_input_dir_path, sbc_validation_data)
            self.sbc_validation.preprocess_data()
            sbc_ext_validation_data = sbc_data.query(
                "Center == 'Greifswald' & Set == 'Validation'"
            )
            self.sbc_ext_validation = Preprocesser(feature_input_dir_path, extdata_input_dir_path, sbc_ext_validation_data)
            self.sbc_ext_validation.preprocess_data()
            if not print_logs:
                return
            print(20 * "#")
            print("SBC Training: ")
            print(
                f"Controls: {self.sbc.get_control_data().shape[0]},"
                f" Sepsis: {self.sbc.get_sepsis_data().shape[0]}"
            )
            print(
                f"Assessable data are {count_cbc_cases(self.sbc.get_data())} cases "
                f"and {count_cbc(self.sbc.get_data())} CBCs"
            )
            print(
                f"Control data are {count_cbc_cases(self.sbc.get_control_data())} cases "
                f"and {count_cbc(self.sbc.get_control_data())} CBCs"
            )
            print(
                f"Sepsis data are {count_cbc_cases(self.sbc.get_sepsis_data())} cases "
                f"and {count_cbc(self.sbc.get_sepsis_data())} CBCs"
            )
            print(20 * "#")
            print("SBC Validation: ")
            print(
                f"Controls: {self.sbc_validation.get_control_data().shape[0]},"
                f" Sepsis: {self.sbc_validation.get_sepsis_data().shape[0]}"
            )
            print(
                f"Assessable data are {count_cbc_cases(self.sbc_validation.get_data())} cases "
                f"and {count_cbc(self.sbc_validation.get_data())} CBCs"
            )
            print(
                f"Control data are {count_cbc_cases(self.sbc_validation.get_control_data())} cases "
                f"and {count_cbc(self.sbc_validation.get_control_data())} CBCs"
            )
            print(
                f"Sepsis data are {count_cbc_cases(self.sbc_validation.get_sepsis_data())} cases "
                f"and {count_cbc(self.sbc_validation.get_sepsis_data())} CBCs"
            )
            print(20 * "#")
            print("SBC External Validation: ")
            print(
                f"Controls: {self.sbc_ext_validation.get_control_data().shape[0]},"
                f" Sepsis: {self.sbc_ext_validation.get_sepsis_data().shape[0]}"
            )
            print(
                f"Assessable data are {count_cbc_cases(self.sbc_ext_validation.get_data())} cases "
                f"and {count_cbc(self.sbc_ext_validation.get_data())} CBCs"
            )
            print(
                f"Control data are {count_cbc_cases(self.sbc_ext_validation.get_control_data())} cases "
                f"and {count_cbc(self.sbc_ext_validation.get_control_data())} CBCs"
            )
            print(
                f"Sepsis data are {count_cbc_cases(self.sbc_ext_validation.get_sepsis_data())} cases "
                f"and {count_cbc(self.sbc_ext_validation.get_sepsis_data())} CBCs"
            )

    def get_X_mimic(self):
        return self.mimic.get_X()

    def get_y_mimic(self):
        return self.mimic.get_y()

    def get_mimic_data(self):
        return self.mimic.get_data()

    def split_mimic_data_time_based(self):
        mimic_data = self.mimic.get_data()
        mimic_data["anchor_year_group"] = mimic_data["anchor_year_group"].str.split("-").str[0].astype(int)
        mimic_data["anchor_year_group"] = pd.to_datetime(mimic_data["anchor_year_group"], format='%Y')
        mimic_data = mimic_data.sort_values(by="anchor_year_group").reset_index(drop=True)
        
        val_split_time = pd.to_datetime("2012-01-01 00:00:00")
        split_time = pd.to_datetime("2015-01-01 00:00:00")
        
        train_data = mimic_data[mimic_data["anchor_year_group"] < val_split_time]
        val_data = mimic_data[(mimic_data["anchor_year_group"] >= val_split_time) & (mimic_data["anchor_year_group"] < split_time)]
        test_data = mimic_data[mimic_data["anchor_year_group"] >= split_time]
        return train_data, val_data, test_data

    def write_mimic_processed_data(self, path):
        train_data, val_data, test_data = self.split_mimic_data_time_based()

        features = train_data.filter(regex="^f__").columns.tolist()
        with open(os.path.join(self.feature_input_dir_path, "feature_names.txt"), "w") as f:
            print(f"Features: {features}")
            f.write(",".join(features))
        
        train_data.to_csv(path.replace(".csv", "_train.csv"), index=False)
        val_data.to_csv(path.replace(".csv", "_val.csv"), index=False)
        test_data.to_csv(path.replace(".csv", "_test.csv"), index=False)
        print(f"Wrote train data with {train_data.shape[0]} rows to {path.replace('.csv', '_train.csv')}")
        print(f"Wrote validation data with {val_data.shape[0]} rows to {path.replace('.csv', '_val.csv')}")
        print(f"Wrote test data with {test_data.shape[0]} rows to {path.replace('.csv', '_test.csv')}")

    def write_sbc_processed_data(self, path):
        if not self.include_sbc:
            print("SBC data inclusion is disabled in the config. Skipping writing SBC processed data.")
            return
        train_data = self.sbc.get_data()
        features = train_data.filter(regex="^f__").columns.tolist()
        with open(os.path.join(self.feature_input_dir_path, "sbc_feature_names.txt"), "w") as f:
            print(f"Features: {features}")
            f.write(",".join(features))

        train_data.to_csv(path, index=False)
        self.sbc_validation.get_data().to_csv(
            path.replace(".csv", "_validation.csv"), index=False
        )
        self.sbc_ext_validation.get_data().to_csv(
            path.replace(".csv", "_ext_validation.csv"), index=False
        )

def process_mimic_data(input_dir_path, feature_input_dir_path, extdata_input_dir_path):
    mimic_data = pd.read_csv(
       os.path.join(input_dir_path, "mimic_processed.csv"), header=0
    )
    preprocess_wrapper = PreprocessWrapper(feature_input_dir_path, extdata_input_dir_path,
       mimic_data=mimic_data, print_logs=True
    )
    preprocess_wrapper.write_mimic_processed_data(
       os.path.join(output_dir_path, "mimic_processed.csv")
    )

def process_sbc_data(feature_input_dir_path, extdata_input_dir_path):
    sbc_data = pd.read_csv(
        os.path.join(input_dir_path, "sbc_processed.csv"), header=0
    )
    preprocess_wrapper = PreprocessWrapper(feature_input_dir_path, extdata_input_dir_path,
        sbc_data=sbc_data, print_logs=True
    )
    preprocess_wrapper.write_sbc_processed_data(
        os.path.join(output_dir_path, "sbc_processed.csv")
    )

if __name__ == "__main__":
    print(os.getcwd())

    input_dir_path = "/app/input" 
    feature_input_dir_path = "/app/features"
    output_dir_path = "/app/output" 
    extdata_input_dir_path = "/app/extdata"
    
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)    
        
    mimic_path = os.path.join(input_dir_path, "mimic_processed.csv")
    if os.path.exists(mimic_path):
        print(f"Found {mimic_path}, processing...")
        process_mimic_data(input_dir_path, feature_input_dir_path, extdata_input_dir_path)
    else:
        print(f"Could not find {mimic_path}")
    
    sbc_path = os.path.join(input_dir_path, "sbc_processed.csv")
    if os.path.exists(sbc_path):
        print(f"Found {sbc_path}, processing...")
        process_sbc_data(input_dir_path, feature_input_dir_path, extdata_input_dir_path)
    else:
        print(f"Could not find {sbc_path}")