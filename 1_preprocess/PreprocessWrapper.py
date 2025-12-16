import os
from Preprocesser import Preprocesser
from utils.CountFunctions import count_cbc_cases, count_cbc
import pandas as pd


class PreprocessWrapper:
    def __init__(self, sbc_data=None, mimic_data=None, print_logs=False, path=None):
        if mimic_data is not None:
            mimic_validation_data = mimic_data.query(
                "Center == 'MIMIC-IV' & Set == 'Validation'"
            )
            self.mimic = Preprocesser(mimic_validation_data, path=path)
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
        if sbc_data is not None:
            sbc_training_data = sbc_data.query(
                "Center == 'Leipzig' & Set == 'Training'"
            )
            self.sbc = Preprocesser(sbc_training_data, path=path)
            self.sbc.preprocess_data()
            sbc_validation_data = sbc_data.query(
                "Center == 'Leipzig' & Set == 'Validation'"
            )
            self.sbc_validation = Preprocesser(sbc_validation_data, path=path)
            self.sbc_validation.preprocess_data()
            sbc_ext_validation_data = sbc_data.query(
                "Center == 'Greifswald' & Set == 'Validation'"
            )
            self.sbc_ext_validation = Preprocesser(sbc_ext_validation_data, path=path)
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
        
        # Splits such that training data ca. 64%, validation data ca. 20% and test data ca. 16%
        val_split_time = pd.to_datetime("2012-01-01 00:00:00")
        split_time = pd.to_datetime("2015-01-01 00:00:00")
        
        train_data = mimic_data[mimic_data["anchor_year_group"] < val_split_time]
        val_data = mimic_data[(mimic_data["anchor_year_group"] >= val_split_time) & (mimic_data["anchor_year_group"] < split_time)]
        test_data = mimic_data[mimic_data["anchor_year_group"] >= split_time]
        return train_data, val_data, test_data

    def write_mimic_processed_data(self, path):
        train_data, val_data, test_data = self.split_mimic_data_time_based()
        train_data.to_csv(path.replace(".csv", "_train.csv"), index=False)
        val_data.to_csv(path.replace(".csv", "_val.csv"), index=False)
        test_data.to_csv(path.replace(".csv", "_test.csv"), index=False)

    def write_sbc_processed_data(self, path):
        self.sbc.get_data().to_csv(path, index=False)
        self.sbc_validation.get_data().to_csv(
            path.replace(".csv", "_validation.csv"), index=False
        )
        self.sbc_ext_validation.get_data().to_csv(
            path.replace(".csv", "_ext_validation.csv"), index=False
        )


if __name__ == "__main__":
    print(os.getcwd())
    path = os.path.join(os.getcwd(), "0_mimic_preprocess") 
    mimic_data = pd.read_csv(
        os.path.join(path, "preprocessed_file/mimic_processed.csv"), header=0
    )
    preprocess_wrapper = PreprocessWrapper(
        mimic_data=mimic_data, print_logs=True, path=path
    )
    preprocess_wrapper.write_mimic_processed_data(
        r"./data/preprocessed_data/mimic_processed.csv"
    )
