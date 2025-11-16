from data.data_preprocessing.Preprocesser import Preprocesser
from data.data_preprocessing.utils.CountFunctions import count_cbc_cases, count_cbc
import pandas as pd

class PreprocessWrapper:
    def __init__(self, sbc_data = None, mimic_data = None, print_logs = False):
        if mimic_data is not None:
            mimic_validation_data = mimic_data.query("Center == 'MIMIC-IV' & Set == 'Validation'")
            self.mimic = Preprocesser(mimic_validation_data)
            self.mimic.preprocess_data()
            if print_logs: 
                  print(20 * "$")
                  print("MIMIC: ")
                  print(f"Controls: {self.mimic.get_control_data().shape[0]},"
                        f" Sepsis: {self.mimic.get_sepsis_data().shape[0]}")
                  print(f"Assessable data are {count_cbc_cases(self.mimic.get_data())} cases "
                        f"and {count_cbc(self.mimic.get_data())} CBCs")
                  print(f"Control data are {count_cbc_cases(self.mimic.get_control_data())} cases "
                        f"and {count_cbc(self.mimic.get_control_data())} CBCs")
                  print(f"Sepsis data are {count_cbc_cases(self.mimic.get_sepsis_data())} cases "
                        f"and {count_cbc(self.mimic.get_sepsis_data())} CBCs")
        if sbc_data is not None:
            sbc_training_data = sbc_data.query("Center == 'Leipzig' & Set == 'Training'")
            self.sbc = Preprocesser(sbc_training_data)
            self.sbc.preprocess_data()
            sbc_validation_data = sbc_data.query("Center == 'Leipzig' & Set == 'Validation'")
            self.sbc_validation = Preprocesser(sbc_validation_data)
            self.sbc_validation.preprocess_data()
            sbc_ext_validation_data = sbc_data.query("Center == 'Greifswald' & Set == 'Validation'")
            self.sbc_ext_validation = Preprocesser(sbc_ext_validation_data)
            self.sbc_ext_validation.preprocess_data()
            if not print_logs: return 
            print(20 * "#")
            print("SBC Training: ")
            print(f"Controls: {self.sbc.get_control_data().shape[0]},"
                  f" Sepsis: {self.sbc.get_sepsis_data().shape[0]}")
            print(f"Assessable data are {count_cbc_cases(self.sbc.get_data())} cases "
                  f"and {count_cbc(self.sbc.get_data())} CBCs")
            print(f"Control data are {count_cbc_cases(self.sbc.get_control_data())} cases "
                  f"and {count_cbc(self.sbc.get_control_data())} CBCs")
            print(f"Sepsis data are {count_cbc_cases(self.sbc.get_sepsis_data())} cases "
                  f"and {count_cbc(self.sbc.get_sepsis_data())} CBCs")
            print(20 * "#")
            print("SBC Validation: ")
            print(f"Controls: {self.sbc_validation.get_control_data().shape[0]},"
                  f" Sepsis: {self.sbc_validation.get_sepsis_data().shape[0]}")
            print(f"Assessable data are {count_cbc_cases(self.sbc_validation.get_data())} cases "
                  f"and {count_cbc(self.sbc_validation.get_data())} CBCs")  
            print(f"Control data are {count_cbc_cases(self.sbc_validation.get_control_data())} cases "
                    f"and {count_cbc(self.sbc_validation.get_control_data())} CBCs")
            print(f"Sepsis data are {count_cbc_cases(self.sbc_validation.get_sepsis_data())} cases "
                  f"and {count_cbc(self.sbc_validation.get_sepsis_data())} CBCs")
            print(20 * "#")
            print("SBC External Validation: ")
            print(f"Controls: {self.sbc_ext_validation.get_control_data().shape[0]},"
                  f" Sepsis: {self.sbc_ext_validation.get_sepsis_data().shape[0]}")
            print(f"Assessable data are {count_cbc_cases(self.sbc_ext_validation.get_data())} cases "
                    f"and {count_cbc(self.sbc_ext_validation.get_data())} CBCs")
            print(f"Control data are {count_cbc_cases(self.sbc_ext_validation.get_control_data())} cases "
                  f"and {count_cbc(self.sbc_ext_validation.get_control_data())} CBCs")
            print(f"Sepsis data are {count_cbc_cases(self.sbc_ext_validation.get_sepsis_data())} cases "
                  f"and {count_cbc(self.sbc_ext_validation.get_sepsis_data())} CBCs")
            
    
    def get_X_mimic(self):
        return self.mimic.get_X()
    
    def get_y_mimic(self):
        return self.mimic.get_y()
    
    def get_mimic_data(self):
        return self.mimic.get_data()
    
    def write_mimic_processed_data(self, path):
        self.mimic.get_data().to_csv(path, index=False)

    def write_sbc_processed_data(self, path):
        self.sbc.get_data().to_csv(path, index=False)
        self.sbc_validation.get_data().to_csv(path.replace(".csv", "_validation.csv"), index=False)
        self.sbc_ext_validation.get_data().to_csv(path.replace(".csv", "_ext_validation.csv"), index=False)
    
if __name__ == "__main__":
    mimic_data = pd.read_csv(r"./data/raw_data/mimic_processed.csv", header=0)
    sbc_data = pd.read_csv(r"./data/raw_data/sbc_processed.csv", header=0)
    print(mimic_data.shape)
    preprocess_wrapper = PreprocessWrapper(sbc_data = sbc_data, mimic_data = mimic_data, print_logs=True)
    preprocess_wrapper.write_mimic_processed_data(r"./data/preprocessed_data/mimic_processed.csv")
    preprocess_wrapper.write_sbc_processed_data(r"./data/preprocessed_data/sbc_processed.csv")