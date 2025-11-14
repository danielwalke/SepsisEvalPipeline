from data.data_preprocessing.Preprocesser import Preprocesser
from data.data_preprocessing.utils.CountFunctions import count_cbc_cases, count_cbc
import pandas as pd

class PreprocessWrapper:
    def __init__(self, mimic_data = None, print_logs = False):
        if mimic_data is not None:
            mimic_validation_data = mimic_data.query("Center == 'MIMIC-IV' & Set == 'Validation'")
            self.mimic = Preprocesser(mimic_validation_data)
            self.mimic.preprocess_data()
            if not print_logs: return 
            print(20 * "$")
            print("Testing: ")
            print(f"Controls: {self.mimic.get_control_data().shape[0]},"
                  f" Sepsis: {self.mimic.get_sepsis_data().shape[0]}")
            print(f"Assessable data are {count_cbc_cases(self.mimic.get_data())} cases "
                  f"and {count_cbc(self.mimic.get_data())} CBCs")
            print(f"Control data are {count_cbc_cases(self.mimic.get_control_data())} cases "
                  f"and {count_cbc(self.mimic.get_control_data())} CBCs")
            print(f"Sepsis data are {count_cbc_cases(self.mimic.get_sepsis_data())} cases "
                  f"and {count_cbc(self.mimic.get_sepsis_data())} CBCs")        
    
    def get_X_mimic(self):
        return self.mimic.get_X()
    
    def get_y_mimic(self):
        return self.mimic.get_y()
    
    def get_mimic_data(self):
        return self.mimic.get_data()
    
if __name__ == "__main__":
    mimic_data = pd.read_csv(r"./data/raw_data/mimic_processed.csv", header=0)
    print(mimic_data.shape)
    preprocess_wrapper = PreprocessWrapper(mimic_data, print_logs=True)