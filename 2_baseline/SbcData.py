import os
import pandas as pd
from sklearn.model_selection import train_test_split

class SbcData:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.full_train_X, self.full_train_y, ids = self.load_train_data()
        unique_ids = ids.unique()
        train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
        train_mask = ids.isin(train_ids)
        val_mask = ids.isin(val_ids)
        
        self.train_X = self.full_train_X[train_mask]
        self.train_y = self.full_train_y[train_mask]
        self.val_X = self.full_train_X[val_mask]
        self.val_y = self.full_train_y[val_mask]
        
        self.test_X, self.test_y, _ = self.load_test_data()
        self.ext_test_X, self.ext_test_y, _ = self.load_ext_test_data()
        self.test_data_containers = [("SBC_TEST", self.test_X, self.test_y), ("SBC_EXT_TEST", self.ext_test_X, self.ext_test_y)]
        
    def load_data(self, path):
        data_path = os.path.join(self.input_dir, path)
        data = pd.read_csv(data_path)
        y = data["y"] == "Sepsis"
        y = y.astype(int)
        ## sample columns starting with "f__"
        X = data.filter(regex="^f__")
        ids = data["Id"]
        return X, y, ids
    
    def load_train_data(self):
        return self.load_data("sbc_processed.csv")
    
    def load_val_data(self):
        return self.load_data("sbc_processed.csv")
    
    def load_test_data(self):
        return self.load_data("sbc_processed_validation.csv")
    
    def load_ext_test_data(self):
        return self.load_data("sbc_processed_ext_validation.csv")
    
    @property
    def name(self):
        return "SBC"