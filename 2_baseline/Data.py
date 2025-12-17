import os
import pandas as pd

class Data:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.train_X, self.train_y = self.load_train_data()
        self.val_X, self.val_y = self.load_val_data()
        self.test_X, self.test_y = self.load_test_data()
        
    def load_data(self, path):
        data_path = os.path.join(self.input_dir, path)
        data = pd.read_csv(data_path)
        y = data["y"]
        ## sample columns starting with "f__"
        X = data.filter(regex="^f__")
        return X, y
    
    def load_train_data(self):
        return self.load_data("mimic_processed_train.csv")
    
    def load_val_data(self):
        return self.load_data("mimic_processed_val.csv")
    
    def load_test_data(self):
        return self.load_data("mimic_processed_test.csv")