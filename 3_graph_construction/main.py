import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import os

class GraphPreprocesser:
    def __init__(self, data):
        self.data = data
        print("GraphPreprocesser initialized.")
        print(f"Data shape: {self.data.shape}")
        # Additional preprocessing steps would go here

    def sort_data(self):
        ## TODO in future create additional edges based on previous admission of the same patients Id is here based on hadm_id but the same patient can have different Ids when admitted multiple time
        ## More difficult than you think -> Consider data leakage when only using boolean masks during training
        self.data = self.data.sort_values(by=['Id', 'Time'])
        print("Data sorted by Subject ID and Timestamp.")
    
    def get_edges(self):
        dataset = self.data.copy()
        dataset = dataset.reset_index(drop=True)
        source_edge_index = []
        target_edge_index = []
        edge_weights = []

        for Id, group in tqdm(dataset.groupby("Id")):
            indices = group.index
            offset = indices[0]
            num_nodes = len(indices)
            edge_index = torch.zeros((2, sum(range(num_nodes + 1))), dtype=torch.long)+offset
            ## Self edges
            edge_index[:, 0:num_nodes] = (torch.arange(num_nodes) + offset).view(1, -1)
            idx = num_nodes
            for i in range(1, num_nodes):
                edge_index[1, idx:idx + i] = i+offset
                edge_index[0, idx:idx + i] = torch.arange(i)+offset
                idx += i
            src_idc = edge_index[0, :] - offset
            trt_idc = edge_index[1, :] - offset
            group_time = np.expand_dims(group["Time"].values, 0) if group["Time"].values.shape[0] <= 1 else (group["Time"].values - group["Time"].values.min()) / (group["Time"].values.max() - group["Time"].values.min())
            #group_time = stable_softmax(group_time) # torch.nn.functional.softmax(torch.from_numpy(group["Time"].values))
            time_diff = 1 - (group_time[trt_idc] - group_time[src_idc])
            source_edge_index.extend(edge_index[0, :].numpy().tolist())
            target_edge_index.extend(edge_index[1, :].numpy().tolist())
            edge_weights.extend(time_diff.tolist())

        edge_index = np.asarray([np.asarray(source_edge_index), np.asarray(target_edge_index)])
        edge_index = torch.tensor(edge_index)
        edge_weight = torch.tensor(edge_weights)
        return edge_index, edge_weight
    
    def write_edges(self, edge_path, include_weights=True):
        if os.path.exists(edge_path):
            print(f"Edges file {edge_path} already exists. Skipping edge writing.")
            return
        edge_index, edge_weight = self.get_edges()
        edge_index_df = pd.DataFrame(edge_index.numpy().transpose(), columns=["source", "target"])
        if not include_weights:
            edge_index_df.to_csv(edge_path, index=False)
            print(f"Edges saved to {edge_path}")
            return
        edge_weight_df = pd.DataFrame(edge_weight.numpy().transpose(), columns=["weight"])
        pd.concat([edge_index_df, edge_weight_df], axis=1).to_csv(edge_path, index=False)
        print(f"Edges saved to {edge_path}")

    def get_pos_encoding(self, seq_len, n=10000):
        d = self.data.filter(regex="^f__").shape[1]
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P
    
    def get_pos_encodings(self):
        dataset = self.data.copy()
        dataset = dataset.reset_index(drop=True)
        pos_encodings = None
        for Id, group in tqdm(dataset.groupby("Id")):
            encoding = self.get_pos_encoding(group.shape[0])
            pos_encodings = encoding if pos_encodings is None else np.concatenate((pos_encodings, encoding), axis=0)
        return pos_encodings
    
    def write_pos_encodings(self, path):
        if os.path.exists(path):
            print(f"Positional encodings file {path} already exists. Skipping positional encoding writing.")
            return
        pos_encodings = self.get_pos_encodings()
        pos_encodings = torch.tensor(pos_encodings, dtype=torch.float)
        df = pd.DataFrame({
            "pos_encodings": [str(x.tolist()).replace(" ", "") for x in pos_encodings.numpy()]
        })
        df.index.name = "idx"
        df.to_csv(path, index=True)
        print(f"Positional encodings saved to {path}")
        
    def write_nodes(self, path):
        if os.path.exists(path):
            print(f"Nodes file {path} already exists. Skipping node writing.")
            return
        dataset = self.data.copy()
        y = dataset["y"]
        X_df = dataset.filter(regex="^f__")
        
        df = pd.DataFrame({
            "X": [str(x.tolist()).replace(" ", "") for x in X_df.to_numpy()],
            "y": y
        })
        df.index.name = "idx"
        df.to_csv(path, index=True)
        print(f"Sorted graph nodes saved to {path}")

if __name__ == "__main__":
    data_input_dir = "/app/input"
    data_output_dir = "/app/output"
    
    for split in ["train", "val", "test"]:
        mimic_preprocessed_data = pd.read_csv(f"{data_input_dir}/mimic_processed_{split}.csv", header=0)
        graph_preprocesser = GraphPreprocesser(mimic_preprocessed_data) 
        graph_preprocesser.sort_data()
        graph_preprocesser.write_edges(f"{data_output_dir}/mimic_{split}_edges.csv")
        graph_preprocesser.write_pos_encodings(f"{data_output_dir}/mimic_{split}_pos_encodings.csv")
        graph_preprocesser.write_nodes(f"{data_output_dir}/mimic_{split}_nodes.csv")
        
    for split in ["", "_validation", "_ext_validation"]:
        sbc_preprocessed_data = pd.read_csv(f"{data_input_dir}/sbc_processed{split}.csv", header=0)
        graph_preprocesser = GraphPreprocesser(sbc_preprocessed_data) 
        graph_preprocesser.sort_data()
        graph_preprocesser.write_edges(f"{data_output_dir}/sbc{split}_edges.csv", include_weights=False)
        # graph_preprocesser.write_pos_encodings(f"{data_output_dir}/sbc{split}_pos_encodings.csv")
        graph_preprocesser.write_nodes(f"{data_output_dir}/sbc{split}_nodes.csv")
    
