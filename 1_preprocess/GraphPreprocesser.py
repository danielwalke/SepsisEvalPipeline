import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from constants.feature_names import FEATURES

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
    
    def write_edges(self, edge_idx_path, edge_weight_path):
        edge_index, edge_weight = self.get_edges()
        torch.save(edge_index, edge_idx_path)
        torch.save(edge_weight, edge_weight_path)
        print(f"Edge index saved to {edge_idx_path}")
        print(f"Edge weight saved to {edge_weight_path}")

    def get_pos_encoding(self, seq_len, d = len(FEATURES), n=10000):
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
        pos_encodings = self.get_pos_encodings()
        pos_encodings = torch.tensor(pos_encodings, dtype=torch.float)
        torch.save(pos_encodings, path)
        print(f"Positional encodings saved to {path}")

if __name__ == "__main__":
    mimic_preprocessed_data = pd.read_csv(r"./data/preprocessed_data/mimic_processed.csv", header=0)    
    graph_preprocesser = GraphPreprocesser(mimic_preprocessed_data) 
    graph_preprocesser.sort_data()
    train_ids, test_ids = train_test_split(graph_preprocesser.data["Id"].unique(), test_size=0.2, random_state=42)
    test_data = graph_preprocesser.data[graph_preprocesser.data["Id"].isin(test_ids)].copy()
    graph_preprocesser.data = graph_preprocesser.data[graph_preprocesser.data["Id"].isin(train_ids)]
    graph_preprocesser.sort_data()
    graph_preprocesser.data.to_csv(r"./data/graph_data/mimic_sorted_processed.csv", index=False)
    graph_preprocesser.write_edges(r"./data/graph_data/mimic_edge_index.pt",
                                   r"./data/graph_data/mimic_edge_weight.pt")
    graph_preprocesser.write_pos_encodings(r"./data/graph_data/mimic_pos_encodings.pt")
    graph_preprocesser.data = test_data
    graph_preprocesser.sort_data()
    graph_preprocesser.data.to_csv(r"./data/graph_data/mimic_test_sorted_processed.csv", index=False)
    graph_preprocesser.write_edges(r"./data/graph_data/mimic_test_edge_index.pt",
                                   r"./data/graph_data/mimic_test_edge_weight.pt")
    graph_preprocesser.write_pos_encodings(r"./data/graph_data/mimic_test_pos_encodings.pt")

    sbc_preprocessed_data = pd.read_csv(r"./data/preprocessed_data/sbc_processed.csv", header=0)
    graph_preprocesser = GraphPreprocesser(sbc_preprocessed_data)
    graph_preprocesser.sort_data()
    graph_preprocesser.data.to_csv(r"./data/graph_data/sbc_sorted_processed.csv", index=False)
    graph_preprocesser.write_edges(r"./data/graph_data/sbc_edge_index.pt",
                                   r"./data/graph_data/sbc_edge_weight.pt")
    graph_preprocesser.write_pos_encodings(r"./data/graph_data/sbc_pos_encodings.pt")
    
    sbc_validation_data = pd.read_csv(r"./data/preprocessed_data/sbc_processed_validation.csv", header=0)
    graph_preprocesser = GraphPreprocesser(sbc_validation_data)
    graph_preprocesser.sort_data()
    graph_preprocesser.data.to_csv(r"./data/graph_data/sbc_validation_sorted_processed.csv", index=False)
    graph_preprocesser.write_edges(r"./data/graph_data/sbc_validation_edge_index.pt",
                                   r"./data/graph_data/sbc_validation_edge_weight.pt")
    graph_preprocesser.write_pos_encodings(r"./data/graph_data/sbc_validation_pos_encodings.pt")
    
    sbc_ext_validation_data = pd.read_csv(r"./data/preprocessed_data/sbc_processed_ext_validation.csv", header=0)
    graph_preprocesser = GraphPreprocesser(sbc_ext_validation_data)
    graph_preprocesser.sort_data()
    graph_preprocesser.data.to_csv(r"./data/graph_data/sbc_ext_validation_sorted_processed.csv", index=False)
    graph_preprocesser.write_edges(r"./data/graph_data/sbc_ext_validation_edge_index.pt",
                                   r"./data/graph_data/sbc_ext_validation_edge_weight.pt")
    graph_preprocesser.write_pos_encodings(r"./data/graph_data/sbc_ext_validation_pos_encodings.pt")