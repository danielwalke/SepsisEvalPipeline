import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import os
import configparser

class GraphPreprocesser:
    def __init__(self, data):
        self.data = data
        print("GraphPreprocesser initialized.")
        print(f"Data shape: {self.data.shape}")

    def sort_data(self):
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
            
            edge_index[:, 0:num_nodes] = (torch.arange(num_nodes) + offset).view(1, -1)
            idx = num_nodes
            for i in range(1, num_nodes):
                edge_index[1, idx:idx + i] = i+offset
                edge_index[0, idx:idx + i] = torch.arange(i)+offset
                idx += i
            src_idc = edge_index[0, :] - offset
            trt_idc = edge_index[1, :] - offset
            group_time = np.expand_dims(group["Time"].values, 0) if group["Time"].values.shape[0] <= 1 else (group["Time"].values - group["Time"].values.min()) / (group["Time"].values.max() - group["Time"].values.min())
            
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
            edges_df = pd.read_csv(edge_path)
            source_edge = edges_df["source"].values
            target_edge = edges_df["target"].values
            edge_index = torch.tensor(np.array([source_edge, target_edge]))
            return edge_index

        edge_index, edge_weight = self.get_edges()
        edge_index_df = pd.DataFrame(edge_index.numpy().transpose(), columns=["source", "target"])
        if not include_weights:
            edge_index_df.to_csv(edge_path, index=False)
            print(f"Edges saved to {edge_path}")
            return edge_index
            
        edge_weight_df = pd.DataFrame(edge_weight.numpy().transpose(), columns=["weight"])
        pd.concat([edge_index_df, edge_weight_df], axis=1).to_csv(edge_path, index=False)
        print(f"Edges saved to {edge_path}")
        return edge_index

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
            "y": y,
            "patientId": dataset["Id"],
            "time": dataset["Time"],
            "hadmId": dataset.get("hadm_id")
        })
        df.index.name = "idx"
        df.to_csv(path, index=True)
        print(f"Sorted graph nodes saved to {path}")

    def extract_metrics(self, edge_index):
        num_nodes = self.data.shape[0]
        num_edges = edge_index.shape[1]
        num_unique_ids = self.data['Id'].nunique()
        
        sources = edge_index[0].numpy()
        targets = edge_index[1].numpy()
        
        in_degrees = np.bincount(targets, minlength=num_nodes)
        
        avg_in_degree = float(np.mean(in_degrees))
        max_in_degree = int(np.max(in_degrees))
        min_in_degree = int(np.min(in_degrees))
        median_in_degree = float(np.median(in_degrees))
        
        return {
            "Number of nodes": num_nodes,
            "Number of edges": num_edges,
            "Avg. in-degree": avg_in_degree,
            "Max. in-degree": max_in_degree,
            "Min. in-degree": min_in_degree,
            "Median in-degree": median_in_degree,
            "Number of unique IDs": num_unique_ids
        }

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('/app/config/config.ini')
    panel_name = config["PANEL"]["panel_name"]

    data_input_dir = "/app/input"
    data_output_dir = "/app/output"
    metrics_dir = "/app/metrics"
    
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"{panel_name}.csv")
    
    metrics_list = []
    
    for split in ["train", "val", "test"]:
        if not os.path.exists(f"{data_input_dir}/mimic_processed_{split}.csv"):
            print(f"Input file {data_input_dir}/mimic_processed_{split}.csv not found. Skipping {split} split.")
            continue
        mimic_preprocessed_data = pd.read_csv(f"{data_input_dir}/mimic_processed_{split}.csv", header=0)
        graph_preprocesser = GraphPreprocesser(mimic_preprocessed_data) 
        graph_preprocesser.sort_data()
        
        edge_index = graph_preprocesser.write_edges(f"{data_output_dir}/mimic_{split}_edges.csv")
        graph_preprocesser.write_pos_encodings(f"{data_output_dir}/mimic_{split}_pos_encodings.csv")
        graph_preprocesser.write_nodes(f"{data_output_dir}/mimic_{split}_nodes.csv")
        
        split_metrics = graph_preprocesser.extract_metrics(edge_index)
        split_metrics["Dataset"] = f"mimic_{split}"
        metrics_list.append(split_metrics)
        
    for split in ["", "_validation", "_ext_validation"]:
        if not os.path.exists(f"{data_input_dir}/sbc_processed{split}.csv"):
            print(f"Input file {data_input_dir}/sbc_processed{split}.csv not found. Skipping {split} split.")
            continue
        sbc_preprocessed_data = pd.read_csv(f"{data_input_dir}/sbc_processed{split}.csv", header=0)
        graph_preprocesser = GraphPreprocesser(sbc_preprocessed_data) 
        graph_preprocesser.sort_data()
        
        edge_index = graph_preprocesser.write_edges(f"{data_output_dir}/sbc{split}_edges.csv", include_weights=True)
        graph_preprocesser.write_pos_encodings(f"{data_output_dir}/sbc{split}_pos_encodings.csv")
        graph_preprocesser.write_nodes(f"{data_output_dir}/sbc{split}_nodes.csv")
        
        split_metrics = graph_preprocesser.extract_metrics(edge_index)
        split_metrics["Dataset"] = f"sbc{split}"
        metrics_list.append(split_metrics)
        
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        cols = ["Dataset"] + [c for c in metrics_df.columns if c != "Dataset"]
        metrics_df = metrics_df[cols]
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved successfully to {metrics_path}")