
import torch
from torch_geometric.loader import NeighborLoader
import pandas as pd
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from constants.feature_names import FEATURES
from gnn_evaluation.utils.SeedInitialization import seed_worker

class GraphLoaderSBC:
    def __init__(self):
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_graph = None
        self.test_graph = None
        self.train_edge_index = None
        self.train_edge_weight = None
        self.test_edge_index = None
        self.test_edge_weight = None
        self.train_pos_encodings = None
        self.test_pos_encodings = None
        self.train_df = None
        self.test_df = None
        self.train_labels = None
        self.test_labels = None

    def load_nodes(self):
        self.train_df = pd.read_csv("./data/graph_data/sbc_sorted_processed.csv", header=0)
        self.train_labels = self.train_df["Label"].astype('category').cat.codes.astype(float)
        self.test_df = pd.read_csv("./data/graph_data/sbc_validation_sorted_processed.csv", header=0)
        self.test_labels = self.test_df["Label"].astype('category').cat.codes.astype(float)

    def load_edges(self):
        self.train_edge_index = torch.load("./data/graph_data/sbc_edge_index.pt") 
        self.train_edge_weight = torch.load("./data/graph_data/sbc_edge_weight.pt")
        self.test_edge_index = torch.load("./data/graph_data/sbc_validation_edge_index.pt") 
        self.test_edge_weight = torch.load("./data/graph_data/sbc_validation_edge_weight.pt")

    def load_pos_encodings(self):
        train_pos_encodings = torch.load("./data/graph_data/sbc_pos_encodings.pt")
        test_pos_encodings = torch.load("./data/graph_data/sbc_validation_pos_encodings.pt")
        self.train_pos_encodings = train_pos_encodings
        self.test_pos_encodings = test_pos_encodings

    def scale_features(self):
        scaler = StandardScaler()
        self.train_df[FEATURES] = scaler.fit_transform(self.train_df[FEATURES])
        self.test_df[FEATURES] = scaler.transform(self.test_df[FEATURES])   

    def construct_graphs(self):
        self.train_graph = Data(x = torch.tensor(self.train_df[FEATURES].values, dtype=torch.float),
                            edge_index = self.train_edge_index.long(),
                            edge_attr = self.train_edge_weight.float(),
                            y = torch.tensor(self.train_labels, dtype=torch.float).unsqueeze(1))
        self.test_graph = Data(x = torch.tensor(self.test_df[FEATURES].values, dtype=torch.float),
                            edge_index = self.test_edge_index.long(),
                            edge_attr = self.test_edge_weight.float(),
                            y = torch.tensor(self.test_labels, dtype=torch.float).unsqueeze(1))

    def construct_train_val_test_masks(self):
        train_ids, val_ids = train_test_split(self.train_df["Id"].unique(), test_size=0.2, random_state=42)
        train_mask = self.train_df["Id"].isin(train_ids)
        val_mask = self.train_df["Id"].isin(val_ids)
        self.train_graph.train_mask = torch.tensor(train_mask.values, dtype=torch.bool)
        self.train_graph.test_mask = torch.tensor(val_mask.values, dtype=torch.bool) ## just naming convention to make the mask compatible for evaluation function in ModelWrapper
        self.test_graph.test_mask = torch.tensor([True]*self.test_graph.num_nodes, dtype=torch.bool) 
    
    def add_pos_encodings(self):
        self.train_graph.x = self.train_graph.x + self.train_pos_encodings
        self.test_graph.x = self.test_graph.x + self.test_pos_encodings

    def initialize_graphs(self):
        self.load_edges()
        self.load_pos_encodings()
        self.load_nodes()

        self.scale_features()

        self.construct_graphs()
        self.construct_train_val_test_masks()
        self.add_pos_encodings()
    
    def initialize(self, batch_size):
        self.initialize_graphs()
        g = torch.Generator()
        g.manual_seed(42)
        self.train_loader = NeighborLoader(self.train_graph, num_neighbors=[-1] * 2, batch_size=batch_size, input_nodes=self.train_graph.train_mask, generator=g, worker_init_fn=seed_worker)
        self.val_loader = NeighborLoader(self.train_graph, num_neighbors=[-1] * 2, batch_size=batch_size, input_nodes=self.train_graph.test_mask, generator=g, worker_init_fn=seed_worker)
        self.test_loader = NeighborLoader(self.test_graph, num_neighbors=[-1] * 2, batch_size=batch_size, input_nodes=self.test_graph.test_mask, generator=g, worker_init_fn=seed_worker)